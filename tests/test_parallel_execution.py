import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from flagos_compressor.cli.main import main
from flagos_compressor.core.parallel import normalize_devices
from flagos_compressor.core.validation import validate_artifact


def make_fused_source(path):
    path.mkdir()
    generator = torch.Generator().manual_seed(13)
    def weight(*shape):
        return torch.randn(*shape, generator=generator).bfloat16()
    shards = {
        "a.safetensors": {"model.layers.0.mlp.experts.gate_up_proj": weight(2, 64, 64)},
        "b.safetensors": {"model.layers.0.mlp.experts.down_proj": weight(2, 64, 32)},
        "c.safetensors": {"model.layers.0.self_attn.q_proj.weight": weight(64, 64)},
        "d.safetensors": {"vit.layers.0.self_attn.q_proj.weight": weight(32, 64),
                          "model.norm.weight": torch.ones(64).bfloat16()},
        "e.safetensors": {"model.layers.1.mlp.gate_proj.weight": weight(32, 64)},
        "f.safetensors": {"model.layers.1.mlp.up_proj.weight": weight(32, 64)},
    }
    for shard, state in shards.items():
        save_file(state, path / shard)
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {}, "weight_map": {k: shard for shard, values in shards.items() for k in values},
    }))
    (path / "config.json").write_text(json.dumps({"model_type": "unknown_fused_moe"}))
    (path / "tokenizer_config.json").write_text('{"fixture": true}')


def assert_same_artifact(serial, parallel):
    validation = validate_artifact(parallel)
    assert validation["valid"], validation["errors"]
    assert not (parallel / "_INCOMPLETE").exists()
    for name in ("config.json", "model.safetensors.index.json", "quantization_manifest.json"):
        if (serial / name).exists():
            assert json.loads((serial / name).read_text()) == json.loads((parallel / name).read_text())
    for path in serial.glob("*.safetensors"):
        expected, actual = load_file(path), load_file(parallel / path.name)
        assert actual.keys() == expected.keys()
        for name, tensor in expected.items():
            assert torch.equal(tensor, actual[name]), name


SCHEMES = {
    "w8a8": ["--select", "linear", "--bits", "8", "--activation-bits", "8", "--strategy", "channel"],
    "w8a8-bf16-scale": ["--select", "linear", "--bits", "8", "--activation-bits", "8", "--strategy", "channel", "--scale-dtype", "bf16"],
    "w8a16": ["--select", "linear", "--bits", "8", "--group-size", "32"],
    "w4a16": ["--select", "linear", "--bits", "4", "--group-size", "32"],
    "w8a16-channel": ["--select", "linear", "--exclude", "moe.routed", "--bits", "8", "--strategy", "channel"],
    "mixed": ["--select", "moe=int4", "activation-bits=16", "strategy=group", "group-size=32",
              "--select", "attention=int8", "activation-bits=8", "strategy=channel"],
}


@pytest.fixture(params=["cpu", "cuda"])
def backend(request):
    if request.param == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
        pytest.skip("requires two CUDA devices")
    return request.param


@pytest.mark.parametrize("scheme", SCHEMES)
def test_parallel_cli_matches_serial_with_fused_banks(tmp_path, backend, scheme):
    source, serial, parallel = (tmp_path / p for p in ("source", "serial", "parallel"))
    make_fused_source(source)
    args = ["quantize", "--input", str(source), "--backend", backend, "--n-candidates", "8",
            "--exclude-name", r"^vit\.", *SCHEMES[scheme]]
    main([*args, "--output", str(serial)])
    main([*args, "--output", str(parallel), "--devices", "0,1"])
    assert_same_artifact(serial, parallel)
    assert (parallel / "tokenizer_config.json").read_bytes() == (source / "tokenizer_config.json").read_bytes()
    report = json.loads((parallel / "quantization_report.json").read_text())
    assert report["extras"]["parallel"]["workers"] == 2
    assert sum(x["shards"] for x in report["extras"]["parallel"]["per_device"].values()) == 6
    assert report["extras"]["validation"]["valid"]
    assert all(not e["fallback"] and e["actual_backend"] == backend for e in report["op_events"])


@pytest.mark.parametrize("command", ["quantize", "convert"])
def test_parallel_reads_cross_shard_microscaling_dependencies(tmp_path, backend, command):
    from test_microscaling_inputs import make_source
    source, serial, parallel = (tmp_path / p for p in ("source", "serial", "parallel"))
    make_source(source)
    args = [command, "--input", str(source), "--backend", backend]
    if command == "quantize":
        args += [*SCHEMES["w8a8"], "--n-candidates", "8"]
    main([*args, "--output", str(serial)])
    main([*args, "--output", str(parallel), "--devices", "0", "1"])
    assert_same_artifact(serial, parallel)


def test_parallel_uses_recipe_and_preserves_unselected_source_formats(tmp_path):
    from test_microscaling_inputs import make_source
    source, serial, parallel = (tmp_path / p for p in ("source", "serial", "parallel"))
    make_source(source)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("""version: 1
bits: 8
activation_bits: 8
strategy: channel
n_candidates: 8
select: [attention]
unselected:
  strategy: preserve
""")
    args = ["quantize", "--input", str(source), "--recipe", str(recipe)]
    main([*args, "--output", str(serial)])
    main([*args, "--output", str(parallel), "--devices", "cpu:0", "cpu:1"])
    assert_same_artifact(serial, parallel)


def test_parallel_failure_leaves_incomplete_and_joins_workers(tmp_path):
    source, output = tmp_path / "source", tmp_path / "failed"
    make_fused_source(source)
    shard = source / "a.safetensors"
    state = load_file(shard)
    next(iter(state.values())).fill_(float("nan"))
    save_file(state, shard)
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        main(["quantize", "--input", str(source), "--output", str(output),
              *SCHEMES["w8a8"], "--n-candidates", "8", "--devices", "cpu:0", "cpu:1"])
    assert {p.pid for p in mp.active_children()} == before
    assert (output / "_INCOMPLETE").exists()
    assert (output / "execution_error.json").exists()
    assert not validate_artifact(output)["valid"]
    assert not (output / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
def test_calibrated_parallel_is_rejected_before_loading_model(tmp_path, method):
    with pytest.raises(ValueError, match="sequential calibration"):
        main(["quantize", "--input", "missing-model", "--output", str(tmp_path / "out"),
              "--select", "linear", "--method", method, "--devices", "cpu:0", "cpu:1",
              "--dry-run"])
    assert not (tmp_path / "out").exists()


def test_parallel_dry_run_does_not_require_devices_or_create_output(tmp_path, monkeypatch, capsys):
    import flagos_compressor.core.parallel as parallel_module
    source, output = tmp_path / "source", tmp_path / "out"
    make_fused_source(source)
    monkeypatch.setattr(parallel_module, "_check_device", lambda *args: pytest.fail("touched hardware"))
    main(["quantize", "--input", str(source), "--output", str(output), *SCHEMES["w8a8"],
          "--backend", "cuda", "--devices", "0,1", "--dry-run"])
    assert "parallel workers: 2" in capsys.readouterr().out
    assert not output.exists()


def test_parallel_refuses_nonempty_output(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    make_fused_source(source)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="empty directory"):
        main(["quantize", "--input", str(source), "--output", str(output),
              *SCHEMES["w8a8"], "--devices", "cpu:0", "cpu:1"])
    assert sentinel.read_text() == "keep"
    assert not (output / "_INCOMPLETE").exists()


def test_single_shard_limits_worker_count(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    save_file({"model.layers.0.self_attn.q_proj.weight": torch.ones(32, 32).bfloat16()},
              source / "model.safetensors")
    (source / "config.json").write_text("{}")
    main(["quantize", "--input", str(source), "--output", str(output),
          *SCHEMES["w8a8"], "--n-candidates", "8", "--devices", "cpu:0", "cpu:1"])
    report = json.loads((output / "quantization_report.json").read_text())
    assert report["extras"]["parallel"]["workers"] == 1
    assert validate_artifact(output)["valid"]


@pytest.mark.parametrize("devices", [["0,0"], ["cuda", "cuda:0"], ["0,"]])
def test_duplicate_and_empty_devices_are_rejected(devices):
    with pytest.raises(ValueError):
        normalize_devices("cuda", devices)


def test_device_backend_mismatch_and_cli_mutual_exclusion():
    with pytest.raises(ValueError, match="does not match"):
        normalize_devices("cuda", ["cpu:0"])
    with pytest.raises(SystemExit):
        main(["quantize", "--input", "unused", "--output", "unused", "--device", "cuda:0",
              "--devices", "cuda:1", "--dry-run"])


def abrupt_worker(*args):
    os._exit(17)


def source_mutating_worker(*args):
    from flagos_compressor.core.parallel import _worker
    _worker(*args)
    (Path(args[0]) / "config.json").write_text('{"model_type": "changed"}')


@pytest.mark.parametrize("worker, message", [
    (abrupt_worker, "exited before completion"),
    (source_mutating_worker, "Source checkpoint changed"),
])
def test_worker_crash_and_source_mutation_cannot_publish(tmp_path, monkeypatch, worker, message):
    import flagos_compressor.core.parallel as parallel_module
    source, output = tmp_path / "source", tmp_path / "out"
    make_fused_source(source)
    monkeypatch.setattr(parallel_module, "_worker", worker)
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match=message):
        main(["quantize", "--input", str(source), "--output", str(output),
              *SCHEMES["w8a8"], "--n-candidates", "8", "--devices", "cpu:0", "cpu:1"])
    assert {p.pid for p in mp.active_children()} == before
    assert (output / "_INCOMPLETE").exists()
    assert not (output / "model.safetensors.index.json").exists()


def test_unavailable_parallel_device_fails_before_output_creation(tmp_path, monkeypatch):
    import flagos_compressor.core.parallel as parallel_module
    source, output = tmp_path / "source", tmp_path / "out"
    make_fused_source(source)
    def unavailable(*args):
        raise RuntimeError("Parallel device is unavailable")
    monkeypatch.setattr(parallel_module, "_check_device", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        main(["quantize", "--input", str(source), "--output", str(output),
              *SCHEMES["w8a8"], "--devices", "cpu:0", "cpu:1"])
    assert not output.exists()
