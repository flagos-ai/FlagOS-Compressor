import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from flagos_compressor.core.validation import validate_artifact


def make_source(source):
    source.mkdir()
    mapping = {}
    for layer in range(2):
        state = {}
        for proj in ("w1", "w2", "w3"):
            base = f"model.layers.{layer}.mlp.experts.0.{proj}"
            state[base + ".weight_packed"] = torch.full((2, 16), 0x21, dtype=torch.uint8)
            state[base + ".weight_scale"] = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
            state[base + ".weight_global_scale"] = torch.tensor(2.)
        shard = f"model-{layer + 1:05d}-of-00002.safetensors"
        save_file(state, str(source / shard))
        mapping.update({name: shard for name in state})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (source / "config.json").write_text("{}")


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="requires two CUDA GPUs")
@pytest.mark.parametrize("exclude_layer", [False, True])
def test_parallel_shards_merge_into_one_valid_artifact(tmp_path, exclude_layer):
    source, output, work = (tmp_path / p for p in ("source", "output", "work"))
    make_source(source)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / "examples/parallel_w8a8.py"),
        "--input", str(source), "--output", str(output), "--work-dir", str(work),
        "--gpus", "0,1", "--n-candidates", "8",
    ] + (["--exclude-name", r"model\.layers\.1\."] if exclude_layer else []),
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    assert validation["int8_tensors"] == (3 if exclude_layer else 6)
    assert not (output / "_INCOMPLETE").exists()
    report = json.loads((output / "quantization_report.json").read_text())
    assert report["converted"] == 6
    assert report["skipped_scales"] == 12
    assert all(op["fallback"] == 0 for op in report["ops"].values())


def test_worker_exports_shard_with_only_bf16_actions(tmp_path, monkeypatch):
    from flagos_compressor.backends.registry import build_backend
    from flagos_compressor.core.planner import build_quantize_plan
    from flagos_compressor.core.policy import QuantizationPolicy
    from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors
    from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint

    spec = importlib.util.spec_from_file_location(
        "parallel_w8a8", Path(__file__).resolve().parents[1] / "examples/parallel_w8a8.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    source, output, work = (tmp_path / p for p in ("source", "output", "work"))
    make_source(source)
    output.mkdir()
    profile = scan_hf_safetensors(source)
    plan = build_quantize_plan(profile, QuantizationPolicy(
        selections=("linear",), exclude_names=(r"model\.layers\.1\.",),
        num_bits=8, activation_num_bits=8, strategy="channel", n_candidates=8,
    ))
    monkeypatch.setattr(worker, "PROFILE", profile, raising=False)
    monkeypatch.setattr(worker, "PLAN", plan, raising=False)
    monkeypatch.setattr(worker, "DEVICE", 0, raising=False)
    monkeypatch.setattr(worker, "build_backend", lambda *a, **kw: build_backend("cpu"))
    checkpoint = HfSafetensorsCheckpoint(source)
    mapping, size = {}, 0
    for shard in checkpoint.shard_files():
        result = worker.process_shard(shard, source, output, work)
        mapping.update(result["index"]["weight_map"])
        size += result["index"]["metadata"]["total_size"]
    checkpoint.copy_auxiliary_files(output)
    checkpoint.write_index(output, mapping, total_size=size)
    config = worker._patch_compressed_tensors_config(output, plan)
    worker._write_quantization_manifest(output, plan, config)
    result = validate_artifact(output)
    assert result["valid"], result["errors"]
    assert result["int8_tensors"] == 3
    bf16 = load_file(output / "model-00002-of-00002.safetensors")
    assert len(bf16) == 3
    for weight in bf16.values():
        assert weight.dtype == torch.bfloat16
        torch.testing.assert_close(
            weight.float(), torch.tensor([.25, .5]).repeat(16).expand(2, -1)
        )
