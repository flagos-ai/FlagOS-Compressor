import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.cli.main import main
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.planner import build_convert_plan, build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


def _make_checkpoint(path: Path) -> dict[str, str]:
    path.mkdir()
    routed = "model.layers.0.mlp.experts.0.w1.weight"
    shared = "model.layers.0.mlp.shared_experts.w1.weight"
    attention = "model.layers.0.self_attn.o_proj.weight"
    state = {
        routed: torch.full((2, 16), 0x21, dtype=torch.uint8),
        routed[: -len(".weight")] + ".scale": torch.ones((2, 1), dtype=torch.float32),
        shared: torch.full((2, 16), 0x43, dtype=torch.uint8),
        shared[: -len(".weight")] + ".scale": torch.ones((2, 1), dtype=torch.float32),
        attention: torch.ones((128, 128), dtype=torch.float8_e4m3fn),
        attention[: -len(".weight")] + ".scale": torch.ones((1, 1), dtype=torch.float32),
        "model.embed_tokens.weight": torch.ones((4, 32), dtype=torch.bfloat16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(state, str(path / shard))
    weight_map = {name: shard for name in state}
    total_size = sum(t.numel() * t.element_size() for t in state.values())
    with (path / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)
    with (path / "config.json").open("w") as f:
        json.dump(
            {
                "torch_dtype": "float8_e4m3fn",
                "expert_dtype": "fp4",
                "quantization_config": {"quant_method": "fp8"},
            },
            f,
        )
    return weight_map


def _make_deepseek_v4_mixed_checkpoint(path: Path) -> None:
    """Small checkpoint with the same source-format split as DeepSeek-V4."""
    path.mkdir()
    state: dict[str, torch.Tensor] = {}

    for projection in ("w1", "w2", "w3"):
        name = f"layers.0.ffn.experts.0.{projection}.weight"
        state[name] = torch.full((2, 16), 0x21, dtype=torch.int8)
        state[name[: -len('.weight')] + ".scale"] = torch.ones(
            (2, 1), dtype=torch.float32
        )

    fp8_names = [
        *(f"layers.0.ffn.shared_experts.{proj}.weight" for proj in ("w1", "w2", "w3")),
        *(f"layers.0.attn.{proj}.weight" for proj in ("wq_a", "wkv", "wq_b", "wo_a", "wo_b")),
        "mtp.0.e_proj.weight",
        "mtp.0.h_proj.weight",
    ]
    for name in fp8_names:
        state[name] = torch.ones((128, 128), dtype=torch.float8_e4m3fn)
        state[name[: -len('.weight')] + ".scale"] = torch.ones(
            (1, 1), dtype=torch.float32
        )

    for projection in ("wkv", "wgate"):
        state[f"layers.0.attn.compressor.{projection}.weight"] = torch.ones(
            (128, 128), dtype=torch.bfloat16
        )
    shard = "model-00001-of-00001.safetensors"
    save_file(state, str(path / shard))
    with (path / "model.safetensors.index.json").open("w") as f:
        json.dump(
            {
                "metadata": {
                    "total_size": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in state.values()
                    )
                },
                "weight_map": {name: shard for name in state},
            },
            f,
        )
    with (path / "config.json").open("w") as f:
        json.dump(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "torch_dtype": "bfloat16",
                "expert_dtype": "fp4",
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                },
            },
            f,
        )


def test_quantize_moe_end_to_end(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "quantized"
    _make_checkpoint(source)
    with (source / "quant_manifest.json").open("w") as f:
        json.dump({"schema": "legacy"}, f)
    profile = scan_hf_safetensors(source)
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(selections=("moe",), n_candidates=8, chunk_size=16),
    )
    execute_plan(source, output, plan, build_backend("cpu"))

    with (output / "quantization_manifest.json").open() as f:
        manifest = json.load(f)
    assert manifest["schema"] == "flagos-compressor.provenance.v1"
    assert manifest["producer"]["name"] == "FlagOS-Compressor"
    routed = "model.layers.0.mlp.experts.0.w1.weight"
    routed_packed = routed.replace(".weight", ".weight_packed")
    routed_scale = routed.replace(".weight", ".weight_scale")
    routed_shape = routed.replace(".weight", ".weight_shape")
    spec = manifest["tensors"][routed_packed]
    assert spec["logical_shape"] == [2, 32]
    assert spec["storage_shape"] == [2, 4]
    assert spec["scale_shape"] == [2, 1]

    shard = load_file(output / "model-00001-of-00001.safetensors")
    assert shard[routed_packed].dtype == torch.int32
    assert shard[routed_scale].dtype == torch.bfloat16
    assert torch.equal(shard[routed_shape], torch.tensor([2, 32]))
    assert shard["model.layers.0.self_attn.o_proj.weight"].dtype == torch.bfloat16
    with (output / "config.json").open() as f:
        config = json.load(f)
    assert config["torch_dtype"] == "bfloat16"
    assert "expert_dtype" not in config
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"
    assert config["quantization_config"]["format"] == "pack-quantized"
    assert not (output / "quant_manifest.json").exists()
    assert manifest["unselected_weights"] == {
        "strategy": "convert",
        "format": "bf16",
    }
    assert validate_artifact(output)["valid"]
    rescanned = scan_hf_safetensors(output)
    assert (
        rescanned.tensors[routed_packed].storage_format
        == "compressed-tensors-pack-quantized-int4"
    )


def test_deepseek_v4_per_selector_quantization_end_to_end(tmp_path):
    source = tmp_path / "deepseek-v4-source"
    output = tmp_path / "deepseek-v4-per-selector"
    _make_deepseek_v4_mixed_checkpoint(source)

    main(
        [
            "quantize",
            "--input",
            str(source),
            "--output",
            str(output),
            "--select",
            "moe=int4",
            "activation-bits=16",
            "strategy=group",
            "group-size=32",
            "--select",
            "attention=int8",
            "activation-bits=8",
            "strategy=channel",
            "--select-name",
            r"^mtp\.=int8",
            "activation-bits=16",
            "strategy=group",
            "group-size=64",
            "--n-candidates",
            "8",
            "--backend",
            "cpu",
        ]
    )

    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    assert validation["int4_tensors"] == 6
    assert validation["int8_tensors"] == 7

    config = json.loads((output / "config.json").read_text())
    assert "expert_dtype" not in config
    quant_config = config["quantization_config"]
    assert quant_config["format"] == "mixed-precision"
    assert set(quant_config["config_groups"]) == {
        "w4a16_g32",
        "w8a16_g64",
        "w8a8_channel_token",
    }
    int4_targets = quant_config["config_groups"]["w4a16_g32"]["targets"]
    int8_targets = quant_config["config_groups"]["w8a8_channel_token"]["targets"]
    assert quant_config["config_groups"]["w4a16_g32"]["format"] == "pack-quantized"
    assert quant_config["config_groups"]["w8a16_g64"]["format"] == "pack-quantized"
    assert (
        quant_config["config_groups"]["w8a8_channel_token"]["format"]
        == "int-quantized"
    )
    assert (
        r"re:^.*layers\.0\.ffn\.experts\.\d+\."
        r"(?:gate_proj|up_proj|down_proj|w1|w2|w3)$"
    ) in int4_targets
    assert "layers.0.attn.fused_wqa_wkv" in int8_targets
    assert "layers.0.ffn.shared_experts.gate_up_proj" in int4_targets
    assert "layers.0.ffn.shared_experts.down_proj" in int4_targets
    manifest = json.loads((output / "quantization_manifest.json").read_text())
    assert manifest["algorithm"]["mode"] == "per_selector"
    assert manifest["artifact"]["num_bits"] == [4, 8]
    assert manifest["artifact"]["compression_format"] == "mixed-precision"
    assert manifest["artifact"]["compression_formats"] == [
        "int-quantized",
        "pack-quantized",
    ]
    assert manifest["artifact"]["weight_encoding"] == "mixed"
    assert {
        (scheme["scheme"], scheme["compression_format"])
        for scheme in manifest["artifact"]["schemes"]
    } == {
        ("int4-a16", "pack-quantized"),
        ("int8-a16", "pack-quantized"),
        ("int8-a8", "int-quantized"),
    }

    shard = load_file(output / "model-00001-of-00001.safetensors")
    assert "layers.0.ffn.experts.0.w1.weight_packed" in shard
    assert shard["layers.0.attn.wq_a.weight"].dtype == torch.int8
    assert shard["layers.0.attn.wo_a.weight"].dtype == torch.int8
    assert "layers.0.attn.wo_a.weight_packed" not in shard
    assert "layers.0.attn.wo_a.scale" not in shard
    assert shard["layers.0.attn.wo_b.weight"].dtype == torch.int8
    assert "mtp.0.e_proj.weight_packed" in shard
    assert shard["layers.0.attn.compressor.wkv.weight"].dtype == torch.bfloat16
    assert shard["layers.0.attn.compressor.wgate.weight"].dtype == torch.bfloat16


def test_per_selector_w8a16_group_and_channel_end_to_end(tmp_path):
    source = tmp_path / "deepseek-v4-source"
    output = tmp_path / "deepseek-v4-w8a16-granularity"
    _make_deepseek_v4_mixed_checkpoint(source)

    main(
        [
            "quantize",
            "--input",
            str(source),
            "--output",
            str(output),
            "--select",
            "moe=int8",
            "activation-bits=16",
            "strategy=group",
            "group-size=32",
            "--select",
            "attention=int8",
            "activation-bits=16",
            "strategy=channel",
            "--n-candidates",
            "8",
            "--backend",
            "cpu",
        ]
    )

    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    config = json.loads((output / "config.json").read_text())
    quant_config = config["quantization_config"]
    assert quant_config["format"] == "pack-quantized"
    assert set(quant_config["config_groups"]) == {
        "w8a16_g32",
        "w8a16_channel",
    }
    assert (
        quant_config["config_groups"]["w8a16_g32"]["weights"]["strategy"]
        == "group"
    )
    assert (
        quant_config["config_groups"]["w8a16_channel"]["weights"]["strategy"]
        == "channel"
    )
    manifest = json.loads((output / "quantization_manifest.json").read_text())
    assert manifest["artifact"]["strategy"] == "mixed"
    assert {
        (scheme["scheme"], scheme["strategy"], scheme["group_size"])
        for scheme in manifest["artifact"]["schemes"]
    } == {
        ("int8-a16", "group", 32),
        ("int8-a16", "channel", None),
    }


def test_convert_to_bf16_end_to_end(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "converted"
    _make_checkpoint(source)
    plan = build_convert_plan(scan_hf_safetensors(source))
    execute_plan(source, output, plan, build_backend("cpu"))
    shard = load_file(output / "model-00001-of-00001.safetensors")
    assert all(t.dtype != torch.uint8 for name, t in shard.items() if name.endswith(".weight"))
    with (output / "config.json").open() as f:
        config = json.load(f)
    assert config["expert_dtype"] == "bfloat16"
    assert validate_artifact(output)["valid"]


def test_legacy_int4_manifest_prevents_fp4_misclassification(tmp_path):
    source = tmp_path / "legacy-int4"
    source.mkdir()
    weight_name = "model.layers.0.mlp.down_proj.weight"
    scale_name = f"{weight_name}.scale"
    state = {
        weight_name: torch.zeros((2, 16), dtype=torch.uint8),
        scale_name: torch.ones((2, 1), dtype=torch.bfloat16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(state, str(source / shard))
    with (source / "model.safetensors.index.json").open("w") as f:
        json.dump(
            {
                "metadata": {},
                "weight_map": {name: shard for name in state},
            },
            f,
        )
    with (source / "quant_manifest.json").open("w") as f:
        json.dump(
            {
                "abi_version": "flagos_compressor.artifact.v1",
                "tensors": {
                    weight_name: {
                        "format": "int4_symmetric_groupwise",
                        "logical_shape": [2, 32],
                        "scale": scale_name,
                    }
                },
            },
            f,
        )

    profile = scan_hf_safetensors(source)
    tensor = profile.tensors[weight_name]
    assert tensor.storage_format == "int4_symmetric_groupwise"
    assert profile.metadata["quantization_manifest_file"] == "quant_manifest.json"

    plan = build_convert_plan(profile)
    assert not plan.actions
    assert [item.name for item in plan.unmatched_quantized_tensors] == [weight_name]


def test_cli_commands_and_implicit_convert_dry_run(tmp_path, capsys):
    source = tmp_path / "source"
    _make_checkpoint(source)
    main(["inspect", "--input", str(source)])
    main(
        [
            "quantize",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "quantized"),
            "--select",
            "moe",
            "--dry-run",
        ]
    )
    main(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "converted"),
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert "moe.routed" in output
    assert "compressed_tensors_int4_groupwise" in output
    assert "fp4_e2m1_e8m0" in output
    assert "bf16" in output


def test_int8_weight_quantization_end_to_end(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "quantized-int8"
    _make_checkpoint(source)
    profile = scan_hf_safetensors(source)
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("attention",),
            num_bits=8,
            group_size=128,
            n_candidates=8,
            chunk_size=16,
        ),
    )
    execute_plan(source, output, plan, build_backend("cpu"))

    logical_name = "model.layers.0.self_attn.o_proj.weight"
    packed_name = logical_name.replace(".weight", ".weight_packed")
    scale_name = logical_name.replace(".weight", ".weight_scale")
    shard = load_file(output / "model-00001-of-00001.safetensors")
    assert shard[packed_name].dtype == torch.int32
    assert shard[packed_name].shape == (128, 32)
    assert shard[scale_name].dtype == torch.bfloat16
    assert shard[scale_name].shape == (128, 1)

    with (output / "config.json").open() as f:
        config = json.load(f)
    scheme = next(iter(config["quantization_config"]["config_groups"].values()))
    assert scheme["weights"]["num_bits"] == 8
    assert scheme["weights"]["group_size"] == 128

    with (output / "quantization_manifest.json").open() as f:
        manifest = json.load(f)
    assert manifest["artifact"]["weight_encoding"] == "uint8b128"
    assert manifest["artifact"]["values_per_word"] == 4
    assert (
        manifest["tensors"][packed_name]["format"]
        == "compressed-tensors-pack-quantized-int8"
    )

    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    assert validation["int8_tensors"] == 1
    rescanned = scan_hf_safetensors(output)
    assert (
        rescanned.tensors[packed_name].storage_format
        == "compressed-tensors-pack-quantized-int8"
    )


def test_int8_channel_weight_quantization_end_to_end(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "quantized-int8-channel"
    _make_checkpoint(source)
    profile = scan_hf_safetensors(source)
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("attention",),
            num_bits=8,
            strategy="channel",
            n_candidates=8,
            chunk_size=16,
        ),
    )
    execute_plan(source, output, plan, build_backend("cpu"))

    logical_name = "model.layers.0.self_attn.o_proj.weight"
    packed_name = logical_name.replace(".weight", ".weight_packed")
    scale_name = logical_name.replace(".weight", ".weight_scale")
    shard = load_file(output / "model-00001-of-00001.safetensors")
    assert shard[packed_name].shape == (128, 32)
    assert shard[scale_name].shape == (128, 1)

    config = json.loads((output / "config.json").read_text())
    scheme = config["quantization_config"]["config_groups"][
        "w8a16_channel"
    ]["weights"]
    assert scheme["strategy"] == "channel"
    assert "group_size" not in scheme

    manifest = json.loads(
        (output / "quantization_manifest.json").read_text()
    )
    assert manifest["artifact"]["strategy"] == "channel"
    assert manifest["artifact"]["scale_layout"] == "row_channel"
    spec = manifest["tensors"][packed_name]
    assert spec["strategy"] == "channel"
    assert "group_size" not in spec
    assert spec["scale_shape"] == [128, 1]

    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    assert validation["int8_tensors"] == 1
