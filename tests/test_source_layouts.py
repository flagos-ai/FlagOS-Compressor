import json

import pytest
import torch
from safetensors.torch import load_file, save_file

import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.executor import _deepseek_v4_runtime_targets, execute_plan
from flagos_compressor.core.planner import build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy, UnselectedWeightsPolicy
from flagos_compressor.core.profile import ModelProfile
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.fp8_e8m0 import block_fp8_dequant
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors
from flagos_compressor.inspect.source_layouts import fp8_source_params
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


def test_v41_runtime_targets_cover_multimodal_wrapper_and_fused_projections():
    selected = {
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wo_b.weight",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.0.ffn.shared_experts.w3.weight",
    }
    targets = _deepseek_v4_runtime_targets({"model_type": "deepseek_v41"}, selected)
    for prefix in ("model.", "language_model.model."):
        assert prefix + "layers.0.attn.fused_wqa_wkv" in targets
        assert prefix + "layers.0.attn.wo_b" in targets
        assert prefix + "layers.0.ffn.shared_experts.gate_up_proj" in targets
    assert not any("indexer" in target for target in targets)
    partial = _deepseek_v4_runtime_targets(
        {"model_type": "deepseek_v41"}, {"layers.0.attn.wq_a.weight"}
    )
    assert not any("fused_wqa_wkv" in target for target in partial)


@pytest.mark.parametrize(
    "block,shape", [((32, 32), (70, 95)), ((1, 32), (7, 64)), ((128, 128), (150, 190))]
)
def test_fp8_rectangular_and_padded_blocks(block, shape):
    torch.manual_seed(3)
    w = torch.randn(shape).to(torch.float8_e4m3fn)
    bm, bn = block
    scale = torch.rand(((shape[0] + bm - 1) // bm, (shape[1] + bn - 1) // bn)) + 0.5
    reference = (
        w.float()
        * scale.repeat_interleave(bm, 0).repeat_interleave(bn, 1)[
            : shape[0], : shape[1]
        ]
    ).bfloat16()
    assert torch.equal(block_fp8_dequant(w, scale, block), reference)


def test_fp8_rejects_transposed_scale_grid():
    with pytest.raises(ValueError, match="scale shape"):
        block_fp8_dequant(torch.ones(64, 96), torch.ones(3, 2), 32)


def _mimo_config():
    return {
        "model_type": "mimo_v2",
        "attention_projection_layout": "fused_qkv",
        "hybrid_layer_pattern": [0, 1],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 48,
        "v_head_dim": 32,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 48,
        "swa_v_head_dim": 32,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32]},
    }


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.1.self_attn.qkv_proj.weight",
        "model.mtp.layers.0.self_attn.qkv_proj.weight",
    ],
)
def test_mimo_decodes_independent_group_padding_and_qkv_row_order(name):
    w = (
        torch.arange(352)
        .remainder(12)
        .float()[:, None]
        .expand(-1, 64)
        .to(torch.float8_e4m3fn)
    )
    scale = torch.arange(24).reshape(12, 2).float() + 1
    params = fp8_source_params(name, w.shape, scale.shape, _mimo_config())
    decoded = block_fp8_dequant(w, scale, **params)
    # Independent reference: index each original group's scale grid, then
    # order all Q heads, all K heads, and all V heads.
    group_rows, q_rows, k_rows = 176, 96, 48
    reference = torch.empty_like(w, dtype=torch.bfloat16)
    for row in range(352):
        group, local = divmod(row, group_rows)
        if local < q_rows:
            target = group * q_rows + local
        elif local < q_rows + k_rows:
            target = 2 * q_rows + group * k_rows + local - q_rows
        else:
            target = 2 * (q_rows + k_rows) + group * 32 + local - q_rows - k_rows
        reference[target] = (
            w[row].float() * scale[group * 6 + local // 32].repeat_interleave(32)
        ).bfloat16()
    assert torch.equal(decoded, reference)
    with pytest.raises(ValueError, match="Invalid MiMo"):
        fp8_source_params(name, w.shape, (11, 2), _mimo_config())


def test_mimo_swa_uses_checkpoint_tp_not_its_own_kv_head_count():
    config = _mimo_config()
    config.update(
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=8,
        swa_head_dim=192,
        swa_v_head_dim=128,
    )
    config["quantization_config"]["weight_block_size"] = [128, 128]
    params = fp8_source_params(
        "model.layers.1.self_attn.qkv_proj.weight", (14848, 4096), (116, 32), config
    )
    assert params["qkv_groups"] == 4
    assert params["qkv_group_sizes"] == [3072, 384, 256]


@pytest.mark.parametrize("preserve", [False, True])
def test_indexer_and_engram_export_policy(tmp_path, monkeypatch, preserve):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    linear = "layers.0.attn.wo_b.weight"
    indexer = "layers.0.attn.indexer.wq_b.weight"
    embedding = "layers.1.engram.embed.weight"
    state = {
        linear: torch.ones(64, 64).to(torch.float8_e4m3fn),
        linear.replace(".weight", ".scale"): torch.ones(2, 2),
        indexer: torch.full((64, 64), 2.0).to(torch.float8_e4m3fn),
        indexer.replace(".weight", ".scale"): torch.full((2, 2), 3.0),
        embedding: torch.ones(3, 64).to(torch.float8_e4m3fn),
        embedding.replace(".weight", ".scale"): torch.full((3, 2), 4.0),
        "norm.weight": torch.ones(64, dtype=torch.bfloat16),
        "router.bias": torch.tensor([1.234567], dtype=torch.float32),
    }
    save_file(state, source / "model.safetensors")
    source_config = {
        "quant_method": "fp8",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
    }
    (source / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v41", "quantization_config": source_config})
    )
    # Scanning a 100 GB Engram shard must only read metadata.
    original = HfSafetensorsCheckpoint.load_shard
    monkeypatch.setattr(
        HfSafetensorsCheckpoint,
        "load_shard",
        lambda *args: pytest.fail("scanner loaded weight data"),
    )
    profile = scan_hf_safetensors(source)
    assert profile.tensors[embedding].storage_params["block_size"] == [1, 32]
    assert profile.tensors[indexer].tags == ("attention.indexer",)
    assert (
        ModelProfile.from_dict(profile.to_dict()).tensors[linear]
        == profile.tensors[linear]
    )
    monkeypatch.setattr(HfSafetensorsCheckpoint, "load_shard", original)
    policy = QuantizationPolicy(
        selections=("linear",),
        num_bits=8,
        activation_num_bits=8,
        strategy="channel",
        n_candidates=8,
        unselected=(
            UnselectedWeightsPolicy("preserve", None)
            if preserve
            else UnselectedWeightsPolicy()
        ),
    )
    plan = build_quantize_plan(profile, policy)
    assert {a.tensor.name for a in plan.actions} == (
        {linear} if preserve else {linear, indexer, embedding}
    )
    execute_plan(source, output, plan, build_backend("cpu"))
    result = load_file(output / "model.safetensors")
    assert result[linear].dtype == torch.int8
    if not preserve:
        assert result[indexer].dtype == result[embedding].dtype == torch.bfloat16
        torch.testing.assert_close(
            result[indexer], torch.full((64, 64), 6.0, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            result[embedding], torch.full((3, 64), 4.0, dtype=torch.bfloat16)
        )
        assert indexer.replace(".weight", ".scale") not in result
        assert embedding.replace(".weight", ".scale") not in result
        assert torch.equal(result["router.bias"], state["router.bias"])
        assert result["router.bias"].dtype == torch.float32
        assert not any(
            str(value.dtype).startswith("torch.float8") for value in result.values()
        )
        config = json.loads((output / "config.json").read_text())
        assert "flagos_source_quantization" not in config
        assert config["flagos_bf16_engram"] is True
        assert validate_artifact(output)["valid"]
        return
    for name, tensor in state.items():
        if name.startswith("layers.0.attn.wo_b."):
            continue
        assert result[name].dtype == tensor.dtype
        assert torch.equal(result[name].view(torch.uint8), tensor.view(torch.uint8))
    config = json.loads((output / "config.json").read_text())
    assert config["flagos_source_quantization"]["quantization_config"] == source_config
    manifest = json.loads((output / "quantization_manifest.json").read_text())
    assert set(manifest["preserved_tensors"]) == {embedding, indexer}
    assert manifest["runtime_config"]["requires_source_format_modules"] is True
    assert validate_artifact(output)["valid"]
    rescanned = scan_hf_safetensors(output)
    assert rescanned.tensors[embedding].storage_params == {"block_size": [1, 32]}
    assert rescanned.tensors[indexer].storage_params == {"block_size": [32, 32]}
    manifest["preserved_tensors"][indexer]["scale"] = "missing.scale"
    (output / "quantization_manifest.json").write_text(json.dumps(manifest))
    validation = validate_artifact(output)
    assert not validation["valid"]
    assert any(
        "Preserved source scale is missing" in error for error in validation["errors"]
    )
