import json

import pytest
import torch
from safetensors.torch import load_file, save_file

import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.planner import build_convert_plan, build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy
from flagos_compressor.core.profile import ModelProfile
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.microscaling import mxfp8_dequant, nvfp4_dequant
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


def test_mxfp8_decodes_per_row_exponents():
    weight = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[126, 128], [127, 125]], dtype=torch.uint8)
    expected = torch.tensor([[0.5, 2.0], [1.0, 0.25]]).repeat_interleave(32, dim=1)
    torch.testing.assert_close(mxfp8_dequant(weight, scale).float(), expected)
    with pytest.raises(ValueError, match="255"):
        mxfp8_dequant(weight, torch.full_like(scale, 255))


def test_nvfp4_all_nibbles_and_inverse_global_scale():
    # Low nibble is first. Cover positive, negative, zero and all E2M1 values.
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8)
    scale = torch.tensor([[2.0, 8.0]], dtype=torch.float8_e4m3fn)
    values = torch.tensor([[0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]])
    expected = torch.cat([values * 0.5, values * 2], dim=1)
    torch.testing.assert_close(nvfp4_dequant(packed, scale, torch.tensor(4.)).float(), expected)
    with pytest.raises(ValueError, match="positive"):
        nvfp4_dequant(packed, scale, torch.tensor(0.))


def make_source(path):
    path.mkdir()
    state = {}
    for proj in ("w1", "w2", "w3"):
        base = f"model.layers.0.mlp.experts.0.{proj}"
        state[base + ".weight_packed"] = torch.full((2, 16), 0x21, dtype=torch.uint8)
        state[base + ".weight_scale"] = torch.full((2, 2), 4., dtype=torch.float8_e4m3fn)
        state[base + ".weight_global_scale"] = torch.tensor(2.)
        # Must be removed from the output, not applied to the weight.
        state[base + ".input_global_scale"] = torch.tensor(100.)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        base = "model.layers.0.self_attn." + proj
        state[base + ".weight"] = torch.ones(2, 32, dtype=torch.float8_e4m3fn)
        state[base + ".weight_scale_inv"] = torch.full((2, 1), 126, dtype=torch.uint8)
    state["model.embed_tokens.weight"] = torch.ones(4, 32, dtype=torch.bfloat16)
    # Put all auxiliary scales in a separate shard to exercise their removal
    # and cross-shard lookup, including a generated .weight_scale collision.
    weights = {n: t for n, t in state.items() if n.endswith((".weight", ".weight_packed"))}
    scales = {n: t for n, t in state.items() if n not in weights}
    mapping = {}
    for i, part in enumerate((weights, scales)):
        shard = f"model-0000{i + 1}-of-00002.safetensors"
        save_file(part, str(path / shard))
        mapping.update({n: shard for n in part})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "mxfp8"}}))


@pytest.mark.parametrize("quantize", [False, True])
def test_mixed_microscaling_checkpoint_end_to_end(tmp_path, quantize):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    profile = scan_hf_safetensors(source)
    assert profile.summary()["storage_formats"] == {"mxfp8_e4m3_e8m0": 4, "nvfp4": 3}
    profile = ModelProfile.from_dict(profile.to_dict())
    if quantize:
        policy = QuantizationPolicy(selections=("linear",), num_bits=8,
                                    activation_num_bits=8, strategy="channel", n_candidates=8)
        assert sum(policy.selects(t) for t in profile.tensors.values()) == 7
        plan = build_quantize_plan(profile, policy)
    else:
        plan = build_convert_plan(profile)
    assert len(plan.actions) == 7
    assert not plan.unmatched_quantized_tensors
    execute_plan(source, output, plan, build_backend("cpu", fallback_policy="error"))
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert not any(n.endswith(("weight_packed", "global_scale", "scale_inv")) for n in index)
    state = {}
    for shard in set(index.values()):
        state.update(load_file(output / shard))
    base = "model.layers.0.mlp.experts.0.w1"
    expected = torch.tensor([1., 2.]).repeat(16).expand(2, -1)
    if quantize:
        assert state[base + ".weight"].dtype == torch.int8
        reconstructed = state[base + ".weight"].float() * state[base + ".weight_scale"]
        torch.testing.assert_close(reconstructed, expected, rtol=.02, atol=.02)
        result = validate_artifact(output)
        assert result["valid"], result["errors"]
        assert result["int8_tensors"] == 7
        # The exported logical name and shape must round trip.
        rescanned = scan_hf_safetensors(output)
        assert rescanned.tensors[base + ".weight"].effective_logical_shape == (2, 32)
    else:
        torch.testing.assert_close(state[base + ".weight"].float(), expected)
        assert "quantization_config" not in json.loads((output / "config.json").read_text())


def test_nvfp4_unselected_experts_are_decoded_and_renamed(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    plan = build_quantize_plan(scan_hf_safetensors(source), QuantizationPolicy(
        selections=("attention",), num_bits=8, activation_num_bits=8,
        strategy="channel", n_candidates=8,
    ))
    execute_plan(source, output, plan, build_backend("cpu"))
    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    assert validation["int8_tensors"] == 4
