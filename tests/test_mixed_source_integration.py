"""Cross-PR contracts for rectangular FP8, MXFP8 and NVFP4 sources."""

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
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


@pytest.mark.parametrize("quantize", [False, True])
def test_mixed_source_profile_and_export(tmp_path, quantize):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    block = "model.layers.0.self_attn.q_proj"
    mx = "model.layers.0.self_attn.k_proj"
    nv = "model.layers.0.self_attn.v_proj"
    save_file({
        block + ".weight": torch.full((64, 64), 2., dtype=torch.float8_e4m3fn),
        block + ".weight_scale_inv": torch.full((2, 2), 3.),
        mx + ".weight": torch.ones(2, 32, dtype=torch.float8_e4m3fn),
        mx + ".weight_scale": torch.full((2, 1), 126, dtype=torch.uint8),
        nv + ".weight_packed": torch.full((2, 16), 0x21, dtype=torch.uint8),
        nv + ".weight_scale": torch.full((2, 2), 4., dtype=torch.float8_e4m3fn),
        nv + ".weight_global_scale": torch.tensor(2.),
        nv + ".input_global_scale": torch.tensor(100.),
    }, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32]},
    }))
    profile = scan_hf_safetensors(source)
    restored = ModelProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
    assert restored == profile
    block_info = restored.tensors[block + ".weight"]
    assert block_info.storage_params == {"block_size": [32, 32], "scale_dtype": "float32"}
    assert restored.tensors[mx + ".weight"].storage_format == "mxfp8_e4m3_e8m0"
    nv_info = restored.tensors[nv + ".weight_packed"]
    assert nv_info.effective_name == nv + ".weight"
    assert nv_info.global_scale_name == nv + ".weight_global_scale"
    assert set(nv_info.auxiliary_names) == {nv + ".weight_global_scale", nv + ".input_global_scale"}
    plan = (build_quantize_plan(restored, QuantizationPolicy(
        selections=("linear",), num_bits=8, activation_num_bits=8,
        strategy="channel", n_candidates=8,
    )) if quantize else build_convert_plan(restored))
    execute_plan(source, output, plan, build_backend("cpu"))
    state = load_file(output / "model.safetensors")
    for name, expected in (
        (block, torch.full((64, 64), 6.)),
        (mx, torch.full((2, 32), .5)),
        (nv, torch.tensor([1., 2.]).repeat(16).expand(2, -1)),
    ):
        actual = state[name + ".weight"].float()
        if quantize:
            actual *= state[name + ".weight_scale"]
        else:
            assert state[name + ".weight"].dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
    assert not any(name.endswith(("weight_packed", "scale_inv", "global_scale")) for name in state)
    if quantize:
        result = validate_artifact(output)
        assert result["valid"], result["errors"]
        assert result["int8_tensors"] == 3
    else:
        assert len(state) == 3
