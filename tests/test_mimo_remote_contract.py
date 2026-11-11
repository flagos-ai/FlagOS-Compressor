"""Opt-in contract test against downloaded MiMo remote code (no model weights).

Set FLAGOS_MIMO_REMOTE_SOURCE to a trusted local config + Python source directory.
The normal unit suite remains offline and covers the same API with small fixtures.
"""

import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from flagos_compressor.calibration.inputs import prepare_calibration_inputs, _mimo_mask_compat
from flagos_compressor.calibration.modeling import decoder_layers, _restore_source_fp32


@pytest.mark.parametrize("pattern", [[0, 1, 0], [1, 0, 1]])
def test_downloaded_mimo_forward_replay_and_fp32_exceptions(tmp_path, pattern):
    source = os.environ.get("FLAGOS_MIMO_REMOTE_SOURCE")
    if not source:
        pytest.skip("set FLAGOS_MIMO_REMOTE_SOURCE to downloaded, trusted MiMo remote code")
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    for key, value in dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, v_head_dim=8,
        swa_num_attention_heads=2, swa_num_key_value_heads=1, swa_head_dim=16, swa_v_head_dim=8,
        hybrid_layer_pattern=pattern, n_routed_experts=2, num_experts_per_tok=2,
        moe_intermediate_size=64, n_group=1, topk_group=1, moe_layer_freq=[False, True, True],
        vision_config={}, audio_config={}, sliding_window=4, use_cache=False,
        partial_rotary_factor=1.0, attention_projection_layout="fused_qkv",
        rope_scaling=None, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        rope_theta=10000.0, swa_rope_theta=100.0,
        add_full_attention_sink_bias=True, add_swa_attention_sink_bias=True,
    ).items():
        setattr(config, key, value)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, dtype=torch.bfloat16).eval()
    originals = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("attention_sink_bias", "e_score_correction_bias")):
                parameter.data = torch.full(parameter.shape, 1.0003, dtype=torch.float32)
                originals[name] = parameter.detach().clone()
            elif "layernorm.weight" in name or name == "model.norm.weight":
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.02)
    layers = decoder_layers(model)
    batch = {"input_ids": torch.randint(0, 64, (2, 12)),
             "attention_mask": torch.ones(2, 12, dtype=torch.long)}
    samples, layer_kwargs = prepare_calibration_inputs(
        model, layers, [batch], device=torch.device("cpu"))
    with torch.no_grad(), _mimo_mask_compat(model.model):
        expected = model.model(**batch, use_cache=False).last_hidden_state
        hidden = samples[0][0][0]
        for name, layer in layers:
            hidden = layer(hidden, **layer_kwargs[name][0])
        torch.testing.assert_close(model.model.norm(hidden), expected, rtol=0, atol=0)
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path, trust_remote_code=True, dtype=torch.bfloat16, local_files_only=True)
    _restore_source_fp32(loaded, tmp_path)
    assert originals
    for name, original in originals.items():
        assert loaded.state_dict()[name].dtype == torch.float32
        torch.testing.assert_close(loaded.state_dict()[name], original, rtol=0, atol=0)
