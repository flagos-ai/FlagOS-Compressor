"""AWQ's floating-point transform must preserve the model before rounding."""
import copy
import math

import pytest
import torch
from torch import nn

from flagos_compressor.calibration.mappings import infer_awq_mappings
from flagos_compressor.calibration.moe import linearize_fused_experts
from flagos_compressor.quantizers.awq import apply_awq_scale


@pytest.mark.parametrize('kind', [
    'sliding_attention', 'compressed_sparse_attention', 'heavily_compressed_attention',
])
@pytest.mark.parametrize('previous', ['input_layernorm', 'self_attn.q_a_norm'])
@torch.no_grad()
def test_v4_scaling_preserves_attention_and_floating_branches(kind, previous):
    modeling = pytest.importorskip('transformers.models.deepseek_v4.modeling_deepseek_v4')
    from transformers import DeepseekV4Config
    config = DeepseekV4Config(vocab_size=32, hidden_size=32, moe_intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        qk_rope_head_dim=4, q_lora_rank=16, num_experts_per_tok=1, n_routed_experts=2,
        n_shared_experts=1, max_position_embeddings=512, layer_types=[kind],
        mlp_layer_types=['moe'], hc_mult=2, hc_sinkhorn_iters=2, o_groups=2,
        o_lora_rank=8, index_n_heads=2, index_head_dim=8, index_topk=2,
        num_nextn_predict_layers=0, use_cache=False)
    torch.manual_seed(42)
    model = modeling.DeepseekV4ForCausalLM(config).eval().float()
    layer = model.model.layers[0]
    modules = dict(layer.named_modules())
    x = torch.randn(1, 256, 32)
    positions = torch.arange(256).unsqueeze(0)
    pe = {k: model.model.rotary_emb(x, positions, layer_type=k) for k in ['main', 'compress']}
    def forward():
        return layer.self_attn(layer.input_layernorm(x), position_embeddings=pe,
            position_ids=positions, attention_mask=None, past_key_values=None)[0]
    selected = {'self_attn.' + n for n in ['q_a_proj', 'q_b_proj', 'kv_proj', 'o_b_proj']}
    mapping = next(m for m in infer_awq_mappings(layer, selected) if m.previous_name == previous)
    # Capture every projection, including excluded indexer branches. A top-k
    # selection may stay unchanged even if its query scores have been corrupted.
    captures = {}
    handles = [m.register_forward_hook(lambda _m, _a, out, name=name:
        captures.__setitem__(name, out.detach().clone()))
        for name, m in modules.items() if isinstance(m, nn.Linear)]
    try:
        expected = forward()
        reference = copy.deepcopy(captures)
        dim = modules[mapping.linear_names[0]].weight.shape[1]
        scale = torch.exp(torch.linspace(-math.log(2), math.log(2), dim))
        apply_awq_scale(modules[previous], [modules[n] for n in mapping.linear_names], scale)
        torch.testing.assert_close(forward(), expected, atol=1e-7, rtol=1e-5)
        for name in mapping.linear_names:
            torch.testing.assert_close(captures[name], reference[name], atol=1e-6, rtol=1e-5)
        assert set(mapping.quantized_names) <= selected
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('family', ['v4', 'v41'])
def test_clipped_shared_and_linearized_experts_do_not_scale_across_clamp(family):
    modeling = pytest.importorskip(f'transformers.models.deepseek_{family}.modeling_deepseek_{family}')
    if family == 'v4':
        from transformers import DeepseekV4Config
        config = DeepseekV4Config(hidden_size=8, moe_intermediate_size=8,
                                 n_routed_experts=2, swiglu_limit=10.0)
        mlp, experts = modeling.DeepseekV4MLP, modeling.DeepseekV4Experts
    else:
        from transformers.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
        config = DeepseekV41TextConfig(hidden_size=8, moe_intermediate_size=8,
                                      n_routed_experts=2, swiglu_limit=10.0)
        mlp, experts = modeling.DeepseekV41MLP, modeling.DeepseekV41Experts
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.shared_experts = mlp(config)
    layer.mlp.experts = experts(config)
    linearize_fused_experts(layer)
    selected = {n for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
    assert sum(n.endswith('down_proj') for n in selected) == 3
    mappings = infer_awq_mappings(layer, selected)
    assert not any(m.previous_name.endswith('up_proj') for m in mappings)


def test_unclipped_swiglu_keeps_valid_up_down_scaling():
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.up_proj = nn.Linear(8, 16, bias=False)
    layer.mlp.down_proj = nn.Linear(16, 8, bias=False)
    mappings = infer_awq_mappings(layer, {'mlp.up_proj', 'mlp.down_proj'})
    assert len(mappings) == 1
    assert mappings[0].previous_name == 'mlp.up_proj'


@pytest.mark.parametrize('previous', ['input_layernorm', 'self_attn.q_a_norm'])
@torch.no_grad()
def test_v41_shared_attention_sources_preserve_forward_after_scaling(previous):
    modeling = pytest.importorskip('transformers.models.deepseek_v41.modeling_deepseek_v41')
    from transformers.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
    torch.manual_seed(42)
    config = DeepseekV41TextConfig(vocab_size=128, hidden_size=64, num_hidden_layers=5,
        compress_ratios=[0, 2, 2, 1, 1], num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, qk_rope_head_dim=8, q_lora_rank=32, o_groups=2, o_lora_rank=16,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        moe_intermediate_size=64, sliding_window=8, hc_mult=2, hc_sinkhorn_iters=3,
        index_n_heads=2, index_head_dim=32, index_topk=64, candidate_topk_blocks=8,
        candidate_block_size=64, max_position_embeddings=128, num_nextn_predict_layers=0,
        engram_layer_ids=[], use_cache=False)
    model = modeling.DeepseekV41ForCausalLM(config).eval().float()
    inputs = torch.randint(1,128,(1,12))
    expected = model(input_ids=inputs, use_cache=False).logits
    changed = 0
    for layer in model.model.layers:
        selected = {'self_attn.' + n for n in ['q_a_proj', 'q_b_proj', 'kv_proj', 'o_b_proj']}
        mapping = next(m for m in infer_awq_mappings(layer, selected) if m.previous_name == previous)
        modules = dict(layer.named_modules())
        dim = modules[mapping.linear_names[0]].weight.shape[1]
        # Powers of two isolate algebraic correctness from QAT rounding-boundary
        # changes in the model's native FP4 indexer and FP8 KV cache.
        scale = 2.0 ** (torch.arange(dim) % 3 - 1).float()
        apply_awq_scale(modules[previous], [modules[n] for n in mapping.linear_names], scale)
        changed += len(set(mapping.linear_names) - selected)
    assert changed > 0  # Exercise actual retained compressor/indexer consumers.
    torch.testing.assert_close(model(input_ids=inputs, use_cache=False).logits,
                               expected, atol=1e-6, rtol=1e-5)
