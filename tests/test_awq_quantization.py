import torch
from torch import nn

from flagos_compressor.packing.autoawq import pack_autoawq_gemm
from flagos_compressor.quantizers.awq import (
    apply_awq_scale,
    pseudo_quantize_awq,
    search_awq_clip,
    search_awq_scale,
)


def _unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    inverse = [order.index(index) for index in range(8)]
    words = packed.to(torch.int64) & 0xFFFFFFFF
    ordered = torch.stack([(words >> (4 * i)) & 0xF for i in range(8)], dim=2)
    return ordered[..., inverse].reshape(packed.shape[0], packed.shape[1] * 8)


def test_awq_pseudo_quantize_matches_asymmetric_minmax():
    weight = torch.tensor(
        [[-2.0, -1.0, 0.0, 3.0, -4.0, 1.0, 2.0, 4.0]],
        dtype=torch.float32,
    )
    result = pseudo_quantize_awq(weight, group_size=4, zero_point=True)
    assert result.scales.shape == (1, 2)
    assert result.zeros is not None and result.zeros.shape == (1, 2)
    expected_scale = torch.tensor([5 / 15, 8 / 15])
    torch.testing.assert_close(result.scales[0], expected_scale)


def test_autoawq_gemm_pack_round_trip():
    torch.manual_seed(11)
    weight = torch.randn(8, 16)
    quantized = pseudo_quantize_awq(weight, group_size=8, zero_point=True)
    assert quantized.zeros is not None
    packed = pack_autoawq_gemm(
        quantized.weight,
        quantized.scales,
        quantized.zeros,
        group_size=8,
    )
    assert packed.qweight.shape == (16, 1)
    assert packed.qzeros.shape == (2, 1)
    assert packed.scales.shape == (2, 8)

    codes = _unpack_awq(packed.qweight)
    zeros = _unpack_awq(packed.qzeros)
    groups = torch.arange(16) // 8
    reconstructed = (
        packed.scales.float()[groups] * (codes - zeros[groups])
    ).t()
    torch.testing.assert_close(
        reconstructed, quantized.weight, atol=2e-3, rtol=2e-3
    )


def test_awq_scale_and_clip_search_return_channelwise_values():
    torch.manual_seed(5)
    linear = nn.Linear(8, 8, bias=False)
    inputs = torch.randn(2, 4, 8)
    scales = search_awq_scale(
        linear,
        [linear],
        inputs,
        group_size=4,
        n_grid=4,
        max_chunk_memory=1024 * 1024,
    )
    assert scales.shape == (8,)
    assert torch.isfinite(scales).all()

    maxima = search_awq_clip(
        linear.weight,
        inputs,
        group_size=4,
        n_grid=4,
        sample_tokens=8,
        output_chunk_size=4,
    )
    assert maxima.shape == (8, 2, 1)
    assert torch.isfinite(maxima).all()


def test_awq_scale_preserves_zero_centered_qwen_rmsnorm_linear_pair():
    class Qwen3_5RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.1, -0.2, 0.3, -0.4]))

        def forward(self, inputs):
            return inputs * (1 + self.weight)

    norm = Qwen3_5RMSNorm()
    linear = nn.Linear(4, 3, bias=False)
    inputs = torch.randn(2, 4)
    expected = linear(norm(inputs))

    apply_awq_scale(norm, [linear], torch.tensor([0.5, 2.0, 0.25, 4.0]))

    torch.testing.assert_close(linear(norm(inputs)), expected)


def test_awq_search_uses_export_weight_dtype():
    torch.manual_seed(46)
    linear = nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)
    original = linear.weight.detach().clone()
    inputs = torch.randn(32, 128, dtype=torch.bfloat16)
    candidates = []
    handle = linear.register_forward_pre_hook(
        lambda module, _args: candidates.append(module.weight.detach().clone()))
    try:
        scale = search_awq_scale(linear, [linear], inputs, group_size=128,
            zero_point=False)
    finally:
        handle.remove()
    # Reproduce the real scale-then-quantize path, folding the activation's
    # inverse scale back into the weight only for comparison with search.
    with torch.no_grad():
        linear.weight.mul_(scale.view(1, -1))
        quantized = pseudo_quantize_awq(linear.weight, bits=4,
                                        group_size=128, zero_point=False)
        folded = (quantized.weight / scale.view(1, -1)).to(original.dtype)
    assert any(torch.equal(candidate, folded) for candidate in candidates[1:])


def test_awq_search_replays_mutable_state_from_the_same_start():
    class CachedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)
            self.calls = 0

        def forward(self, x, past_key_values, shared):
            assert past_key_values["tokens"] == 0
            assert shared["calls"] == 0
            past_key_values["tokens"] += x.shape[-2]
            shared["calls"] += 1
            self.calls += 1
            return self.proj(x)

    block = CachedBlock().eval()
    kwargs = {"past_key_values": {"tokens": 0}, "shared": {"calls": 0}}
    original = block.proj.weight.detach().clone()
    scales = search_awq_scale(
        block, [block.proj], torch.randn(2, 4, 8), kwargs=kwargs,
        group_size=4, n_grid=4,
    )
    assert torch.isfinite(scales).all()
    assert block.calls == 5  # One reference and four search candidates.
    assert kwargs == {"past_key_values": {"tokens": 0}, "shared": {"calls": 0}}
    torch.testing.assert_close(block.proj.weight, original, rtol=0, atol=0)
