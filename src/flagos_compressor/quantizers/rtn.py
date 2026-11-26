"""Explicit weight-only rounding for experts without calibration observations."""

import torch

from flagos_compressor.quantizers.gptq import GPTQResult, _fake_quantize, _find_params


@torch.no_grad()
def quantize_rtn(
    weight: torch.Tensor, *, bits: int, group_size: int, symmetric: bool
) -> GPTQResult:
    if weight.ndim != 2 or bits not in {4, 8}:
        raise ValueError("RTN requires a 2D weight and 4 or 8 bits")
    if group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError("RTN group_size must divide in_features")
    if not torch.isfinite(weight).all():
        raise ValueError("RTN cannot quantize non-finite weights")
    grouped = weight.detach().float().reshape(-1, group_size)
    scales, zeros = _find_params(grouped, bits=bits, symmetric=symmetric)
    rounded = _fake_quantize(grouped, scales, zeros, bits=bits)
    return GPTQResult(
        weight=rounded.reshape_as(weight).to(weight.dtype),
        scales=scales.reshape(weight.shape[0], -1),
        zeros=zeros.reshape(weight.shape[0], -1),
        g_idx=torch.arange(weight.shape[1], device=weight.device, dtype=torch.int32)
        // group_size,
    )
