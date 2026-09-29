"""MXFP8 and compressed-tensors NVFP4 source checkpoint decoding."""
from __future__ import annotations

import torch

from flagos_compressor.backends.op_registry import register_op
from flagos_compressor.formats.base import WeightFormat, register_weight_format
from flagos_compressor.formats.fp4_e2m1 import unpack_fp4_e2m1
from flagos_compressor.formats.fp8_e8m0 import decode_e8m0_scale


def mxfp8_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("MXFP8 requires a 2D E4M3 weight")
    rows, cols = weight.shape
    if cols % 32 or tuple(scale.shape) != (rows, cols // 32):
        raise ValueError("MXFP8 requires one E8M0 scale per row group of 32")
    if scale.dtype == torch.uint8:
        if (scale == 255).any().item():
            raise ValueError("MXFP8 E8M0 scale contains reserved NaN code 255")
    elif not (hasattr(torch, "float8_e8m0fnu") and scale.dtype == torch.float8_e8m0fnu):
        raise ValueError("MXFP8 scale must be UINT8 E8M0 bits or float8_e8m0fnu")
    factors = decode_e8m0_scale(scale)
    return (weight.float().reshape(rows, -1, 32) * factors.unsqueeze(-1)).reshape(
        rows, cols
    ).to(torch.bfloat16)


def nvfp4_dequant(
    weight: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    if weight.ndim != 2 or weight.dtype != torch.uint8:
        raise ValueError("NVFP4 requires a 2D UINT8 packed weight")
    rows, packed_cols = weight.shape
    if packed_cols % 8 or tuple(scale.shape) != (rows, packed_cols // 8):
        raise ValueError("NVFP4 requires one E4M3 scale per group of 16 values")
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError("NVFP4 block scale must be E4M3")
    if global_scale is None or global_scale.numel() != 1:
        raise ValueError("NVFP4 requires one weight_global_scale")
    if not torch.isfinite(global_scale).all().item() or global_scale.item() <= 0:
        raise ValueError("NVFP4 weight_global_scale must be finite and positive")
    # compressed-tensors stores the quantization multiplier; undo it by
    # division. input_global_scale describes activations and is not used here.
    values = unpack_fp4_e2m1(weight).float().reshape(rows, -1, 16)
    factors = scale.float() / global_scale.float()
    return (values * factors.unsqueeze(-1)).reshape(rows, packed_cols * 2).to(
        torch.bfloat16
    )


for _backend in ("cpu", "torch"):
    register_op("mxfp8_dequant", _backend)(mxfp8_dequant)
    register_op("nvfp4_dequant", _backend)(nvfp4_dequant)


class Mxfp8Format(WeightFormat):
    name = "mxfp8_e4m3_e8m0"

    def to_canonical(self, weight, scale, backend, context, params):
        if scale is None:
            raise ValueError("MXFP8 requires scale")
        return backend.run("mxfp8_dequant", weight, scale, context=context)


class Nvfp4Format(WeightFormat):
    name = "nvfp4"

    def to_canonical(self, weight, scale, backend, context, params):
        if scale is None or params.get("global_scale") is None:
            raise ValueError("NVFP4 requires block and global weight scales")
        return backend.run(
            "nvfp4_dequant", weight, scale, params["global_scale"], context=context
        )


register_weight_format(Mxfp8Format())
register_weight_format(Nvfp4Format())
