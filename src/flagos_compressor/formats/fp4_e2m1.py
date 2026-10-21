from __future__ import annotations

import torch

from flagos_compressor.backends.base import BackendRunContext, QuantBackend
from flagos_compressor.backends.op_registry import register_op
from flagos_compressor.formats.base import WeightFormat, register_weight_format
from flagos_compressor.formats.fp8_e8m0 import decode_e8m0_scale


FP4_GROUP_SIZE = 32

_FP4_E2M1_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def fp4_e2m1_lut(device: torch.device | str | None = None) -> torch.Tensor:
    return torch.tensor(_FP4_E2M1_VALUES, dtype=torch.bfloat16, device=device)


def unpack_fp4_e2m1(weight_packed: torch.Tensor) -> torch.Tensor:
    """Unpack MXFP4 E2M1 bytes into BF16 values without applying scale."""
    if weight_packed.dim() != 2:
        raise ValueError(f"Expected a 2D packed FP4 tensor, got {weight_packed.dim()}D")
    out_features, packed_in = weight_packed.shape
    raw = weight_packed.to(torch.uint8)
    low = (raw & 0x0F).to(torch.long)
    high = (raw >> 4).to(torch.long)
    lut = fp4_e2m1_lut(weight_packed.device)
    low_values = lut[low]
    high_values = lut[high]
    return torch.stack([low_values, high_values], dim=-1).reshape(out_features, packed_in * 2)


def dequant_fp4_e2m1(weight_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 E2M1 + E8M0 scale into BF16."""
    values = unpack_fp4_e2m1(weight_packed)
    out_features, in_features = values.shape
    scale = decode_e8m0_scale(scale_e8m0)
    if scale.dim() == 2 and scale.shape[0] == out_features:
        groups_per_row = scale.shape[1]
        scale = scale.reshape(out_features, groups_per_row)
    else:
        groups_per_row = scale.numel() // out_features
        scale = scale.reshape(out_features, groups_per_row)
    if groups_per_row <= 0 or in_features % groups_per_row != 0:
        raise ValueError(
            f"Invalid FP4 scale shape {tuple(scale_e8m0.shape)} for packed weight "
            f"{tuple(weight_packed.shape)}"
        )
    group_size = in_features // groups_per_row
    expanded_scale = scale.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_features, in_features)
    return (values * expanded_scale).to(torch.bfloat16)


register_op("fp4_unpack")(unpack_fp4_e2m1)
register_op("fp4_dequant")(dequant_fp4_e2m1)


class Fp4E2M1E8M0Format(WeightFormat):
    name = "fp4_e2m1_e8m0"

    def to_canonical(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict,
    ) -> torch.Tensor:
        del params
        if scale is None:
            raise ValueError("FP4 E2M1 input requires an E8M0 scale tensor")
        return backend.run("fp4_dequant", weight, scale, context=context)


register_weight_format(Fp4E2M1E8M0Format())
