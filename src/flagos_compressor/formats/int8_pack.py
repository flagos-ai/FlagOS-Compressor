from __future__ import annotations

import torch

from flagos_compressor.backends.op_registry import register_op


def pack_uint8b128_int32(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT8 values into compressed-tensors ``uint8b128`` words."""
    if values.dim() != 2:
        raise ValueError(f"Expected a 2D INT8 tensor, got {values.dim()}D")
    if values.numel():
        min_value, max_value = torch.aminmax(values)
        if min_value.item() < -128 or max_value.item() > 127:
            raise ValueError("Signed INT8 values must be in [-128, 127]")
    values = values.to(torch.int8)
    in_features = int(values.shape[1])
    padding = (-in_features) % 4
    codes = (values.to(torch.int16) + 128).to(torch.uint8)
    if padding:
        codes = torch.nn.functional.pad(codes, (0, padding), value=128)
    return codes.contiguous().view(torch.int32)


def unpack_uint8b128_int32(
    packed: torch.Tensor,
    in_features: int | None = None,
) -> torch.Tensor:
    """Unpack compressed-tensors int32 words into signed INT8 values."""
    if packed.dim() != 2 or packed.dtype != torch.int32:
        raise ValueError("Expected a 2D int32 packed tensor")
    codes = packed.contiguous().view(torch.uint8)
    if in_features is not None:
        if in_features < 0 or in_features > codes.shape[1]:
            raise ValueError(
                f"in_features must be in [0, {codes.shape[1]}], got {in_features}"
            )
        codes = codes[:, :in_features]
    return (codes.to(torch.int16) - 128).to(torch.int8)


register_op("int8_pack_uint8b128_int32")(pack_uint8b128_int32)
register_op("int8_unpack_uint8b128_int32")(unpack_uint8b128_int32)
