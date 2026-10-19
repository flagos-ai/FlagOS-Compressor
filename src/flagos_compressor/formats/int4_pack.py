from __future__ import annotations

import torch

from flagos_compressor.backends.op_registry import register_op


def pack_uint4b8_int32(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values into compressed-tensors ``uint4b8`` words.

    ``uint4b8`` stores a signed value ``q`` as the unsigned nibble ``q + 8``.
    Eight consecutive values along the input dimension are packed low-to-high
    into one little-endian int32 word.
    """
    if values.dim() != 2:
        raise ValueError(f"Expected a 2D INT4 tensor, got {values.dim()}D")
    if values.shape[1] % 8:
        raise ValueError(
            "compressed-tensors INT4 packing requires in_features divisible "
            f"by 8, got {tuple(values.shape)}"
        )
    values = values.to(torch.int8)
    if values.numel() and (values.min().item() < -8 or values.max().item() > 7):
        raise ValueError("Signed INT4 values must be in [-8, 7]")
    codes = (values.to(torch.int16) + 8).to(torch.uint8)
    low = codes[:, 0::2]
    high = codes[:, 1::2] << 4
    packed_bytes = (low | high).contiguous()
    return packed_bytes.view(torch.int32)


def unpack_uint4b8_int32(packed: torch.Tensor) -> torch.Tensor:
    """Unpack compressed-tensors int32 words into signed INT4 values."""
    if packed.dim() != 2 or packed.dtype != torch.int32:
        raise ValueError("Expected a 2D int32 packed tensor")
    raw = packed.contiguous().view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    codes = torch.stack([low, high], dim=-1).reshape(
        packed.shape[0], packed.shape[1] * 8
    )
    return (codes.to(torch.int16) - 8).to(torch.int8)


register_op("int4_pack_uint4b8_int32")(pack_uint4b8_int32)
register_op("int4_unpack_uint4b8_int32")(unpack_uint4b8_int32)
