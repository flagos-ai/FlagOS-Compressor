from __future__ import annotations

import torch

from flagos_compressor.backends.base import BackendRunContext, QuantBackend
from flagos_compressor.backends.op_registry import register_op
from flagos_compressor.formats.base import WeightFormat, register_weight_format


def decode_e8m0_scale(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponent-only scale into float32."""
    if scale.dtype == torch.float32:
        return scale
    if scale.dtype in (torch.bfloat16, torch.float16):
        return scale.float()
    if hasattr(torch, "float8_e8m0fnu") and scale.dtype == torch.float8_e8m0fnu:
        return scale.float()
    if scale.element_size() == 1:
        exponent = scale.to(torch.uint8).to(torch.float32)
        return torch.pow(torch.tensor(2.0, device=scale.device), exponent - 127.0)
    return scale.float()


def block_fp8_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int | tuple[int, int] | list[int] = 128,
    qkv_group_sizes: list[int] | tuple[int, ...] | None = None,
    qkv_groups: int | None = None,
) -> torch.Tensor:
    """Dequantize 2D block-FP8 weight into BF16."""
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got {weight.dim()}D")
    bm, bn = (block_size, block_size) if isinstance(block_size, int) else block_size
    if bm <= 0 or bn <= 0:
        raise ValueError("FP8 block sizes must be positive")
    if qkv_group_sizes is not None:
        if (
            not qkv_groups
            or len(qkv_group_sizes) != 3
            or any(v <= 0 for v in qkv_group_sizes)
        ):
            raise ValueError("Invalid grouped QKV dimensions")
        rows = sum(qkv_group_sizes)
        scale_rows = (rows + bm - 1) // bm
        if weight.shape[0] != rows * qkv_groups or scale.shape != (
            scale_rows * qkv_groups, (weight.shape[1] + bn - 1) // bn
        ):
            raise ValueError("Grouped QKV weight/scale shape mismatch")
        parts = [[], [], []]
        for group in range(qkv_groups):
            decoded = block_fp8_dequant(
                weight[group * rows:(group + 1) * rows],
                scale[group * scale_rows:(group + 1) * scale_rows],
                (bm, bn),
            )
            for target, piece in zip(parts, decoded.split(qkv_group_sizes, dim=0)):
                target.append(piece)
        return torch.cat([torch.cat(part, dim=0) for part in parts], dim=0)
    shape = weight.shape
    m, n = shape
    scale = decode_e8m0_scale(scale)

    expected = ((m + bm - 1) // bm, (n + bn - 1) // bn)
    if scale.numel() != expected[0] * expected[1] or (
        scale.dim() != 1 and tuple(scale.shape) != expected
    ):
        raise ValueError(
            f"FP8 scale shape {tuple(scale.shape)} does not match {expected}"
        )
    pad_m = (bm - m % bm) % bm
    pad_n = (bn - n % bn) % bn
    if pad_m or pad_n:
        # CPU/CUDA pad kernels need not support Float8.
        weight = torch.nn.functional.pad(weight.float(), (0, pad_n, 0, pad_m))
    mp, np = weight.shape

    blocks = (
        weight.reshape(mp // bm, bm, np // bn, bn)
        .transpose(1, 2)
        .contiguous()
        .view(-1, bm * bn)
    )
    dequant = (blocks.float() * scale.reshape(-1, 1)).to(torch.bfloat16)
    dequant = (
        dequant.view(mp // bm, np // bn, bm, bn)
        .transpose(1, 2)
        .contiguous()
        .view(mp, np)
    )
    if pad_m or pad_n:
        dequant = dequant[:m, :n]
    return dequant


register_op("e8m0_decode", "cpu")(decode_e8m0_scale)
register_op("e8m0_decode", "torch")(decode_e8m0_scale)
register_op("fp8_dequant", "cpu")(block_fp8_dequant)
register_op("fp8_dequant", "torch")(block_fp8_dequant)


class Fp8BlockE8M0Format(WeightFormat):
    name = "fp8_block_e8m0"

    def to_canonical(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict,
    ) -> torch.Tensor:
        if scale is None:
            raise ValueError("Block FP8 input requires an E8M0 scale tensor")
        return backend.run(
            "fp8_dequant",
            weight,
            scale,
            block_size=params.get("block_size", 128),
            qkv_group_sizes=params.get("qkv_group_sizes"),
            qkv_groups=params.get("qkv_groups"),
            context=context,
        )


register_weight_format(Fp8BlockE8M0Format())
