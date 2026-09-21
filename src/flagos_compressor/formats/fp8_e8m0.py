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


def block_fp8_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantize 2D block-FP8 weight into BF16."""
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got {weight.dim()}D")
    shape = weight.shape
    m, n = shape
    scale = decode_e8m0_scale(scale)

    pad_m = (block_size - m % block_size) % block_size
    pad_n = (block_size - n % block_size) % block_size
    if pad_m or pad_n:
        weight = torch.nn.functional.pad(weight, (0, pad_n, 0, pad_m))
    mp, np = weight.shape

    blocks = (
        weight.view(mp // block_size, block_size, np // block_size, block_size)
        .transpose(1, 2)
        .contiguous()
        .view(-1, block_size * block_size)
    )
    dequant = (blocks.float() * scale.reshape(-1, 1)).to(torch.bfloat16)
    dequant = (
        dequant.view(mp // block_size, np // block_size, block_size, block_size)
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
            block_size=int(params.get("block_size", 128)),
            context=context,
        )


register_weight_format(Fp8BlockE8M0Format())
