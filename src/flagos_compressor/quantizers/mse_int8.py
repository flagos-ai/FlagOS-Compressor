from __future__ import annotations

import torch

from flagos_compressor.backends.op_registry import register_op
from flagos_compressor.quantizers.mse_int import mse_int_quantize


def mse_int8_quantize(
    weight: torch.Tensor,
    group_size: int = 128,
    n_candidates: int = 200,
    chunk_size: int = 1024,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D float weight to signed INT8 values and group scales."""
    return mse_int_quantize(
        weight,
        num_bits=8,
        group_size=group_size,
        n_candidates=n_candidates,
        chunk_size=chunk_size,
        scale_dtype=scale_dtype,
    )


register_op("mse_int8_quant")(mse_int8_quantize)
