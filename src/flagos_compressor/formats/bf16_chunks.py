"""Bound accelerator memory when decoding large FP8/FP4 tables to CPU BF16."""

import torch


def iter_bf16_chunks(
    weight,
    scale,
    input_format,
    backend,
    context,
    params,
    max_chunk_elements=16 * 1024 * 1024,
):
    if weight.ndim != 2 or scale is None:
        raise ValueError("Chunked dequantization requires a scaled 2D weight")
    if params.get("qkv_groups"):
        raise ValueError("Grouped QKV decoding must retain its complete row layout")
    if input_format.name == "fp8_block_e8m0":
        block = params.get("block_size", 128)
        bm, bn = (block, block) if isinstance(block, int) else block
        if bm <= 0 or bn <= 0:
            raise ValueError("FP8 block sizes must be positive")
        columns = weight.shape[1]
        expected = ((weight.shape[0] + bm - 1) // bm, (columns + bn - 1) // bn)
    elif input_format.name == "fp4_e2m1_e8m0":
        bm = 1
        columns = weight.shape[1] * 2
        expected = (weight.shape[0], columns // 32)
    else:
        raise ValueError(f"Unsupported chunked source format: {input_format.name}")
    if scale.numel() != expected[0] * expected[1] or (
        scale.ndim != 1 and tuple(scale.shape) != expected
    ):
        raise ValueError("Source scale grid does not cover the weight")
    scale = scale.reshape(expected)
    rows = max(bm, max_chunk_elements // max(columns, 1) // bm * bm)
    for start in range(0, weight.shape[0], rows):
        end = min(start + rows, weight.shape[0])
        value = input_format.to_canonical(
            weight[start:end],
            scale[start // bm : (end + bm - 1) // bm],
            backend,
            context,
            params,
        )
        yield start, value.to(device="cpu", dtype=torch.bfloat16).contiguous()


def dequantize_bf16_cpu(weight, scale, input_format, backend, context, params):
    columns = weight.shape[1] * (2 if input_format.name == "fp4_e2m1_e8m0" else 1)
    result = torch.empty((weight.shape[0], columns), dtype=torch.bfloat16, device="cpu")
    for start, chunk in iter_bf16_chunks(
        weight, scale, input_format, backend, context, params
    ):
        result[start : start + chunk.shape[0]].copy_(chunk)
    return result
