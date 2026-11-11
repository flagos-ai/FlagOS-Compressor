import pytest
import torch

import flagos_compressor.formats.register  # noqa: F401
from flagos_compressor.backends.base import BackendRunContext
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.report import ConversionReport
from flagos_compressor.formats.base import get_weight_format
from flagos_compressor.formats.bf16_chunks import iter_bf16_chunks


def test_previous_dequantization_name_remains_importable():
    from flagos_compressor.formats.bf16_chunks import (
        dequantize_bf16_cpu, dequantize_bf16_chunked_to_cpu,
    )

    assert dequantize_bf16_cpu is dequantize_bf16_chunked_to_cpu


@pytest.fixture(params=["cpu", "cuda"])
def backend(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    return build_backend(request.param, fallback_policy="error")


@pytest.mark.parametrize("block", [(1, 32), (32, 32), (4, 8)])
def test_bounded_fp8_chunks_match_independent_dequantization(block, backend):
    bm, bn = block
    rows, cols = 67, 65
    torch.manual_seed(19)
    weight = torch.randn(rows, cols).to(torch.float8_e4m3fn)
    scale = torch.rand((rows + bm - 1) // bm, (cols + bn - 1) // bn)
    report = ConversionReport(backend=backend.name)
    chunks = list(
        iter_bf16_chunks(
            weight,
            scale,
            get_weight_format("fp8_block_e8m0"),
            backend,
            BackendRunContext(report=report),
            {"block_size": block},
            max_chunk_elements=500,
        )
    )
    actual = torch.cat([chunk for _, chunk in chunks])
    reference = (
        weight.float()
        * scale.repeat_interleave(bm, 0).repeat_interleave(bn, 1)[:rows, :cols]
    ).bfloat16()
    assert len(chunks) > 1
    assert all(start % bm == 0 for start, _ in chunks)
    assert all(chunk.device.type == "cpu" and chunk.dtype == torch.bfloat16
               for _, chunk in chunks)
    assert len(report.op_events) == len(chunks)
    assert all(event.actual_backend == backend.name and not event.fallback
               for event in report.op_events)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_bounded_fp4_chunks_preserve_nibble_order(backend):
    weight = torch.full((11, 16), 0x72, dtype=torch.uint8)
    scale = torch.full((11, 1), 2.0)
    report = ConversionReport(backend=backend.name)
    chunks = list(
        iter_bf16_chunks(
            weight,
            scale,
            get_weight_format("fp4_e2m1_e8m0"),
            backend,
            BackendRunContext(report=report),
            {},
            max_chunk_elements=64,
        )
    )
    actual = torch.cat([chunk for _, chunk in chunks])
    assert len(chunks) > 1
    assert all(chunk.device.type == "cpu" and chunk.dtype == torch.bfloat16
               for _, chunk in chunks)
    assert len(report.op_events) == len(chunks)
    assert all(event.actual_backend == backend.name and not event.fallback
               for event in report.op_events)
    expected = torch.tensor([2.0, 12.0], dtype=torch.bfloat16).repeat(11, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chunking_rejects_transposed_scale_grid_with_equal_size():
    weight = torch.ones(32, 64).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="scale grid"):
        list(
            iter_bf16_chunks(
                weight,
                torch.ones(2, 1),
                get_weight_format("fp8_block_e8m0"),
                build_backend("cpu"),
                BackendRunContext(),
                {"block_size": [32, 32]},
            )
        )
