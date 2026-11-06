import pytest
import torch

import flagos_compressor.quantizers.gptq as gptq_module
from flagos_compressor.quantizers.gptq import GPTQQuantizer, _inverse_hessian_factor


@pytest.mark.parametrize("size,samples", [(17, 34), (64, 16), (129, 258)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reverse_factor_matches_inverse_and_preserves_column_order(size, samples, dtype):
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(size, samples, generator=generator, dtype=dtype)
    hessian = inputs @ inputs.T / samples
    hessian.diagonal().add_(0.01 * hessian.diagonal().mean())
    original = hessian.clone()

    factor = _inverse_hessian_factor(hessian, reverse=True)

    # Independent double-precision inverse, including rank-deficient inputs
    # made positive definite by GPTQ's usual damping.
    inverse = torch.linalg.solve(
        hessian.double(), torch.eye(size, dtype=torch.float64)
    )
    expected = torch.linalg.cholesky(inverse).T
    tolerance = 5e-5 if dtype == torch.float32 else 1e-11
    assert ((factor.double() - expected).norm() / expected.norm()) < tolerance
    assert ((factor.double().T @ factor.double() - inverse).norm() / inverse.norm()) < tolerance
    assert torch.count_nonzero(factor.tril(-1)) == 0
    assert torch.all(factor.diagonal() > 0)
    torch.testing.assert_close(hessian, original, rtol=0, atol=0)


def test_reverse_factor_does_not_call_explicit_inverse(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit inverse must not be used")

    monkeypatch.setattr(torch, "cholesky_inverse", forbidden)
    monkeypatch.setattr(torch, "cholesky_solve", forbidden)
    monkeypatch.setattr(torch.linalg, "inv", forbidden)
    hessian = torch.tensor([[4.0, 1.0], [1.0, 3.0]])

    factor = _inverse_hessian_factor(hessian, reverse=True)

    torch.testing.assert_close(
        factor.T @ factor @ hessian, torch.eye(2), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_non_positive_hessian_keeps_actionable_error(reverse):
    with pytest.raises(RuntimeError, match="increase damp_percent"):
        _inverse_hessian_factor(torch.diag(torch.tensor([1.0, -1.0])), reverse=reverse)


@pytest.mark.parametrize("bits,desc_act,static_groups", [(4, False, False), (4, True, True), (8, True, False)])
def test_npu_gptq_matches_legacy_without_cholesky_inverse(monkeypatch, bits, desc_act, static_groups):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is unavailable")
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(8, 16, generator=generator)
    inputs = torch.randn(2, 32, 16, generator=generator)
    inputs[..., 3] = 0  # Exercise dead-column repair before damping.
    options = dict(group_size=8, block_size=8, desc_act=desc_act, static_groups=static_groups)
    # Compare the factorization change on the same device: existing CPU/NPU
    # numerical differences are independent of the inverse implementation.
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            gptq_module,
            "_inverse_hessian_factor",
            lambda hessian, **kwargs: _inverse_hessian_factor(hessian),
        )
        baseline = GPTQQuantizer(weight.to("npu"), bits=bits)
        baseline.add_batch(inputs.to("npu"))
        expected = baseline.quantize(**options)

    def forbidden(*args, **kwargs):
        raise AssertionError("NPU GPTQ must not call cholesky_inverse")

    monkeypatch.setattr(torch, "cholesky_inverse", forbidden)
    quantizer = GPTQQuantizer(weight.to("npu"), bits=bits)
    quantizer.add_batch(inputs.to("npu"))
    actual = quantizer.quantize(**options)
    torch.npu.synchronize()

    assert actual.weight.device.type == "npu"
    torch.testing.assert_close(actual.weight.cpu(), expected.weight.cpu(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.scales.cpu(), expected.scales.cpu(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.zeros.cpu(), expected.zeros.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual.g_idx.cpu(), expected.g_idx.cpu(), rtol=0, atol=0)
