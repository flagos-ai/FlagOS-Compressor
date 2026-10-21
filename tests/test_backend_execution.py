import pytest
import torch

import flagos_compressor.backends.base as backend_module
import flagos_compressor.backends.op_registry as registry_module
import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.base import BackendRunContext
from flagos_compressor.backends.op_registry import OpRegistry, register_op
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.report import ConversionReport


@pytest.fixture
def isolated_registry(monkeypatch):
    registry = OpRegistry()
    monkeypatch.setattr(backend_module, "GLOBAL_OP_REGISTRY", registry)
    monkeypatch.setattr(registry_module, "GLOBAL_OP_REGISTRY", registry)
    return registry


def _context():
    return BackendRunContext(report=ConversionReport(backend="cuda"))


def test_unknown_op_is_not_reported_as_device_fallback(isolated_registry):
    with pytest.raises(NotImplementedError, match="missing"):
        build_backend("cpu").run("missing", torch.ones(2))


@pytest.mark.parametrize("policy", ["warn", "error"])
def test_unavailable_device_obeys_fallback_policy(
    isolated_registry, monkeypatch, policy
):
    @register_op("add")
    def add(weight, *, scale):
        return weight + scale

    backend = build_backend("cuda", fallback_policy=policy)
    monkeypatch.setattr(backend, "is_available", lambda: False)
    context = _context()
    if policy == "error":
        with pytest.raises(RuntimeError, match="backend 'cuda' is not available"):
            backend.run("add", torch.ones(2), scale=torch.ones(2), context=context)
        assert not context.report.op_events
    else:
        result = backend.run("add", torch.ones(2), scale=torch.ones(2), context=context)
        torch.testing.assert_close(result, torch.full((2,), 2.0))
        (event,) = context.report.op_events
        assert (event.requested_backend, event.actual_backend, event.fallback) == (
            "cuda",
            "cpu",
            True,
        )
        assert event.reason == "backend 'cuda' is not available"


def test_explicit_cpu_placement_is_not_a_fallback(isolated_registry, monkeypatch):
    register_op("identity")(lambda weight: weight)
    backend = build_backend("cuda", fallback_policy="error")
    backend.op_placement["identity"] = "cpu"
    monkeypatch.setattr(backend, "is_available", lambda: False)
    context = _context()
    result = backend.run("identity", torch.ones(2), context=context)
    assert result.device.type == "cpu"
    (event,) = context.report.op_events
    assert (event.requested_backend, event.actual_backend, event.fallback) == (
        "cpu",
        "cpu",
        False,
    )
    assert event.reason is None


@pytest.mark.parametrize("policy", ["warn", "error"])
@pytest.mark.parametrize("failure_stage", ["transfer", "operation"])
def test_runtime_failure_obeys_fallback_policy(
    isolated_registry, monkeypatch, policy, failure_stage
):
    calls = []
    error = RuntimeError(f"{failure_stage} failed")

    @register_op("add")
    def add(weight, *, scale):
        calls.append(weight.device.type)
        if failure_stage == "operation" and len(calls) == 1:
            raise error
        return weight + scale

    def move(tensor):
        if failure_stage == "transfer":
            raise error
        return tensor

    # Exercise recovery without requiring a physical accelerator.
    backend = build_backend("cuda", fallback_policy=policy)
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "move", move)
    context = _context()
    if policy == "error":
        with pytest.raises(RuntimeError) as raised:
            backend.run("add", torch.ones(2), scale=torch.ones(2), context=context)
        assert raised.value is error
        assert len(calls) == (failure_stage == "operation")
        assert not context.report.op_events
    else:
        result = backend.run("add", torch.ones(2), scale=torch.ones(2), context=context)
        torch.testing.assert_close(result, torch.full((2,), 2.0))
        assert len(calls) == (2 if failure_stage == "operation" else 1)
        (event,) = context.report.op_events
        assert (event.requested_backend, event.actual_backend, event.fallback) == (
            "cuda",
            "cpu",
            True,
        )
        assert event.reason == f"RuntimeError: {failure_stage} failed"


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_input_errors_propagate_without_retry(
    isolated_registry, monkeypatch, error_type
):
    calls = []
    error = error_type("invalid weight")

    @register_op("invalid")
    def invalid(weight):
        calls.append(weight)
        raise error

    backend = build_backend("cuda")
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "move", lambda tensor: tensor)
    with pytest.raises(error_type) as raised:
        backend.run("invalid", torch.ones(2))
    assert raised.value is error
    assert len(calls) == 1


def test_cpu_runtime_failure_is_not_retried(isolated_registry):
    calls = []

    @register_op("invalid")
    def invalid(weight):
        calls.append(weight)
        raise RuntimeError("CPU execution failed")

    with pytest.raises(RuntimeError, match="CPU execution failed"):
        build_backend("cpu").run("invalid", torch.ones(2))
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_moves_positional_and_keyword_tensors_and_retries_on_cpu(
    isolated_registry,
):
    calls = []

    @register_op("cpu_only")
    def cpu_only(weight, *, scale):
        calls.append((weight.device.type, scale.device.type))
        if weight.is_cuda:
            raise NotImplementedError("test operation requires CPU")
        return weight + scale

    context = _context()
    # Different source devices ensure both directions of transfer are covered.
    result = build_backend("cuda").run(
        "cpu_only", torch.ones(2), scale=torch.ones(2, device="cuda"), context=context
    )
    torch.testing.assert_close(result, torch.full((2,), 2.0))
    assert calls == [("cuda", "cuda"), ("cpu", "cpu")]
    (event,) = context.report.op_events
    assert event.actual_backend == "cpu"
    assert event.fallback
    assert event.reason == "NotImplementedError: test operation requires CPU"


def _op_cases():
    weight = torch.tensor([[-1.0, -0.5, 0.5, 1.0] * 8])
    packed_fp4 = torch.full((1, 16), 0x21, dtype=torch.uint8)
    fp8 = torch.ones(1, 32, dtype=torch.float8_e4m3fn)
    e8m0 = torch.full((1, 1), 126, dtype=torch.uint8)
    return [
        ("e8m0_decode", (e8m0,), {}),
        ("fp4_unpack", (packed_fp4,), {}),
        ("fp4_dequant", (packed_fp4,), {"scale_e8m0": e8m0}),
        ("fp8_dequant", (fp8,), {"scale": e8m0, "block_size": 32}),
        ("mxfp8_dequant", (fp8,), {"scale": e8m0}),
        (
            "nvfp4_dequant",
            (packed_fp4,),
            {
                "scale": torch.ones(1, 2, dtype=torch.float8_e4m3fn),
                "global_scale": torch.tensor(2.0),
            },
        ),
        (
            "int4_pack_uint4b8_int32",
            (torch.arange(-8, 8, dtype=torch.int8).reshape(1, -1),),
            {},
        ),
        (
            "int4_unpack_uint4b8_int32",
            (torch.tensor([[-19088744]], dtype=torch.int32),),
            {},
        ),
        (
            "int8_pack_uint8b128_int32",
            (torch.arange(-128, 128, dtype=torch.int8).reshape(1, -1),),
            {},
        ),
        (
            "int8_unpack_uint8b128_int32",
            (torch.tensor([[-19088744]], dtype=torch.int32),),
            {},
        ),
        ("mse_int4_quant", (weight,), {"group_size": 32, "n_candidates": 8}),
        ("mse_int8_quant", (weight,), {"group_size": 32, "n_candidates": 8}),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "op_name,args,kwargs",
    _op_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_builtin_ops_match_cpu_on_cuda_without_fallback(op_name, args, kwargs):
    expected = build_backend("cpu", fallback_policy="error").run(
        op_name, *args, **kwargs
    )
    context = _context()
    actual = build_backend("cuda", fallback_policy="error").run(
        op_name, *args, context=context, **kwargs
    )
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    for actual_value, expected_value in zip(
        actual_values, expected_values, strict=True
    ):
        assert actual_value.device.type == "cuda"
        torch.testing.assert_close(actual_value.cpu(), expected_value)
    (event,) = context.report.op_events
    assert (event.requested_backend, event.actual_backend, event.fallback) == (
        "cuda",
        "cuda",
        False,
    )
    assert event.reason is None
