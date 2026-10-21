from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from flagos_compressor.backends.op_registry import GLOBAL_OP_REGISTRY
from flagos_compressor.core.report import ConversionReport, OpEvent


@dataclass
class BackendRunContext:
    report: ConversionReport | None = None


@dataclass
class QuantBackend:
    name: str
    device_name: str = "cpu"
    fallback_policy: str = "warn"
    op_placement: dict[str, str] = field(default_factory=dict)
    runtime_imports: tuple[str, ...] = ()

    def _load_runtime(self) -> bool:
        for module_name in self.runtime_imports:
            try:
                importlib.import_module(module_name)
            except (ImportError, OSError):
                return False
        return True

    def _device_module(self):
        device = self.device
        try:
            return torch.get_device_module(device)
        except (AttributeError, RuntimeError, ValueError):
            return getattr(torch, device.type, None)

    def is_available(self) -> bool:
        if not self._load_runtime():
            return False
        if self.device_name == "cpu":
            return True
        try:
            runtime = self._device_module()
        except (RuntimeError, TypeError, ValueError):
            return False
        available = getattr(runtime, "is_available", None)
        return bool(available()) if callable(available) else runtime is not None

    @property
    def device(self) -> torch.device:
        if not self._load_runtime():
            missing = ", ".join(self.runtime_imports)
            raise RuntimeError(f"Backend runtime is not installed: {missing}")
        try:
            return torch.device(self.device_name)
        except RuntimeError as exc:
            raise RuntimeError(
                f"PyTorch does not recognize device {self.device_name!r}; "
                f"install and import the {self.name!r} device extension"
            ) from exc

    def move(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device_name == "cpu":
            return tensor.cpu()
        return tensor.to(self.device)

    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu()

    def synchronize(self) -> None:
        if not self.is_available() or self.device_name == "cpu":
            return
        synchronize = getattr(self._device_module(), "synchronize", None)
        if callable(synchronize):
            synchronize()

    def empty_cache(self) -> None:
        if not self.is_available() or self.device_name == "cpu":
            return
        empty_cache = getattr(self._device_module(), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def run(
        self,
        op_name: str,
        *args: Any,
        context: BackendRunContext | None = None,
        **kwargs: Any,
    ) -> Any:
        op = GLOBAL_OP_REGISTRY.resolve(op_name)
        if op is None:
            raise NotImplementedError(f"No implementation registered for op '{op_name}'")

        placement = self.op_placement.get(op_name, "device")
        requested_backend = self.name if placement != "cpu" else "cpu"
        actual_backend = requested_backend
        fallback = False
        reason = None

        if requested_backend != "cpu" and not self.is_available():
            reason = f"backend '{self.name}' is not available"
            if self.fallback_policy == "error":
                raise RuntimeError(f"Cannot run op '{op_name}': {reason}")
            actual_backend = "cpu"
            fallback = True

        def execute(on_cpu: bool) -> Any:
            move = torch.Tensor.cpu if on_cpu else self.move
            run_args = tuple(move(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
            run_kwargs = {
                key: move(value) if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
            return op(*run_args, **run_kwargs)

        started = time.perf_counter()
        try:
            result = execute(actual_backend == "cpu")
        except RuntimeError as exc:
            # Retry runtime/transfer failures only. Input validation errors
            # (ValueError/TypeError) must propagate without a CPU retry.
            if actual_backend == "cpu" or self.fallback_policy == "error":
                raise
            fallback = True
            reason = f"{type(exc).__name__}: {exc}"
            actual_backend = "cpu"
            result = execute(on_cpu=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if context and context.report:
            context.report.record_op(
                OpEvent(
                    op_name=op_name,
                    requested_backend=requested_backend,
                    actual_backend=actual_backend,
                    fallback=fallback,
                    reason=reason,
                    elapsed_ms=elapsed_ms,
                )
            )

        return result
