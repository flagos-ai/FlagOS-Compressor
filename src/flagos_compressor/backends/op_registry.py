from __future__ import annotations

from collections.abc import Callable
from typing import Any


OpFunc = Callable[..., Any]


class OpRegistry:
    """Named PyTorch operations; execution devices are managed by QuantBackend."""

    def __init__(self) -> None:
        self._ops: dict[str, OpFunc] = {}

    def register(self, op_name: str, func: OpFunc) -> None:
        self._ops[op_name] = func

    def resolve(self, op_name: str) -> OpFunc | None:
        return self._ops.get(op_name)


GLOBAL_OP_REGISTRY = OpRegistry()


def register_op(op_name: str) -> Callable[[OpFunc], OpFunc]:
    def decorator(func: OpFunc) -> OpFunc:
        GLOBAL_OP_REGISTRY.register(op_name, func)
        return func

    return decorator

