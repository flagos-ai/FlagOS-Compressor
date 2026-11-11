"""Shared CLI execution selection for checkpoint conversion and MSE quantization."""
from __future__ import annotations

import logging

from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.parallel import execute_plan_parallel, normalize_devices
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint

logger = logging.getLogger(__name__)


def parallel_devices(args, *, method: str = "mse") -> list[str] | None:
    values = getattr(args, "devices", None)
    if values is None:
        return None
    if method != "mse":
        raise ValueError(
            f"--devices supports MSE checkpoint quantization and BF16 conversion; "
            f"{method.upper()} uses sequential calibration. Use --device for {method.upper()}."
        )
    return normalize_devices(args.backend, values)


def print_parallel_plan(args, devices):
    if devices:
        count = len(HfSafetensorsCheckpoint(args.input).shard_files())
        print(f"  parallel source shards: {count}")
        print(f"  parallel devices: {', '.join(devices)}")
        print(f"  parallel workers: {min(count, len(devices))}")


def execute_checkpoint(args, plan, devices):
    if devices:
        return execute_plan_parallel(args.input, args.output, plan,
                                     backend=args.backend, devices=devices)
    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        logger.warning("backend %r is unavailable; CPU fallback may be used.", backend.name)
    return execute_plan(args.input, args.output, plan, backend)
