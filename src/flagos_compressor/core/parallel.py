"""Execute independent checkpoint shards in one spawned process per device."""
from __future__ import annotations

import json
import hashlib
import logging
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import traceback

import torch
from tqdm import tqdm

from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.executor import (
    _build_action_map, _build_scale_names, _prepare_output,
    execute_shard, finalize_execution, merge_shard_result,
)
from flagos_compressor.core.plan import ExecutionPlan
from flagos_compressor.core.report import ConversionReport
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint

logger = logging.getLogger(__name__)


def normalize_devices(backend: str, devices: list[str]) -> list[str]:
    """Accept device names or comma/space-separated indices without touching hardware."""
    normalized = []
    for item in devices:
        for value in item.split(","):
            value = value.strip()
            if not value:
                raise ValueError("--devices contains an empty device")
            name = f"{backend}:{value}" if value.isdigit() else value
            # Load optional device extensions before resolving names such as npu.
            candidate = build_backend(backend, name)
            device = candidate.device if backend != "cpu" else torch.device(name)
            if device.type != backend:
                raise ValueError(f"Device {name!r} does not match --backend {backend}")
            canonical = f"{device.type}:{device.index if device.index is not None else 0}"
            if canonical in normalized:
                raise ValueError(f"Duplicate parallel device: {canonical}")
            normalized.append(canonical)
    if not normalized:
        raise ValueError("At least one parallel device is required")
    return normalized


def _check_device(backend_name: str, device: str):
    backend = build_backend(backend_name, device, fallback_policy="error")
    if not backend.is_available():
        raise RuntimeError(f"Parallel device {device} is unavailable")
    # Also check the index: is_available() alone only checks the device type.
    torch.empty(0, device=backend.device)
    return backend


def _source_snapshot(checkpoint: HfSafetensorsCheckpoint) -> dict:
    source = checkpoint.model_path
    metadata = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("config.json", "model.safetensors.index.json")
        if (source / name).exists()
    }
    shards = {}
    for name in checkpoint.shard_files():
        stat = (source / name).stat()
        shards[name] = (stat.st_size, stat.st_mtime_ns)
    return {"metadata_sha256": metadata, "shards": shards}


def _worker(source, output, plan, backend_name, device, tasks, results):
    shard = None
    try:
        import flagos_compressor.formats.register  # noqa: F401
        import flagos_compressor.quantizers.register  # noqa: F401

        torch.set_num_threads(2)
        backend = _check_device(backend_name, device)
        runtime = torch.get_device_module(backend.device)
        if callable(getattr(runtime, "set_device", None)):
            runtime.set_device(backend.device)
        checkpoint = HfSafetensorsCheckpoint(source)
        action_map, scale_names = _build_action_map(plan), _build_scale_names(plan)
        while (shard := tasks.get()) is not None:
            result = execute_shard(checkpoint, Path(output), shard, action_map, scale_names, backend)
            results.put(("result", device, result))
    except BaseException:
        results.put(("error", device, {"shard": shard, "traceback": traceback.format_exc()}))
        raise


def execute_plan_parallel(
    input_path: str | Path,
    output_path: str | Path,
    plan: ExecutionPlan,
    *,
    backend: str,
    devices: list[str],
) -> ConversionReport:
    """Run an MSE/conversion plan, merge once, and validate before marking complete.

    Each source shard is owned by one worker. Read-only scale dependencies may
    cross shards. A shard (including a fused expert bank) is never split across
    devices. This is checkpoint parallelism, not calibrated layer parallelism.
    """
    devices = normalize_devices(backend, devices)
    algorithm = plan.metadata.get("algorithm", {})
    if algorithm.get("name") not in (None, "mse_grid_search", "per_selector_mse_grid_search"):
        raise ValueError("Parallel checkpoint execution supports MSE and BF16 conversion only")
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Parallel output must be separate from and outside the input directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Parallel output must be a new or empty directory: {output}")
    if (source / "_INCOMPLETE").exists():
        raise ValueError(f"Source checkpoint is marked incomplete: {source}")
    checkpoint = HfSafetensorsCheckpoint(source)
    shards = sorted(checkpoint.shard_files(),
                    key=lambda s: (source / s).stat().st_size, reverse=True)
    if not shards:
        raise ValueError("Source checkpoint has no shards")
    source_snapshot = _source_snapshot(checkpoint)
    # Reject unavailable devices before creating any output. Spawn keeps parent
    # accelerator initialization out of child runtimes.
    for device in devices:
        _check_device(backend, device)
    active_devices = devices[:min(len(devices), len(shards))]
    if len(active_devices) < len(devices):
        logger.warning("Only %d source shards; using %d of %d requested devices",
                       len(shards), len(active_devices), len(devices))
    report = ConversionReport(backend=backend)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "_INCOMPLETE"
    with marker.open("x") as handle:
        handle.write("Parallel execution is in progress; this output is not ready.\n")
    context = mp.get_context("spawn")
    tasks, results = context.Queue(), context.Queue()
    workers = []
    succeeded = False
    try:
        _prepare_output(source, output)
        # Auxiliary source files must not replace the current run's marker.
        marker.write_text("Parallel execution is in progress; this output is not ready.\n")
        for shard in shards:
            tasks.put(shard)
        for _ in active_devices:
            tasks.put(None)
        for device in active_devices:
            process = context.Process(target=_worker, args=(
                str(source), str(output), plan, backend, device, tasks, results,
            ))
            process.start()
            workers.append(process)
        mapping: dict[str, str] = {}
        total_size = generated = 0
        completed: set[str] = set()
        per_device = {device: {"shards": 0, "converted": 0} for device in active_devices}
        with tqdm(total=len(shards), desc="Parallel shards") as progress:
            while len(completed) < len(shards):
                try:
                    kind, device, payload = results.get(timeout=1)
                except Empty:
                    failed = [(p.pid, p.exitcode) for p in workers if p.exitcode not in (None, 0)]
                    if failed or not any(p.is_alive() for p in workers):
                        raise RuntimeError(f"Parallel workers exited before completion: {failed}")
                    continue
                if kind == "error":
                    raise RuntimeError(f"Worker {device} failed on {payload['shard']}:\n{payload['traceback']}")
                if payload.shard in completed:
                    raise ValueError(f"Duplicate shard result: {payload.shard}")
                merge_shard_result(report, mapping, payload)
                completed.add(payload.shard)
                total_size += payload.total_size
                generated += payload.generated_tensors
                per_device[device]["shards"] += 1
                per_device[device]["converted"] += payload.report.converted
                progress.update()
        for process in workers:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Parallel worker {process.pid} exited with {process.exitcode}")
        if _source_snapshot(checkpoint) != source_snapshot:
            raise RuntimeError("Source checkpoint changed during parallel execution")
        report.extras["source_metadata_sha256"] = source_snapshot["metadata_sha256"]
        report.extras["parallel"] = {
            "requested_devices": devices, "devices": active_devices,
            "workers": len(workers), "source_shards": len(shards),
            "per_device": per_device,
        }
        finalize_execution(checkpoint, output, plan, report, mapping, total_size, generated)
        validation = validate_artifact(output, allow_incomplete=True)
        if not validation["valid"]:
            raise RuntimeError(f"Parallel artifact validation failed: {validation['errors'][:8]}")
        report.extras["validation"] = validation
        report.finish()
        report.save(output / report.extras["report_file"])
        marker.unlink()
        succeeded = True
        return report
    except BaseException:
        (output / "execution_error.json").write_text(json.dumps({"traceback": traceback.format_exc()}, indent=2))
        raise
    finally:
        if not succeeded:
            for process in workers:
                if process.is_alive():
                    process.terminate()
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        # Unconsumed jobs must not keep the parent queue's feeder alive on error.
        tasks.cancel_join_thread()
        tasks.close()
        results.close()
