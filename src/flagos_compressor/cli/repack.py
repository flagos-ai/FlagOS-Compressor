from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from flagos_compressor.formats.compressed_tensors import compressed_tensor_names
from flagos_compressor.formats.int8_pack import (
    pack_uint8b128_int32,
    unpack_uint8b128_int32,
)

logger = logging.getLogger(__name__)


def _json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def _digest(value: torch.Tensor) -> str:
    raw = value.contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _link_or_copy(source: str | Path, target: str | Path) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return str(target)


def repack_w8a16(source: Path, output: Path, *, dry_run: bool = False) -> dict:
    """Keep signed codes and FP32 scales exact while changing storage ABI.

    The destination uses ``pack-quantized`` uint8b128 words. This operation
    deliberately does not decode/requantize weights, so it can isolate the
    activation/runtime path in W8A8 versus W8A16 comparisons.
    """
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output must be separate, non-nested directories")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "compressed-tensors" or quant.get("format") != "int-quantized":
        raise ValueError("Repacking requires a compressed-tensors int-quantized source")
    groups = quant.get("config_groups", {})
    if not groups:
        raise ValueError("Source quantization config has no groups")
    for group in groups.values():
        weight = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        if not (weight.get("num_bits") == 8 and weight.get("type") == "int"
                and weight.get("strategy") == "channel" and weight.get("symmetric") is True
                and not weight.get("dynamic") and activation.get("num_bits") == 8
                and activation.get("type") == "int" and activation.get("dynamic") is True
                and activation.get("strategy") == "token"):
            raise ValueError("Only symmetric channelwise W8A8 INT8 sources are supported")
    manifest = json.loads((source / "quantization_manifest.json").read_text())
    selected = {name: entry for name, entry in manifest.get("tensors", {}).items()
                if entry.get("format") == "compressed-tensors-int-quantized-int8"}
    if not selected:
        raise ValueError("Manifest contains no channelwise INT8 weights")
    unsupported = [name for name, entry in manifest.get("tensors", {}).items()
                   if entry.get("num_bits") in (4, 8) and name not in selected]
    if unsupported:
        raise ValueError(f"Mixed or fused integer formats are unsupported: {unsupported[:3]}")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    for name, entry in selected.items():
        if (not name.endswith(".weight") or entry.get("strategy") != "channel"
                or name not in weight_map or entry.get("scale") not in weight_map):
            raise ValueError(f"Invalid channelwise source manifest entry: {name}")
    scale_names = {entry["scale"]: tuple(entry["logical_shape"])
                   for entry in selected.values()}
    # Validate every shard path before creating output files.
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        if Path(shard).name != shard or not (source / shard).is_file():
            raise ValueError(f"Invalid shard path: {shard}")
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            mapped = {name for name, filename in weight_map.items() if filename == shard}
            if set(handle.keys()) != mapped:
                raise ValueError(f"Shard/index tensor mismatch: {shard}")
    report = {"operation": "repack_w8a8_to_w8a16", "producer": "flagos-compressor",
              "input": str(source), "output": str(output), "status": "planned",
              "weights": len(selected), "shards": len(shards),
              "requantized": False, "codes_verified": 0, "unchanged_verified": 0,
              "fp32_verified": 0, "started_at": time.time()}
    if dry_run:
        return report
    output.mkdir(parents=True)
    report["status"] = "writing"
    _json(output / "repack_report.json", report)
    new_map = {}
    total_size = 0
    new_entries = {}
    try:
        for shard in shards:
            tensors, preserved = {}, {}
            with safe_open(source / shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if weight_map.get(name) != shard:
                        raise ValueError(f"Shard/index mismatch: {name}")
                    value = handle.get_tensor(name)
                    if name in selected:
                        entry = selected[name]
                        if value.dtype != torch.int8 or value.dim() != 2:
                            raise ValueError(f"Expected a 2D signed INT8 tensor: {name}")
                        if tuple(value.shape) != tuple(entry.get("logical_shape", ())):
                            raise ValueError(f"Manifest weight shape does not match tensor: {name}")
                        if value.shape[1] % 4:
                            raise ValueError(f"W8A16 packing requires input size divisible by four: {name}")
                        names = compressed_tensor_names(name)
                        for generated in [names.weight, names.shape]:
                            if generated in weight_map or generated in tensors:
                                raise ValueError(f"Generated name already exists: {generated}")
                        packed = pack_uint8b128_int32(value)
                        if not torch.equal(unpack_uint8b128_int32(packed), value):
                            raise RuntimeError(f"INT8 code roundtrip mismatch: {name}")
                        tensors[names.weight] = packed
                        tensors[names.shape] = torch.tensor(value.shape, dtype=torch.int64)
                        updated = copy.deepcopy(entry)
                        updated.update(format="compressed-tensors-pack-quantized-int8",
                                       storage_dtype="int32", storage_shape=list(packed.shape),
                                       activation_num_bits=16, shape=names.shape,
                                       group_size=None, rule="repack_existing_int8")
                        new_entries[names.weight] = updated
                        report["codes_verified"] += 1
                    else:
                        if name in scale_names:
                            rows, _ = scale_names[name]
                            if (value.dtype != torch.float32 or tuple(value.shape) != (rows, 1)
                                    or not torch.isfinite(value).all() or not (value > 0).all()):
                                raise ValueError(f"Expected finite positive FP32 channel scales: {name}")
                        if str(value.dtype).startswith("torch.float8"):
                            raise ValueError(f"Source contains residual FP8: {name}")
                        tensors[name] = value
                        preserved[name] = (_digest(value), value.dtype)
                        if name in manifest.get("tensors", {}):
                            new_entries[name] = copy.deepcopy(manifest["tensors"][name])
                temporary = output / (shard + ".tmp")
                save_file(tensors, temporary, metadata={"format": "pt"})
                temporary.replace(output / shard)
                with safe_open(output / shard, framework="pt", device="cpu") as saved:
                    for name, (expected, dtype) in preserved.items():
                        value = saved.get_tensor(name)
                        if value.dtype != dtype or _digest(value) != expected:
                            raise RuntimeError(f"Preserved tensor changed: {name}")
                        report["unchanged_verified"] += 1
                        report["fp32_verified"] += dtype == torch.float32
                for name, value in tensors.items():
                    new_map[name] = shard
                    total_size += value.numel() * value.element_size()
            del tensors
            report["completed_shards"] = len(set(new_map.values()))
            _json(output / "repack_report.json", report)
            logger.info("Repacked %s (%d/%d shards)", shard, report["completed_shards"], len(shards))
        if report["codes_verified"] != len(selected):
            raise ValueError("Not all manifest weights were found in source shards")
        excluded = {"config.json", "model.safetensors.index.json", "quantization_manifest.json",
                    "quantization_report.json", "repack_report.json"}
        for path in source.iterdir():
            if (path.name in excluded or path.name.startswith(".")
                    or "validation" in path.name or "audit" in path.name
                    or path.suffix == ".safetensors"):
                continue
            target = output / path.name
            if path.is_dir():
                shutil.copytree(path, target, copy_function=_link_or_copy)
            else:
                _link_or_copy(path, target)
        quant["format"] = "pack-quantized"
        for group in groups.values():
            group["input_activations"] = None
            group["output_activations"] = None
            group["format"] = "pack-quantized"
            group["weights"]["group_size"] = -1
        manifest["tensors"] = new_entries
        manifest["algorithm"] = {"name": "repack_existing_int8", "num_bits": 8,
                                 "activation_num_bits": 16, "strategy": "channel"}
        manifest["artifact"] = {"format": "compressed-tensors", "compression_format": "pack-quantized",
                                "weight_encoding": "uint8b128", "num_bits": 8,
                                "activation_num_bits": 16, "strategy": "channel",
                                "scale_layout": "row_channel", "scale_dtype": "float32"}
        manifest["repack_source"] = {"manifest_sha256": hashlib.sha256(
            (source / "quantization_manifest.json").read_bytes()).hexdigest(), "requantized": False}
        _json(output / "config.json", config)
        _json(output / "model.safetensors.index.json", {
            "metadata": dict(index.get("metadata", {}), total_size=total_size), "weight_map": new_map})
        _json(output / "quantization_manifest.json", manifest)
        report.update(status="complete", total_size=total_size, finished_at=time.time())
        _json(output / "repack_report.json", report)
        return report
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        _json(output / "repack_report.json", report)
        raise


def run(args) -> None:
    report = repack_w8a16(Path(args.input), Path(args.output), dry_run=args.dry_run)
    print(json.dumps(report, indent=2))
