from __future__ import annotations

import json
from pathlib import Path

from flagos_compressor.core.profile import ModelProfile, TensorInfo
from flagos_compressor.inspect.source_layouts import fp8_source_params
from flagos_compressor.inspect.scale_pairing import build_scale_map
from flagos_compressor.inspect.tensor_classifier import (
    classify_weight,
    infer_logical_shape,
    infer_storage_format,
)
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


_CURRENT_MANIFEST = "quantization_manifest.json"
_LEGACY_MANIFEST = "quant_manifest.json"

_HEADER_DTYPES = {
    "BOOL": ("bool", 1), "U8": ("uint8", 1), "I8": ("int8", 1),
    "I16": ("int16", 2), "U16": ("uint16", 2), "I32": ("int32", 4),
    "U32": ("uint32", 4), "I64": ("int64", 8), "U64": ("uint64", 8),
    "F16": ("float16", 2), "BF16": ("bfloat16", 2),
    "F32": ("float32", 4), "F64": ("float64", 8),
    "F8_E4M3": ("float8_e4m3fn", 1), "F8_E5M2": ("float8_e5m2", 1),
    "F8_E8M0": ("float8_e8m0fnu", 1),
}


def _load_manifest(model_dir: Path) -> tuple[dict, str | None, str | None]:
    """Load current metadata, falling back to the legacy artifact manifest.

    Legacy INT4 is no longer a supported input format, but retaining its
    declared tensor format prevents the scanner from guessing that its
    byte-packed weights are MXFP4. The planner can then reject the unsupported
    format explicitly instead of decoding it with the wrong codec.
    """
    current_path = model_dir / _CURRENT_MANIFEST
    legacy_path = model_dir / _LEGACY_MANIFEST
    if current_path.exists():
        with current_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest, manifest.get("schema"), _CURRENT_MANIFEST
    if legacy_path.exists():
        with legacy_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest, manifest.get("abi_version"), _LEGACY_MANIFEST
    return {}, None, None


def scan_hf_safetensors(model_path: str | Path) -> ModelProfile:
    model_dir = Path(model_path)
    checkpoint = HfSafetensorsCheckpoint(model_dir)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    scale_map = build_scale_map(set(checkpoint.weight_map.keys()))
    manifest, manifest_abi, manifest_filename = _load_manifest(model_dir)
    manifest_specs: dict[str, dict] = {
        **(manifest.get("preserved_tensors") or {}),
        **(manifest.get("tensors") or {}),
    }
    if manifest_specs:
        for weight_name, spec in manifest_specs.items():
            scale_name = spec.get("scale")
            if weight_name in checkpoint.weight_map and scale_name in checkpoint.weight_map:
                scale_map[weight_name] = scale_name
    scale_targets = set(scale_map.values())
    shape_targets = {
        spec["shape"]
        for spec in manifest_specs.values()
        if spec.get("shape") in checkpoint.weight_map
    }

    profile = ModelProfile(
        model_path=str(model_dir),
        format="hf_safetensors",
        metadata={
            "num_index_keys": len(checkpoint.weight_map),
            "num_shards": len(checkpoint.shard_files()),
            "quantization_manifest_schema": manifest_abi,
            "quantization_manifest_file": manifest_filename,
            "source_quantization_config": config.get("quantization_config"),
        },
    )

    shapes: dict[str, tuple[int, ...]] = {}
    raw: dict[str, dict] = {}
    for tensor_name, shard_name, header in checkpoint.iter_tensor_metadata():
        shape = tuple(header["shape"])
        dtype, element_size = _HEADER_DTYPES[header["dtype"]]
        shapes[tensor_name] = shape
        raw[tensor_name] = {"shape": shape, "dtype": dtype, "shard": shard_name,
                            "element_size": element_size}

    for tensor_name, info in raw.items():
        scale_name = scale_map.get(tensor_name)
        role = (
            "scale"
            if tensor_name in scale_targets
            else "shape"
            if tensor_name in shape_targets
            else "weight"
        )
        storage_format = None
        storage_params = {}
        manifest_spec = manifest_specs.get(tensor_name)
        if role == "weight" and scale_name:
            if manifest_spec and manifest_spec.get("format"):
                storage_format = manifest_spec["format"]
                storage_params = dict(manifest_spec.get("storage_params") or {})
            elif info["dtype"] in {"float8_e4m3fn", "float8_e5m2"}:
                params = fp8_source_params(tensor_name, info["shape"], shapes.get(scale_name), config)
                if params is not None:
                    storage_format = "fp8_block_e8m0"
                    storage_params = params
            else:
                storage_format = infer_storage_format(
                    info["shape"],
                    info["element_size"],
                    shapes.get(scale_name),
                )
        logical_name = (
            manifest_spec.get("logical_name", tensor_name)
            if manifest_spec
            else tensor_name
        )
        module_kind, tags = (
            classify_weight(logical_name) if role == "weight" else (None, ())
        )
        logical_shape = (
            tuple(manifest_spec["logical_shape"])
            if manifest_spec and manifest_spec.get("logical_shape")
            else infer_logical_shape(info["shape"], storage_format)
        )
        profile.tensors[tensor_name] = TensorInfo(
            name=tensor_name,
            shape=info["shape"],
            dtype=info["dtype"],
            shard=info["shard"],
            element_size=info["element_size"],
            scale_name=scale_name,
            role=role,
            storage_format=storage_format,
            logical_shape=logical_shape,
            module_kind=module_kind,
            tags=tags,
            storage_params=storage_params,
        )

    return profile
