from __future__ import annotations

import json
from pathlib import Path

import torch

from flagos_compressor.core.dtypes import (
    dtype_name,
    parse_w8a8_scale_dtype,
)
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


def validate_artifact(model_path: str | Path) -> dict:
    path = Path(model_path)
    checkpoint = HfSafetensorsCheckpoint(path)
    errors: list[str] = []
    observed: set[str] = set()
    tensor_meta: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    logical_shape_values: dict[str, list[int]] = {}
    total_size = 0
    for shard_name, shard_path in checkpoint.iter_shards():
        if not shard_path.exists():
            errors.append(f"Missing shard: {shard_name}")
            continue
        state = checkpoint.load_shard(shard_name)
        expected = {name for name, shard in checkpoint.weight_map.items() if shard == shard_name}
        actual = set(state)
        if expected != actual:
            errors.append(
                f"Shard {shard_name} index mismatch: missing={sorted(expected-actual)[:3]}, "
                f"extra={sorted(actual-expected)[:3]}"
            )
        observed.update(actual)
        tensor_meta.update({name: (tuple(tensor.shape), tensor.dtype) for name, tensor in state.items()})
        logical_shape_values.update(
            {
                name: [int(value) for value in tensor.tolist()]
                for name, tensor in state.items()
                if name.endswith(".weight_shape")
                and tensor.dtype == torch.int64
                and tensor.shape == (2,)
            }
        )
        total_size += sum(t.numel() * t.element_size() for t in state.values())

    if observed != set(checkpoint.weight_map):
        errors.append("Checkpoint index keys do not match stored tensors")
    indexed_size = checkpoint.index.get("metadata", {}).get("total_size")
    if indexed_size is not None and int(indexed_size) != total_size:
        errors.append(f"metadata.total_size={indexed_size} but actual size is {total_size}")

    manifest_path = path / "quantization_manifest.json"
    int4_tensors = 0
    int8_tensors = 0
    observed_quant_bits: set[int] = set()
    observed_compression_formats: set[str] = set()
    observed_strategies: set[str] = set()
    observed_scale_dtypes: set[str] = set()
    native_method: str | None = None
    native_algorithm: str | None = None
    native_quantized_tensors = 0
    quantized_formats = {
        "compressed-tensors-pack-quantized-int4": (
            4,
            torch.int32,
            torch.bfloat16,
            True,
        ),
        "compressed-tensors-pack-quantized-int8": (
            8,
            torch.int32,
            torch.bfloat16,
            True,
        ),
        "compressed-tensors-int-quantized-int8": (8, torch.int8, None, False),
    }
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        schema = manifest.get("schema")
        if schema != "flagos-compressor.provenance.v1":
            errors.append("Unsupported quantization manifest schema")
        for name, spec in manifest.get("tensors", {}).items():
            tensor_format = spec.get("format")
            if name not in tensor_meta:
                errors.append(f"Manifest weight is missing: {name}")
                continue
            if tensor_format not in quantized_formats:
                continue
            num_bits, expected_weight_dtype, expected_scale_dtype, requires_shape = (
                quantized_formats[tensor_format]
            )
            if expected_scale_dtype is None:
                try:
                    expected_scale_dtype = parse_w8a8_scale_dtype(
                        spec.get("scale_dtype", "float32")
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(f"Invalid scale dtype for {name}: {exc}")
                    expected_scale_dtype = torch.float32
            observed_quant_bits.add(num_bits)
            observed_compression_formats.add(
                "int-quantized"
                if tensor_format == "compressed-tensors-int-quantized-int8"
                else "pack-quantized"
            )
            observed_strategies.add(spec.get("strategy", "group"))
            if spec.get("num_bits", num_bits) != num_bits:
                errors.append(
                    f"Manifest bit width mismatch for {name}: "
                    f"{spec.get('num_bits')} vs format INT{num_bits}"
                )
            if num_bits == 4:
                int4_tensors += 1
            else:
                int8_tensors += 1
            weight_shape, weight_dtype = tensor_meta[name]
            scale_name = spec.get("scale")
            if weight_dtype != expected_weight_dtype:
                errors.append(
                    f"INT{num_bits} weight {name} is {weight_dtype}, "
                    f"expected {expected_weight_dtype}"
                )
            if list(weight_shape) != spec.get("storage_shape"):
                errors.append(f"INT{num_bits} storage shape mismatch for {name}")
            if scale_name not in tensor_meta:
                errors.append(
                    f"INT{num_bits} scale is missing for {name}: {scale_name}"
                )
            else:
                scale_shape, scale_dtype = tensor_meta[scale_name]
                observed_scale_dtypes.add(dtype_name(scale_dtype) or "unknown")
                if scale_dtype != expected_scale_dtype:
                    errors.append(
                        f"INT{num_bits} scale {scale_name} is {scale_dtype}, "
                        f"expected {expected_scale_dtype}"
                    )
                if list(scale_shape) != spec.get("scale_shape"):
                    errors.append(
                        f"INT{num_bits} scale shape mismatch for {name}"
                    )
            if requires_shape:
                shape_name = spec.get("shape")
                if shape_name not in tensor_meta:
                    errors.append(
                        f"INT{num_bits} logical shape tensor is missing: {shape_name}"
                    )
                else:
                    shape_shape, shape_dtype = tensor_meta[shape_name]
                    if shape_shape != (2,) or shape_dtype != torch.int64:
                        errors.append(
                            f"INT{num_bits} logical shape tensor {shape_name} "
                            "must be int64[2]"
                        )
                    elif logical_shape_values.get(shape_name) != spec.get(
                        "logical_shape"
                    ):
                        errors.append(
                            f"INT{num_bits} logical shape tensor {shape_name} "
                            "has the wrong value"
                        )

        config_path = path / "config.json"
        artifact = manifest.get("artifact") or {}
        if observed_quant_bits:
            sorted_bits = sorted(observed_quant_bits)
            mixed_bits = len(sorted_bits) > 1
            mixed_storage = len(observed_compression_formats) > 1
            expected_bits = sorted_bits if mixed_bits else sorted_bits[0]
            if artifact.get("num_bits") != expected_bits:
                errors.append("Manifest artifact bit width is inconsistent")
            expected_compression_format = (
                "mixed-precision"
                if mixed_storage
                else next(iter(observed_compression_formats))
            )
            if artifact.get("compression_format") != expected_compression_format:
                errors.append(
                    "Manifest artifact compression format is inconsistent"
                )
            if mixed_storage:
                if artifact.get("weight_encoding") != "mixed":
                    errors.append(
                        "Manifest mixed-precision weight encoding is inconsistent"
                    )
                if artifact.get("compression_formats") != sorted(
                    observed_compression_formats
                ):
                    errors.append(
                        "Manifest mixed-precision format list is inconsistent"
                    )
                if not artifact.get("schemes"):
                    errors.append(
                        "Manifest mixed-precision artifact is missing schemes"
                    )
            elif mixed_bits:
                expected_encodings = {
                    str(bits): "uint4b8" if bits == 4 else "uint8b128"
                    for bits in sorted_bits
                }
                if artifact.get("weight_encoding") != "mixed":
                    errors.append(
                        "Manifest mixed-bit weight encoding is inconsistent"
                    )
                if artifact.get("weight_encodings") != expected_encodings:
                    errors.append(
                        "Manifest mixed-bit encoding map is inconsistent"
                    )
                expected_values_per_word = {
                    str(bits): 32 // bits for bits in sorted_bits
                }
                if artifact.get("values_per_word") != expected_values_per_word:
                    errors.append(
                        "Manifest mixed-bit packing factors are inconsistent"
                    )
            else:
                manifest_bits = sorted_bits[0]
                expected_encoding = (
                    "int8"
                    if artifact.get("compression_format") == "int-quantized"
                    else ("uint4b8" if manifest_bits == 4 else "uint8b128")
                )
                if artifact.get("weight_encoding") != expected_encoding:
                    errors.append(
                        "Manifest artifact weight encoding is inconsistent"
                    )
        if observed_strategies and "strategy" in artifact:
            expected_strategy = (
                next(iter(observed_strategies))
                if len(observed_strategies) == 1
                else "mixed"
            )
            if artifact.get("strategy") != expected_strategy:
                errors.append(
                    "Manifest artifact weight strategy is inconsistent"
                )
        if "scale_dtype" in artifact:
            if len(observed_scale_dtypes) > 1:
                if artifact.get("scale_dtype") != "mixed":
                    errors.append(
                        "Manifest artifact scale dtype must be mixed"
                    )
                if artifact.get("scale_dtypes") != sorted(
                    observed_scale_dtypes
                ):
                    errors.append(
                        "Manifest artifact scale dtype list is inconsistent"
                    )
            else:
                try:
                    artifact_scale_dtype = dtype_name(
                        parse_w8a8_scale_dtype(artifact["scale_dtype"])
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(f"Invalid artifact scale dtype: {exc}")
                else:
                    if (
                        len(observed_scale_dtypes) == 1
                        and artifact_scale_dtype
                        != next(iter(observed_scale_dtypes))
                    ):
                        errors.append(
                            "Manifest artifact scale dtype is inconsistent"
                        )
        if not config_path.exists():
            errors.append("compressed-tensors artifact is missing config.json")
        else:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            quant_config = config.get("quantization_config") or {}
            if quant_config.get("quant_method") != "compressed-tensors":
                errors.append("config.json does not declare compressed-tensors")
            expected_compression_format = artifact.get(
                "compression_format", "pack-quantized"
            )
            if quant_config.get("format") != expected_compression_format:
                errors.append(
                    "config.json compressed-tensors format does not match "
                    "the manifest"
                )
            config_groups = quant_config.get("config_groups") or {}
            group_formats = {
                group.get("format", quant_config.get("format"))
                for group in config_groups.values()
            }
            group_formats.discard(None)
            if (
                observed_compression_formats
                and group_formats != observed_compression_formats
            ):
                errors.append(
                    "config.json per-group formats do not match the manifest"
                )
            for group_name, group in config_groups.items():
                if group.get("format") == "int-quantized":
                    input_quant = group.get("input_activations") or {}
                    if input_quant != {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    }:
                        errors.append(
                            f"config group {group_name} does not declare "
                            "dynamic-token INT8 activations"
                        )
            config_bits = {
                group.get("weights", {}).get("num_bits")
                for group in config_groups.values()
                if group.get("weights", {}).get("type") == "int"
            }
            config_bits.discard(None)
            if observed_quant_bits and config_bits != observed_quant_bits:
                errors.append(
                    "config.json quantized bit widths do not match the manifest"
                )
            config_strategies = {
                group.get("weights", {}).get("strategy")
                for group in config_groups.values()
                if group.get("weights", {}).get("type") == "int"
            }
            config_strategies.discard(None)
            if (
                observed_strategies
                and config_strategies != observed_strategies
            ):
                errors.append(
                    "config.json weight strategies do not match the manifest"
                )

    config_path = path / "config.json"
    if not manifest_path.exists() and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        quant_config = config.get("quantization_config") or {}
        candidate_method = str(quant_config.get("quant_method", "")).lower()
        if candidate_method in {"gptq", "awq"}:
            native_method = candidate_method
            native_algorithm = str(
                quant_config.get("algorithm", candidate_method)
            ).lower()
            bits = int(quant_config.get("bits", 0))
            group_size = int(quant_config.get("group_size", 0))
            if bits not in ({4, 8} if candidate_method == "gptq" else {4}):
                errors.append(
                    f"Unsupported native {candidate_method.upper()} bit width: {bits}"
                )
            if group_size <= 0:
                errors.append(
                    f"Native {candidate_method.upper()} group_size must be positive"
                )
            pack_factor = 32 // bits if bits in {4, 8} else 0
            qweights = sorted(name for name in tensor_meta if name.endswith(".qweight"))
            native_quantized_tensors = len(qweights)
            if bits == 4:
                int4_tensors += native_quantized_tensors
            elif bits == 8:
                int8_tensors += native_quantized_tensors
            if not qweights:
                errors.append(
                    f"Native {candidate_method.upper()} config has no qweight tensors"
                )
            for qweight_name in qweights:
                prefix = qweight_name[: -len(".qweight")]
                if f"{prefix}.weight" in tensor_meta:
                    errors.append(
                        f"Native quantized module {prefix} also stores a float weight"
                    )
                qweight_shape, qweight_dtype = tensor_meta[qweight_name]
                if qweight_dtype != torch.int32 or len(qweight_shape) != 2:
                    errors.append(
                        f"{qweight_name} must be a 2D int32 tensor"
                    )
                    continue
                required = [f"{prefix}.qzeros", f"{prefix}.scales"]
                if candidate_method == "gptq":
                    required.append(f"{prefix}.g_idx")
                missing = [name for name in required if name not in tensor_meta]
                if missing:
                    errors.append(
                        f"Native {candidate_method.upper()} tensors missing for "
                        f"{prefix}: {missing}"
                    )
                    continue
                qzeros_shape, qzeros_dtype = tensor_meta[f"{prefix}.qzeros"]
                scales_shape, scales_dtype = tensor_meta[f"{prefix}.scales"]
                if qzeros_dtype != torch.int32 or len(qzeros_shape) != 2:
                    errors.append(f"{prefix}.qzeros must be a 2D int32 tensor")
                if scales_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                    errors.append(f"{prefix}.scales has unsupported dtype {scales_dtype}")
                if candidate_method == "awq" and scales_dtype != torch.float16:
                    errors.append(f"{prefix}.scales must be float16 for AutoAWQ GEMM")
                if len(scales_shape) != 2 or qzeros_shape[0] != scales_shape[0]:
                    errors.append(f"{prefix} scale/zero group dimensions do not match")
                if pack_factor:
                    if candidate_method == "gptq":
                        in_features = qweight_shape[0] * pack_factor
                        out_features = qweight_shape[1]
                    else:
                        in_features = qweight_shape[0]
                        out_features = qweight_shape[1] * pack_factor
                    if group_size > 0 and in_features % group_size:
                        errors.append(
                            f"{prefix} in_features={in_features} is not divisible "
                            f"by group_size={group_size}"
                        )
                    expected_groups = in_features // max(group_size, 1)
                    if scales_shape != (expected_groups, out_features):
                        errors.append(
                            f"{prefix}.scales has shape {scales_shape}, expected "
                            f"{(expected_groups, out_features)}"
                        )
                    if qzeros_shape != (expected_groups, out_features // pack_factor):
                        errors.append(
                            f"{prefix}.qzeros has shape {qzeros_shape}, expected "
                            f"{(expected_groups, out_features // pack_factor)}"
                        )
                    if candidate_method == "gptq":
                        g_idx_shape, g_idx_dtype = tensor_meta[f"{prefix}.g_idx"]
                        if g_idx_shape != (in_features,) or g_idx_dtype != torch.int32:
                            errors.append(
                                f"{prefix}.g_idx must be int32[{in_features}]"
                            )

    return {
        "valid": not errors,
        "errors": errors,
        "tensors": len(observed),
        "shards": len(checkpoint.shard_files()),
        "total_size": total_size,
        "int4_tensors": int4_tensors,
        "int8_tensors": int8_tensors,
        "has_manifest": manifest_path.exists(),
        "native_method": native_method,
        "native_algorithm": native_algorithm,
        "native_quantized_tensors": native_quantized_tensors,
    }
