from __future__ import annotations

import json
from pathlib import Path
import re

from tqdm import tqdm

from flagos_compressor.backends.base import BackendRunContext, QuantBackend
from flagos_compressor.core.compressed_tensors import build_compressed_tensors_config
from flagos_compressor.core.moe_layout import ExpertProjection, layout_by_name
from flagos_compressor.core.plan import ExecutionPlan, TensorAction
from flagos_compressor.core.report import ConversionReport
from flagos_compressor.formats.base import get_weight_format
from flagos_compressor.formats.compressed_tensors import (
    compressed_tensor_names,
    int_quantized_tensor_names,
)
from flagos_compressor.formats.compressed_tensors_moe import (
    fused_expert_bank_prefix,
)
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint

_LINEAR_FORMAT_BITS = {
    "compressed_tensors_int4_groupwise": 4,
    "compressed_tensors_int8_groupwise": 8,
    "compressed_tensors_int8_channelwise": 8,
    "compressed_tensors_w8a8_channelwise": 8,
}
_FUSED_MOE_FORMAT_BITS = {
    "compressed_tensors_int4_moe_fused": 4,
    "compressed_tensors_int8_moe_fused": 8,
    "compressed_tensors_w8a8_channelwise_moe_fused": 8,
}
_QUANTIZED_FORMAT_BITS = {
    **_LINEAR_FORMAT_BITS,
    **_FUSED_MOE_FORMAT_BITS,
}
_INT_QUANTIZED_FORMATS = {
    "compressed_tensors_w8a8_channelwise",
    "compressed_tensors_w8a8_channelwise_moe_fused",
}


def _fused_expert_projections(action) -> list[tuple[str, ExpertProjection]]:
    """Expand a fused-expert bank action into per-expert projection specs.

    Returns ``(module_prefix, ExpertProjection)`` for every logical per-expert
    weight the fused format emits, using the same :class:`MoeLayout` that drove
    quantization so the names/shapes match the tensors written to disk.
    """
    tensor = action.tensor
    prefix = fused_expert_bank_prefix(tensor.name)
    params = action.output_format.params
    proj_kind = params.get("proj_kind") or tensor.name.split(".")[-1]
    layout = layout_by_name(params["layout"])
    projections = layout.expand_bank(
        proj_kind,
        int(params["num_experts"]),
        tuple(tensor.effective_logical_shape),
    )
    return [(prefix, proj) for proj in projections]


def _fused_expert_logical_weights(action) -> list[str]:
    """Per-expert logical ``.weight`` names for CT config target/ignore sets."""
    return [
        f"{prefix}.{proj.expert_id}.{proj.proj_name}.weight"
        for prefix, proj in _fused_expert_projections(action)
    ]


def _build_action_map(plan: ExecutionPlan) -> dict[str, TensorAction]:
    return {action.tensor.name: action for action in plan.actions}


def _build_scale_names(plan: ExecutionPlan) -> set[str]:
    return {action.tensor.scale_name for action in plan.actions if action.tensor.scale_name}


_STRIP_CONFIG_KEYS = (
    "quantization_config",
    "compression_config",
    "quant_method",
)

_BF16_FORMATS = {"bf16"}


def _patch_bf16_config(output_path: Path) -> None:
    """Rewrite the output ``config.json`` so it describes a plain BF16 model.

    After dequantization the artifact no longer carries FP8/FP4 weights, so
    the source quantization metadata must be stripped and ``torch_dtype`` set
    to ``bfloat16``. ``expert_dtype``, when present on MoE configs, is
    rewritten rather than dropped so downstream loaders keep the field.
    """
    config_path = output_path / "config.json"
    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["torch_dtype"] = "bfloat16"
    for key in _STRIP_CONFIG_KEYS:
        config.pop(key, None)
    if "expert_dtype" in config:
        config["expert_dtype"] = "bfloat16"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _logical_weight_names(plan: ExecutionPlan) -> set[str]:
    names = {
        tensor.name
        for tensor in (
            [action.tensor for action in plan.actions] + plan.kept_tensors
        )
        if tensor.role == "weight" and tensor.name.endswith(".weight")
    }
    for action in plan.actions:
        if action.output_format.name in _FUSED_MOE_FORMAT_BITS:
            names.update(_fused_expert_logical_weights(action))
    return names


_QUANTIZED_OUTPUT_FORMATS = set(_QUANTIZED_FORMAT_BITS)


def _ignore_modules(plan: ExecutionPlan) -> set[str]:
    """Module names of unquantized Linear weights, for the CT ``ignore`` list.

    The vLLM compressed-tensors loader resolves a scheme for every Linear
    (and MoE) module and errors on any it cannot match. Every 2D float
    ``.weight`` that is not quantized is therefore listed in ``ignore`` so it
    loads as plain bf16. 1D weights (norms) and non-2D tensors are never
    treated as Linear by the loader and are left out.
    """
    quantized = {
        action.tensor.name
        for action in plan.actions
        if action.output_format.name in _QUANTIZED_OUTPUT_FORMATS
    }
    candidates = [action.tensor for action in plan.actions] + plan.kept_tensors
    ignore: set[str] = set()
    for tensor in candidates:
        if tensor.role != "weight" or not tensor.name.endswith(".weight"):
            continue
        if tensor.name in quantized:
            continue
        if len(tensor.effective_logical_shape) != 2:
            continue
        ignore.add(tensor.name[: -len(".weight")])
    return ignore


def _deepseek_v4_runtime_targets(
    config: dict,
    selected_logical_weights: set[str],
) -> set[str]:
    """Return runtime aliases for source projections fused by DeepSeek-V4.

    The HF checkpoint stores ``wq_a``/``wkv`` and shared-expert ``w1``/``w3``
    separately, while vLLM instantiates ``fused_wqa_wkv`` and ``gate_up_proj``.
    compressed-tensors matches its config against those runtime module names,
    so both source projections must select the corresponding fused alias.
    """
    text_config = config.get("text_config") or {}
    model_type = config.get("model_type") or text_config.get("model_type")
    if model_type != "deepseek_v4":
        return set()

    modules = {
        name[: -len(".weight")]
        for name in selected_logical_weights
        if name.endswith(".weight")
    }
    aliases: set[str] = set()
    for module in modules:
        if module.endswith(".wq_a"):
            prefix = module[: -len(".wq_a")]
            if f"{prefix}.wkv" in modules:
                aliases.add(f"{prefix}.fused_wqa_wkv")
        if module.endswith(".shared_experts.w1"):
            prefix = module[: -len(".w1")]
            if f"{prefix}.w3" in modules:
                aliases.add(f"{prefix}.gate_up_proj")
        if module.endswith(".shared_experts.w2"):
            aliases.add(module[: -len(".w2")] + ".down_proj")
    return aliases


def _routed_moe_runtime_targets(
    selected_logical_weights: set[str],
) -> set[str]:
    """Map per-bank expert names to vLLM's unfused projection lookups."""
    pattern = re.compile(
        r"^(?P<bank>.*\.experts)\.\d+\."
        r"(?:gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$"
    )
    targets: set[str] = set()
    for name in selected_logical_weights:
        match = pattern.match(name)
        if match is None:
            continue
        parts = match.group("bank").split(".")
        while len(parts) > 1 and parts[0] in {
            "model",
            "language_model",
            "visual",
            "vision_model",
        }:
            parts.pop(0)
        bank = re.escape(".".join(parts))
        targets.add(
            r"re:^.*"
            + bank
            + r"\.\d+\.(?:gate_proj|up_proj|down_proj|w1|w2|w3)$"
        )
    return targets


def _patch_compressed_tensors_config(
    output_path: Path,
    plan: ExecutionPlan,
) -> dict:
    config_path = output_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "compressed-tensors export requires config.json in the source checkpoint"
        )
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    selected_by_scheme: dict[
        tuple[int, int, str, int | None], set[str]
    ] = {}
    for action in plan.actions:
        output_name = action.output_format.name
        if output_name not in _QUANTIZED_FORMAT_BITS:
            continue
        scheme = (
            _QUANTIZED_FORMAT_BITS[output_name],
            int(action.output_format.params.get("activation_num_bits", 16)),
            action.output_format.params.get("strategy", "group"),
            (
                int(action.output_format.params["group_size"])
                if action.output_format.params.get("group_size") is not None
                else None
            ),
        )
        selected = selected_by_scheme.setdefault(scheme, set())
        if output_name in _FUSED_MOE_FORMAT_BITS:
            selected.update(_fused_expert_logical_weights(action))
        else:
            selected.add(action.tensor.name)
    if not selected_by_scheme:
        raise ValueError("Cannot export compressed-tensors without quantized weights")

    all_logical_weights = _logical_weight_names(plan)
    ignore_modules = _ignore_modules(plan)
    quantization_config: dict | None = None
    for scheme, selected in sorted(selected_by_scheme.items(), key=lambda item: str(item[0])):
        num_bits, activation_num_bits, strategy, group_size = scheme
        runtime_targets = _deepseek_v4_runtime_targets(config, selected)
        runtime_targets.update(_routed_moe_runtime_targets(selected))
        group_config = build_compressed_tensors_config(
            all_logical_weights,
            selected,
            num_bits=num_bits,
            activation_num_bits=activation_num_bits,
            strategy=strategy,
            group_size=group_size,
            ignore_modules=ignore_modules,
            additional_targets=runtime_targets,
        )
        if quantization_config is None:
            quantization_config = group_config
            continue
        if group_config["format"] != quantization_config["format"]:
            quantization_config["format"] = "mixed-precision"
        overlap = set(quantization_config["config_groups"]) & set(
            group_config["config_groups"]
        )
        if overlap:
            raise ValueError(
                "Compressed-tensors config group collision: "
                + ", ".join(sorted(overlap))
            )
        quantization_config["config_groups"].update(
            group_config["config_groups"]
        )
        quantization_config["ignore"] = sorted(
            set(quantization_config.get("ignore", ()))
            | set(group_config.get("ignore", ()))
        )
    assert quantization_config is not None

    config["torch_dtype"] = "bfloat16"
    for key in (*_STRIP_CONFIG_KEYS, "expert_dtype"):
        config.pop(key, None)
    config["quantization_config"] = quantization_config
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return quantization_config


def _write_quantization_manifest(
    output_path: Path,
    plan: ExecutionPlan,
    quantization_config: dict,
) -> None:
    tensors: dict[str, dict] = {}
    for action in plan.actions:
        tensor = action.tensor
        if action.output_format.name == "compressed_tensors_w8a8_channelwise":
            logical_shape = tensor.effective_logical_shape
            names = int_quantized_tensor_names(tensor.name)
            tensors[names.weight] = {
                "logical_name": tensor.name,
                "format": "compressed-tensors-int-quantized-int8",
                "input_format": action.input_format.name,
                "storage_dtype": "int8",
                "storage_shape": list(logical_shape),
                "logical_shape": list(logical_shape),
                "scale": names.scale,
                "scale_shape": [logical_shape[0], 1],
                "scale_dtype": action.output_format.params.get(
                    "scale_dtype", "float32"
                ),
                "num_bits": 8,
                "activation_num_bits": 8,
                "strategy": "channel",
                "rule": action.rule_name,
            }
        elif action.output_format.name in _LINEAR_FORMAT_BITS:
            num_bits = _LINEAR_FORMAT_BITS[action.output_format.name]
            logical_shape = tensor.effective_logical_shape
            strategy = action.output_format.params.get("strategy", "group")
            group_size = action.output_format.params.get("group_size")
            scale_columns = (
                1
                if strategy == "channel"
                else logical_shape[1] // int(group_size)
            )
            names = compressed_tensor_names(tensor.name)
            tensors[names.weight] = {
                "logical_name": tensor.name,
                "format": f"compressed-tensors-pack-quantized-int{num_bits}",
                "input_format": action.input_format.name,
                "storage_dtype": "int32",
                "storage_shape": [
                    logical_shape[0],
                    (logical_shape[1] * num_bits + 31) // 32,
                ],
                "logical_shape": list(logical_shape),
                "scale": names.scale,
                "scale_shape": [
                    logical_shape[0],
                    scale_columns,
                ],
                "shape": names.shape,
                "num_bits": num_bits,
                "activation_num_bits": 16,
                "scale_dtype": "bfloat16",
                "strategy": strategy,
                "rule": action.rule_name,
            }
            if group_size is not None:
                tensors[names.weight]["group_size"] = int(group_size)
        elif (
            action.output_format.name
            == "compressed_tensors_w8a8_channelwise_moe_fused"
        ):
            for prefix, proj in _fused_expert_projections(action):
                logical_name = (
                    f"{prefix}.{proj.expert_id}.{proj.proj_name}.weight"
                )
                names = int_quantized_tensor_names(logical_name)
                tensors[names.weight] = {
                    "logical_name": logical_name,
                    "format": "compressed-tensors-int-quantized-int8",
                    "input_format": action.input_format.name,
                    "storage_dtype": "int8",
                    "storage_shape": [proj.out_features, proj.in_features],
                    "logical_shape": [proj.out_features, proj.in_features],
                    "scale": names.scale,
                    "scale_shape": [proj.out_features, 1],
                    "scale_dtype": action.output_format.params.get(
                        "scale_dtype", "float32"
                    ),
                    "num_bits": 8,
                    "activation_num_bits": 8,
                    "strategy": "channel",
                    "rule": action.rule_name,
                    "source_fused_bank": tensor.name,
                }
        elif action.output_format.name in _FUSED_MOE_FORMAT_BITS:
            num_bits = _FUSED_MOE_FORMAT_BITS[action.output_format.name]
            group_size = int(action.output_format.params["group_size"])
            # Emit one spec per expert projection that actually lands on disk,
            # using the same schema as regular integer weights so the scanner and
            # convert path recognize the artifact by its real tensor names.
            for prefix, proj in _fused_expert_projections(action):
                logical_name = f"{prefix}.{proj.expert_id}.{proj.proj_name}.weight"
                names = compressed_tensor_names(logical_name)
                tensors[names.weight] = {
                    "logical_name": logical_name,
                    "format": f"compressed-tensors-pack-quantized-int{num_bits}",
                    "input_format": action.input_format.name,
                    "storage_dtype": "int32",
                    "storage_shape": [
                        proj.out_features,
                        (proj.in_features * num_bits + 31) // 32,
                    ],
                    "logical_shape": [proj.out_features, proj.in_features],
                    "scale": names.scale,
                    "scale_shape": [
                        proj.out_features,
                        proj.in_features // group_size,
                    ],
                    "shape": names.shape,
                    "num_bits": num_bits,
                    "activation_num_bits": 16,
                    "scale_dtype": "bfloat16",
                    "strategy": "group",
                    "group_size": group_size,
                    "rule": action.rule_name,
                    "source_fused_bank": tensor.name,
                }
        elif action.output_format.name in _BF16_FORMATS:
            tensors[tensor.name] = {
                "logical_name": tensor.name,
                "format": "bf16",
                "input_format": action.input_format.name,
                "storage_dtype": "bfloat16",
                "storage_shape": list(tensor.effective_logical_shape),
                "logical_shape": list(tensor.effective_logical_shape),
                "rule": action.rule_name,
            }
    for tensor in plan.kept_tensors:
        if tensor.role == "weight":
            tensors[tensor.name] = {
                "logical_name": tensor.name,
                "format": tensor.storage_format or tensor.dtype,
                "storage_dtype": tensor.dtype,
                "storage_shape": list(tensor.shape),
                "logical_shape": list(tensor.effective_logical_shape),
                "kept_from_source": True,
            }

    quantized_actions = [
        action
        for action in plan.actions
        if action.output_format.name in _QUANTIZED_FORMAT_BITS
    ]
    quantized_bits = {
        _QUANTIZED_FORMAT_BITS[action.output_format.name]
        for action in quantized_actions
    }
    if not quantized_bits:
        raise ValueError("Expected at least one quantized bit width")
    sorted_bits = sorted(quantized_bits)
    quantized_strategies = {
        action.output_format.params.get("strategy", "group")
        for action in quantized_actions
    }
    single_strategy = (
        next(iter(quantized_strategies))
        if len(quantized_strategies) == 1
        else None
    )
    compression_formats = {
        (
            "int-quantized"
            if action.output_format.name in _INT_QUANTIZED_FORMATS
            else "pack-quantized"
        )
        for action in quantized_actions
    }
    compression_format = (
        next(iter(compression_formats))
        if len(compression_formats) == 1
        else "mixed-precision"
    )
    is_int_quantized = compression_format == "int-quantized"
    scale_dtypes = {
        action.output_format.params.get("scale_dtype", "bfloat16")
        for action in quantized_actions
    }
    mixed_bits = len(sorted_bits) > 1
    mixed_storage = len(compression_formats) > 1
    schemes: dict[
        tuple[str, int, int, str, int | None, str], dict
    ] = {}
    for action in quantized_actions:
        output_name = action.output_format.name
        num_bits = _QUANTIZED_FORMAT_BITS[output_name]
        activation_num_bits = int(
            action.output_format.params.get("activation_num_bits", 16)
        )
        action_strategy = action.output_format.params.get("strategy", "group")
        group_size = action.output_format.params.get("group_size")
        scale_dtype = action.output_format.params.get(
            "scale_dtype", "bfloat16"
        )
        storage_format = (
            "int-quantized"
            if output_name in _INT_QUANTIZED_FORMATS
            else "pack-quantized"
        )
        key = (
            storage_format,
            num_bits,
            activation_num_bits,
            action_strategy,
            group_size,
            scale_dtype,
        )
        scheme = {
            "scheme": f"int{num_bits}-a{activation_num_bits}",
            "compression_format": storage_format,
            "weight_encoding": (
                "int8"
                if storage_format == "int-quantized"
                else ("uint4b8" if num_bits == 4 else "uint8b128")
            ),
            "num_bits": num_bits,
            "activation_num_bits": activation_num_bits,
            "strategy": action_strategy,
            "group_size": group_size,
            "scale_dtype": scale_dtype,
        }
        if storage_format == "pack-quantized":
            scheme["values_per_word"] = 32 // num_bits
        schemes[key] = scheme
    artifact = {
        "format": "compressed-tensors",
        "compression_format": compression_format,
        "weight_encoding": (
            "mixed"
            if mixed_storage
            else (
                "int8"
                if is_int_quantized
                else (
                    "mixed"
                    if mixed_bits
                    else ("uint4b8" if sorted_bits[0] == 4 else "uint8b128")
                )
            )
        ),
        "num_bits": sorted_bits if mixed_bits else sorted_bits[0],
        "scale_dtype": (
            next(iter(scale_dtypes))
            if len(scale_dtypes) == 1
            else "mixed"
        ),
        "scale_layout": (
            "mixed"
            if len(quantized_strategies) > 1
            else (
                "row_channel"
                if single_strategy == "channel"
                else "row_group"
            )
        ),
        "strategy": (
            single_strategy if single_strategy is not None else "mixed"
        ),
        "schemes": [schemes[key] for key in sorted(schemes, key=str)],
    }
    if mixed_storage:
        artifact["compression_formats"] = sorted(compression_formats)
        artifact["scale_dtypes"] = sorted(scale_dtypes)
    elif not is_int_quantized:
        artifact.update(
            {
                "pack_dtype": "int32",
                "pack_axis": "input",
                "values_per_word": (
                    {str(bits): 32 // bits for bits in sorted_bits}
                    if mixed_bits
                    else 32 // sorted_bits[0]
                ),
            }
        )
        if mixed_bits:
            artifact["weight_encodings"] = {
                str(bits): "uint4b8" if bits == 4 else "uint8b128"
                for bits in sorted_bits
            }
    manifest = {
        "schema": "flagos-compressor.provenance.v1",
        "producer": {"name": "FlagOS-Compressor"},
        "algorithm": plan.metadata.get("algorithm", {}),
        "artifact": artifact,
        "unselected_weights": plan.metadata.get("unselected_weights", {}),
        "runtime_config": {
            "quant_method": quantization_config["quant_method"],
            "config_groups": list(quantization_config["config_groups"]),
        },
        "tensors": tensors,
    }
    with (output_path / "quantization_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def execute_plan(
    input_path: str | Path,
    output_path: str | Path,
    plan: ExecutionPlan,
    backend: QuantBackend,
) -> ConversionReport:
    checkpoint = HfSafetensorsCheckpoint(input_path)
    output = Path(output_path)
    if Path(input_path).resolve() == output.resolve():
        raise ValueError("Input and output checkpoint directories must be different")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.copy_auxiliary_files(output)
    for manifest_name in (
        "quant_manifest.json",
        "quantization_manifest.json",
    ):
        manifest_path = output / manifest_name
        if manifest_path.exists():
            manifest_path.unlink()

    action_map = _build_action_map(plan)
    old_scale_names = _build_scale_names(plan)
    generated_tensor_locations: dict[str, str] = {}
    output_tensor_locations: dict[str, str] = {}
    report = ConversionReport(backend=backend.name)
    context = BackendRunContext(report=report)

    loaded_shards: dict[str, dict] = {}
    total_size = 0

    def get_tensor(name: str):
        shard = checkpoint.weight_map[name]
        if shard not in loaded_shards:
            loaded_shards[shard] = checkpoint.load_shard(shard)
        return loaded_shards[shard][name]

    for shard_name, _ in tqdm(list(checkpoint.iter_shards()), desc="Converting"):
        current = checkpoint.load_shard(shard_name)
        loaded_shards[shard_name] = current
        new_state: dict = {}

        for tensor_name, tensor in current.items():
            if tensor_name in old_scale_names:
                report.skipped_scales += 1
                continue

            action = action_map.get(tensor_name)
            if action is None:
                new_state[tensor_name] = tensor
                report.kept += 1
                continue

            scale = get_tensor(action.tensor.scale_name) if action.tensor.scale_name else None
            input_format = get_weight_format(action.input_format.name)
            canonical_weight = input_format.to_canonical(
                tensor,
                scale,
                backend,
                context,
                action.input_format.params,
            )
            output_format = get_weight_format(action.output_format.name)
            result = output_format.from_canonical(
                tensor_name,
                canonical_weight,
                backend,
                context,
                action.output_format.params,
            )
            new_state.update(result.tensors)
            for output_name in result.tensors:
                output_tensor_locations[output_name] = shard_name
            report.converted += 1
            for generated_name in result.generated_tensor_names:
                if generated_name in checkpoint.weight_map and generated_name not in old_scale_names:
                    raise ValueError(
                        f"Generated tensor {generated_name!r} collides with an existing tensor"
                    )
                generated_tensor_locations[generated_name] = shard_name
        total_size += sum(item.numel() * item.element_size() for item in new_state.values())

        checkpoint.save_shard(output, shard_name, new_state)
        while len(loaded_shards) > 2:
            oldest = next(iter(loaded_shards))
            del loaded_shards[oldest]

    replaced_input_names = {action.tensor.name for action in plan.actions}
    new_weight_map = {
        name: shard
        for name, shard in checkpoint.weight_map.items()
        if name not in old_scale_names and name not in replaced_input_names
    }
    new_weight_map.update(output_tensor_locations)
    checkpoint.write_index(output, new_weight_map, total_size=total_size)
    artifact_kind = plan.metadata.get("artifact_kind")
    if artifact_kind == "compressed_tensors":
        quantization_config = _patch_compressed_tensors_config(output, plan)
        _write_quantization_manifest(output, plan, quantization_config)
    else:
        _patch_bf16_config(output)
    report.output_tensors = len(new_weight_map)
    report.extras.update(
        {
            "artifact_kind": plan.metadata.get("artifact_kind"),
            "input_formats": dict(plan.input_format_counts),
            "output_formats": dict(plan.output_format_counts),
            "generated_tensors": len(generated_tensor_locations),
        }
    )
    report.finish()
    report_filename = (
        "quantization_report.json"
        if artifact_kind == "compressed_tensors"
        else "conversion_report.json"
    )
    report.extras["report_file"] = report_filename
    report.save(output / report_filename)
    return report
