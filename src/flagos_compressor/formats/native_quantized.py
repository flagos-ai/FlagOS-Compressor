"""Write GPTQ- and AWQ-compatible native safetensors checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
import torch
from torch import nn

from flagos_compressor.calibration.runner import NativeQuantizedLayer
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint
from flagos_compressor.packing.autogptq import AutoGPTQPacked


def _quantized_tensors(
    quantized: dict[str, NativeQuantizedLayer],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for prefix, result in quantized.items():
        packed = result.packed
        tensors[f"{prefix}.qweight"] = packed.qweight.contiguous().cpu()
        tensors[f"{prefix}.qzeros"] = packed.qzeros.contiguous().cpu()
        tensors[f"{prefix}.scales"] = packed.scales.contiguous().cpu()
        if isinstance(packed, AutoGPTQPacked):
            tensors[f"{prefix}.g_idx"] = packed.g_idx.contiguous().cpu()
    return tensors


def _state_for_export(
    model: nn.Module,
    quantized: dict[str, NativeQuantizedLayer],
) -> dict[str, torch.Tensor]:
    removed = {f"{prefix}.weight" for prefix in quantized}
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name not in removed and tensor.device.type != "meta"
    }
    state.update(_quantized_tensors(quantized))
    return state


def _runtime_linear_names(model: nn.Module) -> list[str]:
    """Include custom runtime Linear variants that calibration leaves float."""
    names: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        weight = getattr(module, "weight", None)
        is_custom_linear = (
            module.__class__.__name__.lower().endswith("linear")
            and isinstance(weight, nn.Parameter)
            and weight.dim() == 2
        )
        if isinstance(module, nn.Linear) or is_custom_linear:
            names.append(name)
    return sorted(names)


def _patch_config(
    output_path: Path,
    *,
    model: nn.Module,
    method: str,
    bits: int,
    group_size: int,
    desc_act: bool,
    damp_percent: float,
    true_sequential: bool,
    static_groups: bool,
    symmetric: bool,
    awq_zero_point: bool,
    awq_version: str,
    autoround_config: dict[str, Any] | None,
    quantized_modules: list[str],
    unquantized_modules: list[str],
    fallbacks: dict[str, dict[str, str]] | None = None,
    module_quantization: dict[str, dict[str, int]] | None = None,
    packing_format: str | None = None,
) -> dict[str, Any]:
    config_path = output_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("Native quantized export requires config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # AutoModelForCausalLM may intentionally select the text-only model from a
    # multimodal source checkpoint (for example Qwen3.5/3.6).  Keep the outer
    # config because runtimes use its nested text config, but declare the model
    # class whose state dict was actually exported.
    config["architectures"] = [model.__class__.__name__]
    config.pop("compression_config", None)
    if method in {"gptq", "autoround"} or packing_format == "gptq":
        quantization_config = {
            "bits": bits,
            "group_size": group_size,
            "damp_percent": damp_percent,
            "desc_act": desc_act,
            "sym": symmetric,
            "true_sequential": true_sequential,
            "static_groups": static_groups,
            "quant_method": "gptq",
            "checkpoint_format": "gptq",
            # Full names work for heterogeneous selection in vLLM and retain
            # the sequential grouping contract expected by GPTQModel.
            "modules_in_block_to_quantize": [quantized_modules],
            "dynamic": {
                f"-:^{re.escape(name)}$": {}
                for name in unquantized_modules
            },
        }
        if module_quantization:
            quantization_config.update(
                provider="flagos-compressor",
                flagos_module_quantization=module_quantization,
            )
            quantization_config["dynamic"].update(
                {
                    f"+:^{re.escape(name)}$": settings
                    for name, settings in module_quantization.items()
                    if settings != {"bits": bits, "group_size": group_size}
                }
            )
        if method == "awq":
            quantization_config.update(
                algorithm="awq",
                provider="flagos-compressor",
                sym=bits == 8 or not awq_zero_point,
                desc_act=False,
                true_sequential=False,
                static_groups=False,
                zero_point=awq_zero_point,
                packing_description="GPTQ tensor ABI; calibrated by AWQ, W8 symmetric and W4 follows awq.zero_point",
            )
        if method == "autoround":
            # Keep the loader-facing ABI canonical while recording the native
            # algorithm independently from its GPTQ-compatible packing.
            quantization_config.update(
                {
                    "algorithm": "autoround",
                    "provider": "flagos-compressor",
                    "desc_act": False,
                    "true_sequential": False,
                    "static_groups": False,
                }
            )
            if autoround_config is not None:
                quantization_config.update(autoround_config)
            # These fields are part of the official AutoGPTQ export contract
            # and must not be overridden by a training config.
            quantization_config.update(
                {
                    "bits": bits,
                    "group_size": group_size,
                    "sym": True,
                    "data_type": "int",
                    "provider": "flagos-compressor",
                    "algorithm": "autoround",
                    "quant_method": "gptq",
                    "checkpoint_format": "gptq",
                    "desc_act": False,
                    "true_sequential": False,
                    "static_groups": False,
                }
            )
    else:
        quantization_config = {
            "bits": bits,
            "group_size": group_size,
            "zero_point": awq_zero_point,
            "version": awq_version.lower(),
            "quant_method": "awq",
            "modules_to_not_convert": unquantized_modules,
        }
    if fallbacks:
        quantization_config.update(
            algorithm=method + "+rtn",
            calibration_method=method,
            fallback_quantization={
                "method": "rtn",
                "module_count": len(fallbacks),
                "modules": fallbacks,
            },
        )
    config["quantization_config"] = quantization_config
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return quantization_config


def save_native_quantized_model(
    source_path: str | Path,
    output_path: str | Path,
    model: nn.Module,
    quantized: dict[str, NativeQuantizedLayer],
    *,
    method: str,
    bits: int,
    group_size: int,
    desc_act: bool = False,
    damp_percent: float = 0.01,
    true_sequential: bool = True,
    static_groups: bool = False,
    symmetric: bool = True,
    awq_zero_point: bool = True,
    awq_version: str = "gemm",
    autoround_config: dict[str, Any] | None = None,
    max_shard_size: int | str = "5GB",
) -> None:
    if not quantized:
        raise RuntimeError(f"{method.upper()} selectors did not match any Linear modules")
    if method not in {"gptq", "awq", "autoround"}:
        raise ValueError(f"Unsupported native quantization method: {method}")
    fallbacks = {
        name: {"requested_method": method, "reason": result.fallback_reason}
        for name, result in quantized.items()
        if result.algorithm == "rtn"
        and result.fallback_from == method
        and result.fallback_reason in {"no_calibration_input", "no_optimization_input"}
    }
    mismatched = sorted(
        name
        for name, result in quantized.items()
        if result.algorithm != method and name not in fallbacks
    )
    if mismatched:
        raise ValueError(
            f"Native {method.upper()} export received tensors from another method: "
            f"{mismatched[:3]}"
        )
    expected_packing = (
        "gptq"
        if method in {"gptq", "autoround"}
        or all(
            isinstance(result.packed, AutoGPTQPacked) for result in quantized.values()
        )
        else "awq"
    )
    packing_mismatches = sorted(
        name
        for name, result in quantized.items()
        if result.packing != expected_packing
    )
    if packing_mismatches:
        raise ValueError(
            f"Native {method.upper()} export requires {expected_packing.upper()} "
            f"packing: {packing_mismatches[:3]}"
        )
    schemes = {
        name: {
            "bits": result.num_bits if result.num_bits is not None else bits,
            "group_size": (
                result.group_size if result.group_size is not None else group_size
            ),
        }
        for name, result in quantized.items()
    }
    if method == "awq" and expected_packing == "gptq":
        for name, settings in schemes.items():
            if quantized[name].symmetric is None:
                raise ValueError(
                    f"AWQ GPTQ packing requires explicit symmetry metadata for {name}"
                )
            settings["sym"] = quantized[name].symmetric
    per_module = any(
        value != {"bits": bits, "group_size": group_size} for value in schemes.values()
    )
    if per_module and expected_packing != "gptq":
        raise ValueError("Per-module native export requires GPTQ-compatible packing")
    if expected_packing == "gptq":
        from flagos_compressor.core.compressed_tensors import validate_fusion_closure
        from collections import defaultdict

        by_scheme = defaultdict(set)
        for name, settings in schemes.items():
            packed = quantized[name].packed
            module_bits, module_group = settings["bits"], settings["group_size"]
            if module_bits not in (4, 8) or module_group <= 0:
                raise ValueError(f"Invalid GPTQ bit width/group size for {name}")
            in_features = packed.g_idx.numel()
            out_features = packed.qweight.shape[1]
            factor = 32 // module_bits
            if (
                in_features % module_group
                or in_features % factor
                or out_features % factor
                or tuple(packed.qweight.shape) != (in_features // factor, out_features)
                or tuple(packed.scales.shape)
                != (in_features // module_group, out_features)
                or tuple(packed.qzeros.shape)
                != (in_features // module_group, out_features // factor)
            ):
                raise ValueError(
                    f"Packed GPTQ tensors disagree with module scheme for {name}"
                )
            by_scheme[(module_bits, module_group, settings.get("sym", symmetric))].add(
                name + ".weight"
            )
        if per_module:
            all_weights = [name + ".weight" for name in _runtime_linear_names(model)]
            for members in by_scheme.values():
                validate_fusion_closure(all_weights, members)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = HfSafetensorsCheckpoint(source_path)
    checkpoint.copy_auxiliary_files(output)
    state = _state_for_export(model, quantized)
    filename_pattern = (
        f"gptq_model-{bits}bit-{group_size}g{{suffix}}.safetensors"
        if expected_packing == "gptq" and not per_module
        else "model{suffix}.safetensors"
    )
    split = split_torch_state_dict_into_shards(
        state,
        filename_pattern=filename_pattern,
        max_shard_size=max_shard_size,
    )
    for filename, names in split.filename_to_tensors.items():
        shard: dict[str, torch.Tensor] = {}
        seen_storage: set[int] = set()
        for name in names:
            tensor = state[name]
            storage = tensor.untyped_storage()
            storage_id = storage.data_ptr()
            tensor_nbytes = tensor.numel() * tensor.element_size()
            # Safetensors rejects shared backing stores and partial views.
            # Clone only those tensors inside the current shard, keeping peak
            # export memory bounded by max_shard_size.
            if (
                storage_id in seen_storage
                or tensor.storage_offset() != 0
                or storage.nbytes() != tensor_nbytes
            ):
                tensor = tensor.clone(memory_format=torch.contiguous_format)
            elif not tensor.is_contiguous():
                tensor = tensor.contiguous()
            shard[name] = tensor
            seen_storage.add(storage_id)
        save_file(shard, str(output / filename), metadata={"format": "pt"})
    checkpoint.write_index(
        output,
        dict(split.tensor_to_filename),
        total_size=int(split.metadata["total_size"]),
    )
    if expected_packing == "gptq" and split.is_sharded:
        # AutoGPTQ searches for an index alongside its canonical model basename,
        # while Transformers searches model.safetensors.index.json.
        native_index = output / (
            filename_pattern.format(suffix="") + ".index.json"
        )
        native_index.write_text(
            json.dumps(
                {
                    "metadata": split.metadata,
                    "weight_map": split.tensor_to_filename,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    quantized_modules = sorted(quantized)
    all_linears = _runtime_linear_names(model)
    unquantized_modules = [name for name in all_linears if name not in quantized]
    from flagos_compressor.calibration.moe import LinearExperts2D

    for name, module in model.named_modules():
        if not name or not isinstance(module, LinearExperts2D):
            continue
        if not any(prefix.startswith(name + ".") for prefix in quantized):
            # vLLM sees the original fused expert unit, not its temporary
            # per-expert Linear modules, when deciding whether to quantize it.
            unquantized_modules.append(name)
    unquantized_modules = sorted(set(unquantized_modules))
    quantization_config = _patch_config(
        output,
        model=model,
        method=method,
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp_percent,
        true_sequential=true_sequential,
        static_groups=static_groups,
        symmetric=symmetric,
        awq_zero_point=awq_zero_point,
        awq_version=awq_version,
        autoround_config=autoround_config,
        quantized_modules=quantized_modules,
        unquantized_modules=unquantized_modules,
        fallbacks=fallbacks,
        module_quantization=schemes if per_module else None,
        packing_format=expected_packing,
    )
    external_config_name = (
        "quantize_config.json"
        if expected_packing == "gptq"
        else "quant_config.json"
    )
    (output / external_config_name).write_text(
        json.dumps(quantization_config, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["save_native_quantized_model"]
