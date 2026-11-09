"""Sequential Transformers runner for GPTQ, AWQ, and AutoRound calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import logging
from typing import Any

import torch
from torch import nn

from flagos_compressor.calibration.awq_inputs import AWQCall, AWQReplay
from flagos_compressor.calibration.awq_safety import (
    AWQConsumerDiscovery, AWQScalingCheck, AWQScaleTransaction,
)
from flagos_compressor.calibration.mappings import (
    gptq_sequential_groups,
    infer_awq_mappings,
)
from flagos_compressor.calibration.modeling import (
    forward_layer_samples,
    move_to_device,
    prepare_forward_kwargs,
    sanitize_kwargs,
)
from flagos_compressor.core.policy import QuantizationPolicy
from flagos_compressor.inspect.tensor_classifier import classify_weight
from flagos_compressor.packing.autoawq import AutoAWQPacked, pack_autoawq_gemm
from flagos_compressor.packing.autogptq import AutoGPTQPacked, pack_autogptq
from flagos_compressor.quantizers.awq import (
    apply_awq_clip,
    apply_awq_scale,
    pseudo_quantize_awq,
    search_awq_clip,
    search_awq_scale,
    should_skip_awq_clip,
)
from flagos_compressor.quantizers.autoround import AutoRoundLinear, SignSGD
from flagos_compressor.quantizers.gptq import GPTQQuantizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeQuantizedLayer:
    """A quantized layer with independent algorithm and checkpoint packing."""

    algorithm: str
    packed: AutoGPTQPacked | AutoAWQPacked
    packing: str | None = None

    def __post_init__(self) -> None:
        if self.packing is None:
            object.__setattr__(self, "packing", self.algorithm)

    @property
    def method(self) -> str:
        """Backward-compatible alias for callers written before packing split."""
        return self.algorithm


def _validate_fused_moe_selection(
    layer_name: str,
    layer: nn.Module,
    selected: set[str],
) -> None:
    """A native runtime quantizes a routed expert unit as one closed set."""
    from flagos_compressor.calibration.moe import LinearExperts2D

    for root, module in layer.named_modules():
        if not isinstance(module, LinearExperts2D):
            continue
        expert_linears = {
            f"{root}.{name}" if root else name
            for name, child in module.named_modules()
            if name and isinstance(child, nn.Linear)
        }
        chosen = expert_linears & selected
        if chosen and chosen != expert_linears:
            missing = sorted(expert_linears - chosen)
            raise ValueError(
                f"Routed experts {layer_name}.{root} are partially selected; "
                "Native calibrated runtimes require gate/up/down for every expert. "
                f"First missing modules: {missing[:3]}"
            )


def _fused_expert_linear_names(layer: nn.Module) -> set[str]:
    """Return layer-relative Linear names owned by fused routed experts."""
    from flagos_compressor.calibration.moe import LinearExperts2D

    names: set[str] = set()
    for root, module in layer.named_modules():
        if not isinstance(module, LinearExperts2D):
            continue
        names.update(
            f"{root}.{name}" if root else name
            for name, child in module.named_modules()
            if name and isinstance(child, nn.Linear)
        )
    return names


def _require_routed_expert_coverage(
    layer: nn.Module,
    selected: set[str],
    observed: set[str],
    method: str,
    *,
    fused_names: set[str] | None = None,
) -> None:
    required = (fused_names or _fused_expert_linear_names(layer)) & selected
    missing = sorted(required - observed)
    if missing:
        raise RuntimeError(
            f"{method} calibration did not route any tokens to "
            f"{len(missing)} selected expert projections. Increase calibration "
            "samples or use more representative data. First missing modules: "
            f"{missing[:3]}"
        )


def _validate_native_shapes(
    linears: dict[str, nn.Linear],
    policy: QuantizationPolicy,
) -> None:
    group_size = int(policy.group_size or -1)
    pack_factor = 32 // policy.num_bits
    for name, linear in linears.items():
        if group_size <= 0 or linear.in_features % group_size:
            raise ValueError(
                f"{name} in_features={linear.in_features} must be divisible by "
                f"group_size={group_size}"
            )
        if linear.out_features % pack_factor:
            raise ValueError(
                f"{name} out_features={linear.out_features} must be divisible by "
                f"native pack factor {pack_factor}"
            )
        if policy.method in {"gptq", "autoround"} and linear.in_features % pack_factor:
            raise ValueError(
                f"{name} in_features={linear.in_features} must be divisible by "
                f"native pack factor {pack_factor}"
            )


def selected_linears(
    layer_name: str,
    layer: nn.Module,
    policy: QuantizationPolicy,
) -> dict[str, nn.Linear]:
    selected: dict[str, nn.Linear] = {}
    for relative_name, module in layer.named_modules():
        if not relative_name or not isinstance(module, nn.Linear):
            continue
        full_weight_name = f"{layer_name}.{relative_name}.weight"
        _, tags = classify_weight(full_weight_name)
        if policy.selects_name(full_weight_name, tags):
            selected[relative_name] = module
    return selected


@torch.no_grad()
def _run_samples(
    layer: nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    device: torch.device,
    *,
    before_forward=None,
    after_forward=None,
) -> None:
    layer.to(device)
    for index, (args, kwargs) in enumerate(samples):
        moved_args = move_to_device(args, device)
        moved_kwargs = prepare_forward_kwargs(layer, kwargs, device)
        if before_forward is not None:
            before_forward(index)
        output = layer(*moved_args, **moved_kwargs)
        if after_forward is not None:
            after_forward(index, output, moved_kwargs)


def _capture_awq_input(
    _module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    features: dict[str, list[torch.Tensor]],
) -> None:
    value = args[0] if args else kwargs.get("hidden_states")
    if not isinstance(value, torch.Tensor):
        value = next(
            (item for item in kwargs.values() if isinstance(item, torch.Tensor)),
            None,
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Could not capture canonical AWQ input for {name}")
    features[name].append(value.detach().cpu())


@torch.no_grad()
def quantize_layer_gptq(
    layer_name: str,
    layer: nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    policy: QuantizationPolicy,
    *,
    device: torch.device,
) -> dict[str, NativeQuantizedLayer]:
    linears = selected_linears(layer_name, layer, policy)
    if not linears:
        return {}
    _validate_native_shapes(linears, policy)
    _validate_fused_moe_selection(layer_name, layer, set(linears))
    groups = (
        gptq_sequential_groups(list(linears))
        if policy.gptq.true_sequential
        else [list(linears)]
    )
    results: dict[str, NativeQuantizedLayer] = {}
    layer.to(device)
    for group in groups:
        quantizers = {
            name: GPTQQuantizer(
                linears[name].weight,
                bits=policy.num_bits,
                symmetric=policy.gptq.symmetric,
            )
            for name in group
        }
        handles = [
            linears[name].register_forward_pre_hook(
                lambda _module, args, q=quantizers[name]: q.add_batch(args[0])
            )
            for name in group
        ]
        try:
            _run_samples(layer, samples, device)
        finally:
            for handle in handles:
                handle.remove()
        _require_routed_expert_coverage(
            layer,
            set(group),
            {
                name
                for name, quantizer in quantizers.items()
                if quantizer.num_samples > 0
            },
            "GPTQ",
        )
        for name in group:
            linear = linears[name]
            result = quantizers[name].quantize(
                block_size=policy.gptq.block_size,
                damp_percent=policy.gptq.damp_percent,
                group_size=int(policy.group_size or -1),
                desc_act=policy.gptq.desc_act,
                static_groups=policy.gptq.static_groups,
            )
            linear.weight.data.copy_(result.weight)
            packed = pack_autogptq(
                result.weight,
                result.scales,
                result.zeros,
                result.g_idx,
                bits=policy.num_bits,
                scale_dtype=linear.weight.dtype,
            )
            results[f"{layer_name}.{name}"] = NativeQuantizedLayer("gptq", packed)
    return results


def _hidden_output(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("Decoder layer did not return a hidden-state tensor")
    return hidden


def _sample_indices(
    count: int,
    requested: int,
    generator: torch.Generator,
) -> list[int]:
    if requested <= count:
        return torch.randperm(count, generator=generator)[:requested].tolist()
    return torch.randint(count, (requested,), generator=generator).tolist()


def _autoround_loss(
    layer: nn.Module,
    input_samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    fp_outputs: list[tuple[tuple[Any, ...], dict[str, Any]]],
    indices: list[int],
    *,
    device: torch.device,
    backward: bool,
) -> float:
    loss_value = 0.0
    for sample_index in indices:
        args, kwargs = input_samples[sample_index]
        moved_args = move_to_device(args, device)
        moved_kwargs = move_to_device(
            sanitize_kwargs(layer, kwargs),
            device,
        )
        predicted = _hidden_output(layer(*moved_args, **moved_kwargs))
        target = fp_outputs[sample_index][0][0].to(
            device=device,
            dtype=predicted.dtype,
        )
        sample_loss = torch.mean(
            (predicted.float() - target.float()).square()
        )
        if backward:
            (sample_loss / len(indices)).backward()
        loss_value += float(sample_loss.detach()) / len(indices)
    return loss_value


def _snapshot_autoround_parameters(
    wrappers: dict[str, AutoRoundLinear],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return {
        name: (
            wrapper.value.detach().cpu().clone(),
            wrapper.min_scale.detach().cpu().clone(),
            wrapper.max_scale.detach().cpu().clone(),
        )
        for name, wrapper in wrappers.items()
    }


def _restore_submodules(
    layer: nn.Module,
    linears: dict[str, nn.Linear],
) -> None:
    for name, linear in linears.items():
        layer.set_submodule(name, linear)


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cpu":
        return
    try:
        runtime = torch.get_device_module(device)
    except (AttributeError, RuntimeError):
        runtime = getattr(torch, device.type, None)
    empty_cache = getattr(runtime, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def quantize_layer_autoround(
    layer_name: str,
    layer: nn.Module,
    fp_samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    quantized_samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    policy: QuantizationPolicy,
    *,
    device: torch.device,
    layer_index: int = 0,
) -> tuple[
    dict[str, NativeQuantizedLayer],
    list[tuple[tuple[Any, ...], dict[str, Any]]],
    list[tuple[tuple[Any, ...], dict[str, Any]]],
]:
    """Optimize one block and return its FP and quantized output streams."""
    if not fp_samples:
        raise ValueError("AutoRound requires at least one calibration sample")
    linears = selected_linears(layer_name, layer, policy)
    fp_outputs = forward_layer_samples(layer, fp_samples, device=device)
    input_samples = (
        quantized_samples
        if policy.autoround.enable_quantized_input
        else fp_samples
    )
    if not linears:
        quantized_outputs = forward_layer_samples(
            layer,
            input_samples,
            device=device,
        )
        return {}, fp_outputs, quantized_outputs

    _validate_native_shapes(linears, policy)
    _validate_fused_moe_selection(layer_name, layer, set(linears))
    fused_expert_linears = _fused_expert_linear_names(layer)
    layer.to(device)
    original_grad_state = [
        (parameter, parameter.requires_grad) for parameter in layer.parameters()
    ]
    for parameter, _ in original_grad_state:
        parameter.requires_grad_(False)

    wrappers = {
        name: AutoRoundLinear(
            linear,
            bits=policy.num_bits,
            group_size=int(policy.group_size or -1),
            enable_minmax_tuning=policy.autoround.enable_minmax_tuning,
        )
        for name, linear in linears.items()
    }
    for name, wrapper in wrappers.items():
        layer.set_submodule(name, wrapper)

    rounding_parameters = [wrapper.value for wrapper in wrappers.values()]
    range_parameters = [
        parameter
        for wrapper in wrappers.values()
        for parameter in (wrapper.min_scale, wrapper.max_scale)
        if parameter.requires_grad
    ]
    learning_rate = policy.autoround.lr or (1.0 / policy.autoround.iters)
    minmax_learning_rate = policy.autoround.minmax_lr or learning_rate
    parameter_groups: list[dict[str, Any]] = [
        {"params": rounding_parameters, "lr": learning_rate}
    ]
    if range_parameters:
        parameter_groups.append(
            {"params": range_parameters, "lr": minmax_learning_rate}
        )
    optimizer = SignSGD(
        parameter_groups,
        lr=learning_rate,
        momentum=policy.autoround.momentum,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(policy.calibration.seed + layer_index)
    best_loss = float("inf")
    best_parameters: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    effective_batch_size = min(
        len(input_samples),
        policy.autoround.batch_size
        * policy.autoround.gradient_accumulate_steps,
    )
    last_indices: list[int] = []

    try:
        with torch.enable_grad():
            for iteration in range(policy.autoround.iters):
                optimizer.zero_grad(set_to_none=True)
                indices = _sample_indices(
                    len(input_samples),
                    effective_batch_size,
                    generator,
                )
                last_indices = indices
                loss_value = _autoround_loss(
                    layer,
                    input_samples,
                    fp_outputs,
                    indices,
                    device=device,
                    backward=True,
                )

                if loss_value < best_loss:
                    best_loss = loss_value
                    best_parameters = _snapshot_autoround_parameters(wrappers)
                remaining = 1.0 - (iteration / policy.autoround.iters)
                optimizer.param_groups[0]["lr"] = learning_rate * remaining
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = (
                        minmax_learning_rate * remaining
                    )
                optimizer.step()
                with torch.no_grad():
                    for wrapper in wrappers.values():
                        wrapper.min_scale.clamp_(0, 1)
                        wrapper.max_scale.clamp_(0, 1)

        # The loss measured inside the loop belongs to the parameters before
        # ``optimizer.step``. Evaluate the final update explicitly so it can be
        # selected as the best state instead of being silently discarded.
        with torch.no_grad():
            final_loss = _autoround_loss(
                layer,
                input_samples,
                fp_outputs,
                last_indices,
                device=device,
                backward=False,
            )
        if final_loss < best_loss:
            best_loss = final_loss
            best_parameters = _snapshot_autoround_parameters(wrappers)

        _require_routed_expert_coverage(
            layer,
            set(linears),
            {
                name
                for name, wrapper in wrappers.items()
                if wrapper.num_forwards > 0
            },
            "AutoRound",
            fused_names=fused_expert_linears,
        )

        results: dict[str, NativeQuantizedLayer] = {}
        with torch.no_grad():
            for name, wrapper in wrappers.items():
                best_value, best_minimum, best_maximum = best_parameters[name]
                wrapper.value.copy_(best_value.to(device))
                wrapper.min_scale.copy_(best_minimum.to(device))
                wrapper.max_scale.copy_(best_maximum.to(device))
                quantized = wrapper.quantized()
                linear = linears[name]
                linear.weight.copy_(quantized.weight)
                g_idx = torch.arange(
                    linear.in_features,
                    device=device,
                    dtype=torch.int32,
                ) // int(policy.group_size or -1)
                packed = pack_autogptq(
                    quantized.weight.cpu(),
                    quantized.scales.cpu(),
                    quantized.zeros.cpu(),
                    g_idx.cpu(),
                    bits=policy.num_bits,
                    scale_dtype=linear.weight.dtype,
                )
                results[f"{layer_name}.{name}"] = NativeQuantizedLayer(
                    "autoround",
                    packed,
                    packing="gptq",
                )
    finally:
        _restore_submodules(layer, linears)
        for parameter, requires_grad in original_grad_state:
            parameter.requires_grad_(requires_grad)

    quantized_outputs = forward_layer_samples(
        layer,
        input_samples,
        device=device,
    )
    logger.info("%s AutoRound best calibration loss: %.6g", layer_name, best_loss)
    return results, fp_outputs, quantized_outputs


@torch.no_grad()
def quantize_layer_awq(
    layer_name: str,
    layer: nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    policy: QuantizationPolicy,
    *,
    device: torch.device,
    scaling_report: dict[str, Any] | None = None,
) -> dict[str, NativeQuantizedLayer]:
    linears = selected_linears(layer_name, layer, policy)
    if not linears:
        if scaling_report is not None:
            scaling_report.update(state="skipped", reason="no_selected_linears")
        return {}
    _validate_native_shapes(linears, policy)
    _validate_fused_moe_selection(layer_name, layer, set(linears))
    layer.to(device)
    modules = dict(layer.named_modules())
    mappings = infer_awq_mappings(layer, set(linears))
    discovery = AWQConsumerDiscovery(layer, mappings)
    scaling_check = AWQScalingCheck(layer, samples)
    if scaling_report is not None:
        scaling_report.update(scaling_check.report)
        scaling_check.report = scaling_report
    feature_names = set(linears)
    feature_names.update(
        mapping.input_name for mapping in mappings if mapping.input_name in modules
    )
    features: dict[str, list[torch.Tensor]] = {
        name: [] for name in sorted(feature_names)
    }
    calls = {mapping.inspect_name: [] for mapping in mappings}
    handles = [
        modules[name].register_forward_pre_hook(
            lambda module, args, kwargs, name=name: _capture_awq_input(
                module,
                args,
                kwargs,
                name=name,
                features=features,
            ),
            with_kwargs=True,
        )
        for name in sorted(feature_names)
    ]
    handles.extend(
        modules[name].register_forward_pre_hook(
            lambda _m, args, kwargs, name=name: calls[name].append(AWQCall.capture(args, kwargs)),
            with_kwargs=True,
        ) for name in calls
    )
    try:
        with discovery.capture(), scaling_check.capture_branches(layer):
            _run_samples(layer, samples, device, before_forward=scaling_check.begin_sample,
                         after_forward=scaling_check.observe)
    finally:
        for handle in handles:
            handle.remove()
    mappings, scaling_check.report["mappings"] = discovery.resolve()
    inputs = {name: torch.cat(values, dim=0) for name, values in features.items() if values}
    _require_routed_expert_coverage(
        layer,
        set(linears),
        set(inputs),
        "AWQ",
    )

    try:
        with AWQScaleTransaction(layer) as transaction:
            for mapping in mappings:
                if mapping.input_name not in inputs:
                    continue
                previous = modules[mapping.previous_name]
                search_linears = [linears[name] for name in mapping.quantized_names]
                balance = [modules[name] for name in mapping.linear_names]
                inspect_module = AWQReplay(modules[mapping.inspect_name], calls[mapping.inspect_name])
                transaction.remember(previous, balance)
                scales = search_awq_scale(
                    inspect_module,
                    search_linears,
                    inputs[mapping.input_name],
                    group_size=int(policy.group_size or 128),
                    zero_point=policy.awq.zero_point,
                    duo_scaling=policy.awq.duo_scaling,
                    n_grid=policy.awq.n_grid,
                    max_chunk_memory=policy.awq.max_chunk_memory,
                )
                apply_awq_scale(previous, balance, scales)
                scaling_check.branch_exclusions.add(mapping.previous_name)
                scaling_check.report["applied_mappings"] += 1
                for name in mapping.linear_names:
                    if name in inputs:
                        inputs[name].div_(scales.view(1, -1))
            scaling_check.verify(layer, samples, device)
    except Exception as exc:
        scaling_check.report.update(state="failed", error=str(exc), weights_restored=True)
        raise

    if policy.awq.apply_clip:
        for name, linear in linears.items():
            if name not in inputs or should_skip_awq_clip(name):
                continue
            maximum = search_awq_clip(
                linear.weight,
                inputs[name],
                group_size=int(policy.group_size or 128),
                zero_point=policy.awq.zero_point,
                n_grid=policy.awq.n_grid,
            )
            apply_awq_clip(linear.weight, maximum)

    results: dict[str, NativeQuantizedLayer] = {}
    for name, linear in linears.items():
        quantized = pseudo_quantize_awq(
            linear.weight,
            bits=4,
            group_size=int(policy.group_size or 128),
            zero_point=policy.awq.zero_point,
        )
        if quantized.zeros is None:
            raise ValueError("Native AutoAWQ GEMM export requires zero_point=true")
        linear.weight.data.copy_(quantized.weight)
        packed = pack_autoawq_gemm(
            quantized.weight,
            quantized.scales,
            quantized.zeros,
            group_size=int(policy.group_size or 128),
            # AutoAWQ's native GEMM ABI serializes scales as FP16 even when
            # the source model itself uses BF16.
            scale_dtype=torch.float16,
        )
        results[f"{layer_name}.{name}"] = NativeQuantizedLayer("awq", packed)
    return results


def _write_awq_report(
    path: str | Path | None, layers: list[dict[str, Any]], state: str
) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"state": state, "layers": layers}, indent=2) + "\n")
    temporary.replace(path)


@torch.no_grad()
def quantize_model_sequential(
    layers: list[tuple[str, nn.Module]],
    first_layer_samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    policy: QuantizationPolicy,
    *,
    device: torch.device,
    awq_report_path: str | Path | None = None,
) -> dict[str, NativeQuantizedLayer]:
    awq_layers: list[dict[str, Any]] = []
    samples = first_layer_samples
    fp_samples = first_layer_samples
    all_results: dict[str, NativeQuantizedLayer] = {}
    for layer_index, (layer_name, layer) in enumerate(layers, start=1):
        logger.info(
            "[%d/%d] %s calibration: %s",
            layer_index,
            len(layers),
            policy.method.upper(),
            layer_name,
        )
        if policy.method == "gptq":
            results = quantize_layer_gptq(
                layer_name, layer, samples, policy, device=device
            )
        elif policy.method == "awq":
            record = {"name": layer_name, "state": "running", "awq_scaling": {}}
            awq_layers.append(record)
            try:
                results = quantize_layer_awq(
                    layer_name, layer, samples, policy, device=device,
                    scaling_report=record["awq_scaling"],
                )
            except Exception as exc:
                record.update(state="failed", error=str(exc))
                if record["awq_scaling"].get("state") in {None, "pending"}:
                    record["awq_scaling"].update(state="failed", error=str(exc))
                _write_awq_report(awq_report_path, awq_layers, "failed")
                raise
            record["state"] = "complete"
            _write_awq_report(awq_report_path, awq_layers, "running")
        elif policy.method == "autoround":
            results, fp_samples, samples = quantize_layer_autoround(
                layer_name,
                layer,
                fp_samples,
                samples,
                policy,
                device=device,
                layer_index=layer_index,
            )
        else:
            raise ValueError(f"Sequential runner does not support {policy.method}")
        all_results.update(results)
        if policy.method != "autoround":
            samples = forward_layer_samples(layer, samples, device=device)
        layer.cpu()
        _empty_device_cache(device)
    if policy.method == "awq":
        _write_awq_report(awq_report_path, awq_layers, "complete")
    return all_results


__all__ = [
    "NativeQuantizedLayer",
    "quantize_layer_awq",
    "quantize_layer_autoround",
    "quantize_layer_gptq",
    "quantize_model_sequential",
    "selected_linears",
]
