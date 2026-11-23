"""Resolve and validate native calibration schemes before collecting activations."""

from collections import Counter, defaultdict
from dataclasses import asdict

from torch import nn

from flagos_compressor.core.compressed_tensors import validate_fusion_closure
from flagos_compressor.inspect.tensor_classifier import classify_weight


def validate_layer_schemes(layer_name, layer, settings):
    all_weights = {
        f"{layer_name}.{name}.weight"
        for name, module in layer.named_modules()
        if name and isinstance(module, nn.Linear)
    }
    groups = defaultdict(set)
    for name, policy in settings.items():
        groups[(policy.num_bits, policy.group_size)].add(f"{layer_name}.{name}.weight")
    # Validate each scheme independently: selecting both halves of a fused
    # projection is insufficient if they request different bits/group sizes.
    for members in groups.values():
        validate_fusion_closure(all_weights, members)


def build_calibration_plan(layers, policy):
    from flagos_compressor.calibration.runner import (
        selected_linears,
        _validate_native_shapes,
    )

    selected, retained = {}, {}
    for layer_name, layer in layers:
        linears = selected_linears(layer_name, layer, policy)
        settings = {}
        for name, linear in linears.items():
            weight_name = f"{layer_name}.{name}.weight"
            _, tags = classify_weight(weight_name)
            resolved = policy.settings_for_name(weight_name, tags)
            _validate_native_shapes({name: linear}, resolved)
            settings[name] = resolved
            selected[f"{layer_name}.{name}"] = {
                "bits": resolved.num_bits,
                "activation_bits": 16,
                "group_size": resolved.group_size,
                "shape": list(linear.weight.shape),
                "dtype_before_quantization": str(linear.weight.dtype),
            }
            if policy.method == "awq":
                selected[f"{layer_name}.{name}"]["sym"] = not (
                    resolved.awq.zero_point and resolved.num_bits == 4
                )
        validate_layer_schemes(layer_name, layer, settings)
        for name, module in layer.named_modules():
            weight = getattr(module, "weight", None)
            if (
                name
                and isinstance(weight, nn.Parameter)
                and weight.ndim >= 2
                and name not in linears
            ):
                retained[f"{layer_name}.{name}"] = {
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype),
                    "reason": (
                        "not_selected"
                        if isinstance(module, nn.Linear)
                        else "not_supported_by_native_linear_calibration"
                    ),
                }
    if not selected:
        raise ValueError(
            "Calibration rules did not select any supported Linear modules"
        )
    return {
        "schema": "flagos-compressor.native-calibration-plan.v1",
        "method": policy.method,
        "format": policy.format,
        "rules": [asdict(rule) for rule in policy.target_scheme_rules],
        "selected_modules": selected,
        "retained_modules": retained,
        "weight_bits_counts": dict(
            Counter(str(item["bits"]) for item in selected.values())
        ),
        "selected_module_count": len(selected),
    }
