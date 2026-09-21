from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from flagos_compressor.inspect.tensor_classifier import classify_weight


# Container path segments that a model's WeightsMapper commonly rewrites as a
# prefix (``model.``, ``model.language_model.`` -> ``language_model.model.``,
# ``model.visual.`` -> ``visual.``, ...). We strip these leading segments so the
# ignore regex anchors on the stable trailing module path.
_CONTAINER_PREFIX_SEGMENTS = {"model", "language_model", "visual", "vision_model"}


def _prefix_agnostic_ignore(module_name: str) -> str:
    """Turn a checkpoint module name into a prefix-agnostic ``ignore`` regex.

    vLLM matches ``ignore`` against the runtime module path, which a model's
    ``WeightsMapper`` may rewrite relative to the checkpoint name (for example
    ``model.visual.`` -> ``visual.`` or ``model.language_model.`` ->
    ``language_model.model.``). These rewrites only touch leading *container*
    segments, so we drop them and anchor the regex on the stable trailing path
    with a leading ``.*``. The remaining suffix (layer index + module + leaf,
    or the vision block/leaf) uniquely identifies the Linear module.
    """
    parts = module_name.split(".")
    start = 0
    while start < len(parts) - 1 and parts[start] in _CONTAINER_PREFIX_SEGMENTS:
        start += 1
    suffix = ".".join(parts[start:])
    return "re:.*" + re.escape(suffix) + "$"


def module_name_from_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"Expected a logical weight name ending in '.weight': {name}")
    return name[: -len(".weight")]


_COMPACT_TARGETS = {
    "moe.routed": (
        r"re:^.*\.experts\.\d+\."
        r"(?:gate_proj|up_proj|down_proj|w1|w2|w3)$"
    ),
    "moe.shared": (
        r"re:^.*\.(?:shared_expert|shared_experts)\."
        r"(?:gate_proj|up_proj|down_proj|w1|w2|w3)$"
    ),
    "attention": (
        r"re:^.*\.(?:self_attn|attention|attn)\..*"
        r"(?:q_proj|k_proj|v_proj|o_proj|out_proj|query|key|value|dense|"
        r"c_attn|c_proj|qkv_proj|query_key_value|wq|wk|wv|wo|wq_a|wq_b|"
        r"wkv|wkv_a|wkv_b|wo_a|wo_b|kv_a_proj_with_mqa|kv_b_proj|"
        r"kv_proj|q_a_proj|q_b_proj|o_b_proj)$"
    ),
    "mlp": (
        r"re:^.*\.mlp\."
        r"(?:gate_proj|up_proj|down_proj|fc1|fc2|w1|w2|w3|"
        r"dense_h_to_4h|dense_4h_to_h)$"
    ),
}


def _matches_target(module_name: str, target: str) -> bool:
    if target.startswith("re:"):
        return re.search(target[3:], module_name) is not None
    return module_name == target


def compile_compressed_tensors_targets(
    all_logical_weights: Iterable[str],
    selected_logical_weights: Iterable[str],
) -> list[str]:
    """Compile exact tensor selection into compact standard layer targets.

    A category regex is emitted only when it matches exactly the selected
    modules in the scanned checkpoint. Any irregular remainder is represented
    by exact module paths, so compactness never changes quantization intent.
    """
    all_weights = {name for name in all_logical_weights if name.endswith(".weight")}
    selected = {
        name for name in selected_logical_weights if name.endswith(".weight")
    }
    unknown = selected - all_weights
    if unknown:
        raise ValueError(
            "Selected weights are absent from the checkpoint: "
            + ", ".join(sorted(unknown)[:3])
        )

    all_modules = {module_name_from_weight(name) for name in all_weights}
    remaining = {module_name_from_weight(name) for name in selected}
    targets: list[str] = []

    category_modules: dict[str, set[str]] = defaultdict(set)
    for name in all_weights:
        _, tags = classify_weight(name)
        module_name = module_name_from_weight(name)
        for category in _COMPACT_TARGETS:
            if category in tags:
                category_modules[category].add(module_name)

    for category in ("moe.routed", "moe.shared", "attention", "mlp"):
        members = category_modules[category]
        if not members or not members.issubset(remaining):
            continue
        target = _COMPACT_TARGETS[category]
        regex_matches = {
            module for module in all_modules if _matches_target(module, target)
        }
        if regex_matches != members:
            continue
        targets.append(target)
        remaining.difference_update(members)

    targets.extend(sorted(remaining))
    return targets


def validate_fusion_closure(
    all_logical_weights: Iterable[str],
    selected_logical_weights: Iterable[str],
) -> None:
    """Reject tensor selections that split a vLLM fused execution unit."""
    all_weights = set(all_logical_weights)
    selected = set(selected_logical_weights)
    errors: list[str] = []

    # DeepSeek-V4 constructs this fused state-compressor projection with
    # quant_config=None in vLLM. Packed integer source tensors therefore have
    # no matching runtime parameters (for example ``weight_packed``), even if
    # both source halves are selected. Keep the pair in BF16 until that runtime
    # module supports a quantization config.
    unsupported_compressor = sorted(
        name
        for name in selected
        if name.endswith(
            (".compressor.wkv.weight", ".compressor.wgate.weight")
        )
    )
    if unsupported_compressor:
        errors.append(
            "DeepSeek-V4 compressor wkv/wgate must remain BF16 because the "
            "vLLM fused_wkv_wgate runtime module does not accept a quantization "
            "config: "
            + ", ".join(unsupported_compressor[:4])
        )

    paired_suffixes = (
        (".gate_proj.weight", ".up_proj.weight"),
        (".w1.weight", ".w3.weight"),
        (".q_a_proj.weight", ".kv_a_proj_with_mqa.weight"),
        (".wq_a.weight", ".wkv.weight"),
        (".wk.weight", ".weights_proj.weight"),
    )
    visited_pairs: set[frozenset[str]] = set()
    for name in all_weights:
        for left_suffix, right_suffix in paired_suffixes:
            if name.endswith(left_suffix):
                sibling = name[: -len(left_suffix)] + right_suffix
            elif name.endswith(right_suffix):
                sibling = name[: -len(right_suffix)] + left_suffix
            else:
                continue
            pair = frozenset((name, sibling))
            if sibling not in all_weights or pair in visited_pairs:
                continue
            visited_pairs.add(pair)
            selected_count = sum(item in selected for item in pair)
            if selected_count == 1:
                errors.append(
                    "fused pair has mixed formats: " + ", ".join(sorted(pair))
                )

    routed_pattern = re.compile(
        r"^(?P<bank>.*\.experts)\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$"
    )
    routed_banks: dict[str, set[str]] = defaultdict(set)
    for name in all_weights:
        match = routed_pattern.match(name)
        if match:
            routed_banks[match.group("bank")].add(name)
    for bank, members in routed_banks.items():
        selected_count = len(members & selected)
        if selected_count not in (0, len(members)):
            errors.append(
                f"routed MoE bank {bank} is partially selected "
                f"({selected_count}/{len(members)} weights)"
            )

    if errors:
        details = "\n  - ".join(errors[:8])
        raise ValueError(
            "The selected tensors split fused inference units. Select the whole "
            f"unit or change the recipe:\n  - {details}"
        )


def build_compressed_tensors_config(
    all_logical_weights: Iterable[str],
    selected_logical_weights: Iterable[str],
    *,
    num_bits: int = 4,
    activation_num_bits: int = 16,
    strategy: str = "group",
    group_size: int | None = None,
    ignore_modules: Iterable[str] = (),
    additional_targets: Iterable[str] = (),
) -> dict:
    if num_bits not in (4, 8):
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    if activation_num_bits not in (8, 16):
        raise ValueError(
            f"activation_num_bits must be 8 or 16, got {activation_num_bits}"
        )
    if strategy not in {"group", "channel"}:
        raise ValueError(f"Unsupported weight strategy: {strategy}")
    if strategy == "channel" and num_bits != 8:
        raise ValueError("channel strategy is currently supported only for INT8")
    if strategy == "group" and (group_size is None or group_size <= 0):
        raise ValueError("group strategy requires a positive group_size")
    if strategy == "channel" and group_size is not None:
        raise ValueError("channel strategy must not declare group_size")
    if activation_num_bits == 8 and (num_bits != 8 or strategy != "channel"):
        raise ValueError(
            "W8A8 requires 8-bit weights with channel weight strategy"
        )
    validate_fusion_closure(all_logical_weights, selected_logical_weights)
    targets = compile_compressed_tensors_targets(
        all_logical_weights, selected_logical_weights
    )
    targets.extend(
        target for target in sorted(set(additional_targets)) if target not in targets
    )
    if not targets:
        raise ValueError("Cannot build compressed-tensors config without targets")
    # The vLLM compressed-tensors loader calls ``get_scheme`` for every Linear
    # (and MoE) module. Any Linear that is NOT quantized must be listed in
    # ``ignore`` or the loader raises "Unable to find matching target". We list
    # the unquantized Linear modules explicitly so unrelated 1D weights
    # (norms, embeddings) are never touched.
    ignore = sorted(
        _prefix_agnostic_ignore(module) for module in set(ignore_modules)
    )
    weights = {
        "num_bits": num_bits,
        "type": "int",
        "strategy": strategy,
        "symmetric": True,
        "dynamic": False,
    }
    if strategy == "group":
        weights["group_size"] = group_size
    group_name = (
        "w8a8_channel_token"
        if activation_num_bits == 8
        else (
            f"w{num_bits}a16_g{group_size}"
            if strategy == "group"
            else f"w{num_bits}a16_channel"
        )
    )
    group = {
        "targets": targets,
        "weights": weights,
    }
    if activation_num_bits == 8:
        group["input_activations"] = {
            "num_bits": 8,
            "type": "int",
            "strategy": "token",
            "symmetric": True,
            "dynamic": True,
        }
    compression_format = (
        "int-quantized"
        if activation_num_bits == 8
        else "pack-quantized"
    )
    group["format"] = compression_format
    return {
        "quant_method": "compressed-tensors",
        "format": compression_format,
        "quantization_status": "compressed",
        "config_groups": {
            group_name: group,
        },
        "ignore": ignore,
    }
