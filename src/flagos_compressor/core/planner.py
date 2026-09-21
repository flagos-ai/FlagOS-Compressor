from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from flagos_compressor.core.compressed_tensors import validate_fusion_closure
from flagos_compressor.core.moe_layout import MoeLayout
from flagos_compressor.core.plan import ExecutionPlan, FormatSpec
from flagos_compressor.core.policy import QuantizationPolicy, UnselectedWeightsPolicy
from flagos_compressor.core.profile import ModelProfile, TensorInfo


def _is_scaled_fp4(tensor: TensorInfo) -> bool:
    return (
        tensor.role != "scale"
        and bool(tensor.scale_name)
        and tensor.storage_format == "fp4_e2m1_e8m0"
    )


def _is_scaled_fp8(tensor: TensorInfo) -> bool:
    return (
        tensor.role != "scale"
        and bool(tensor.scale_name)
        and tensor.storage_format == "fp8_block_e8m0"
    )


def _is_float_weight(tensor: TensorInfo) -> bool:
    return (
        tensor.role == "weight"
        and tensor.scale_name is None
        and tensor.dtype in {"bfloat16", "float16", "float32"}
        and len(tensor.shape) == 2
    )


def _fused_expert_proj_kind(tensor: TensorInfo) -> str | None:
    """Return the projection leaf of a fused routed-expert bank, else None."""
    if tensor.module_kind != "moe_routed_fused":
        return None
    return tensor.name.split(".")[-1]


def _is_fused_moe_expert(tensor: TensorInfo) -> bool:
    """A 3D fused routed-expert bank (``[num_experts, out, in]``) in float."""
    return (
        tensor.role == "weight"
        and tensor.scale_name is None
        and tensor.dtype in {"bfloat16", "float16", "float32"}
        and len(tensor.shape) == 3
        and _fused_expert_proj_kind(tensor) is not None
    )


def _input_format_for(tensor: TensorInfo) -> FormatSpec | None:
    if _is_scaled_fp4(tensor):
        return FormatSpec("fp4_e2m1_e8m0")
    if _is_scaled_fp8(tensor):
        return FormatSpec("fp8_block_e8m0", {"block_size": 128})
    if _is_float_weight(tensor):
        return FormatSpec(
            {
                "bfloat16": "bf16",
                "float16": "fp16",
                "float32": "fp32",
            }[tensor.dtype]
        )
    return None


def _bf16_output() -> FormatSpec:
    return FormatSpec("bf16")


def _plan_unselected_weight(
    plan: ExecutionPlan,
    tensor: TensorInfo,
    policy: UnselectedWeightsPolicy,
) -> None:
    """Plan one unselected weight using an inference-safe strategy.

    New strategies belong here only after their checkpoint metadata and
    runtime loader contract are implemented end to end.
    """
    if policy.strategy != "convert" or policy.format != "bf16":
        requested = (
            policy.strategy
            if policy.format is None
            else f"{policy.strategy}:{policy.format}"
        )
        raise ValueError(
            f"Unsupported unselected-weight strategy {requested!r}. "
            "Preserving a source format requires a runtime config exporter "
            "and a compatible inference kernel."
        )

    if _is_scaled_fp4(tensor):
        plan.add_action(
            tensor,
            input_format=FormatSpec("fp4_e2m1_e8m0"),
            output_format=_bf16_output(),
            rule_name="unselected_convert_bf16",
        )
    elif _is_scaled_fp8(tensor):
        plan.add_action(
            tensor,
            input_format=FormatSpec("fp8_block_e8m0", {"block_size": 128}),
            output_format=_bf16_output(),
            rule_name="unselected_convert_bf16",
        )
    else:
        plan.kept_tensors.append(tensor)
        if (
            tensor.element_size == 1
            and tensor.scale_name
            and tensor.storage_format is None
        ):
            plan.unmatched_quantized_tensors.append(tensor)


def _fused_bank_module(tensor: TensorInfo) -> str:
    """The ``...experts`` module prefix shared by one routed bank's projections."""
    return tensor.name.rsplit(".", 1)[0]


def _plan_fused_moe_expert(
    plan: ExecutionPlan,
    tensor: TensorInfo,
    policy: QuantizationPolicy,
    proj_kind: str,
    layout: MoeLayout,
) -> None:
    """Plan integer quantization of a fused 3D routed-expert bank.

    Each expert is quantized as an independent 2D ``[out, in]`` weight. The
    per-expert ``(out, in)`` is derived from the model's :class:`MoeLayout`
    (axis order is model specific). W4A16/W8A16 use groupwise packed weights;
    W8A8 uses raw INT8 weights with per-output-channel scales. A fused
    ``gate_up_proj`` must have an even split dimension.
    """
    shape = tensor.effective_logical_shape
    if policy.strategy == "channel" and not policy.is_w8a8:
        raise ValueError(
            "Fused routed MoE channel quantization requires W8A8; "
            "set activation_num_bits=8 (CLI: --activation-bits 8)"
        )
    if policy.strategy == "group" and policy.group_size is None:
        raise ValueError(
            "Fused routed MoE group quantization requires a group_size"
        )
    if len(shape) != 3:
        raise ValueError(
            f"Selected fused expert {tensor.name!r} has shape {shape}; expected 3D"
        )
    num_experts = int(shape[0])
    out_features, in_features = layout.per_expert_out_in(proj_kind, shape)
    if proj_kind == "gate_up_proj" and int(shape[1] if layout.out_in_order else shape[2]) % 2:
        raise ValueError(
            f"Selected fused expert {tensor.name!r} has an odd fused output "
            "dimension; cannot split into gate and up projections"
        )
    if (
        policy.strategy == "group"
        and policy.group_size is not None
        and in_features % policy.group_size
    ):
        raise ValueError(
            f"Selected fused expert {tensor.name!r} has in_features={in_features}; "
            f"must be divisible by group_size={policy.group_size}"
        )
    if policy.num_bits == 4 and in_features % 8:
        raise ValueError(
            f"Selected fused expert {tensor.name!r} has in_features={in_features}; "
            "INT4 pack-quantized storage requires in_features divisible by 8"
        )
    plan.add_action(
        tensor,
        input_format=FormatSpec("bf16"),
        output_format=FormatSpec(
            (
                "compressed_tensors_w8a8_channelwise_moe_fused"
                if policy.is_w8a8
                else f"compressed_tensors_int{policy.num_bits}_moe_fused"
            ),
            {
                "quantizer": policy.method,
                "num_bits": policy.num_bits,
                "activation_num_bits": policy.activation_num_bits,
                "scale_dtype": (
                    policy.scale_dtype if policy.is_w8a8 else "bfloat16"
                ),
                "strategy": policy.strategy,
                "group_size": policy.group_size,
                "n_candidates": policy.n_candidates,
                "chunk_size": policy.chunk_size,
                "proj_kind": proj_kind,
                "layout": layout.name,
                "num_experts": num_experts,
                "out_features": out_features,
                "in_features": in_features,
            },
        ),
        rule_name=f"user_selected_int{policy.num_bits}_moe",
    )


def _validate_fused_bank_closure(
    profile: ModelProfile,
    selected_banks: list[TensorInfo],
) -> None:
    """Reject partial selection that would split a routed expert bank.

    vLLM treats the whole FusedMoE (gate/up/down of every expert) as one
    quantized unit. If any projection bank of an ``experts`` module is selected,
    all of that module's banks must be selected too — otherwise the exported
    config marks the unit quantized while some projections are still bf16, which
    fails at load time.
    """
    selected_by_module: dict[str, set[str]] = defaultdict(set)
    for tensor in selected_banks:
        selected_by_module[_fused_bank_module(tensor)].add(
            tensor.name.split(".")[-1]
        )
    all_by_module: dict[str, set[str]] = defaultdict(set)
    for tensor in profile.tensors.values():
        if tensor.module_kind == "moe_routed_fused":
            all_by_module[_fused_bank_module(tensor)].add(
                tensor.name.split(".")[-1]
            )
    errors: list[str] = []
    for module, chosen in selected_by_module.items():
        present = all_by_module.get(module, set())
        missing = present - chosen
        if missing:
            errors.append(
                f"routed expert bank {module!r} is partially selected: "
                f"chose {sorted(chosen)} but {sorted(missing)} are not selected"
            )
    if errors:
        details = "\n  - ".join(errors[:8])
        raise ValueError(
            "The selection splits fused routed-expert banks; select the whole "
            f"expert module (gate/up/down) or none:\n  - {details}"
        )


def build_convert_plan(profile: ModelProfile) -> ExecutionPlan:
    """Build a dequantization plan.

    All scaled FP8/FP4 weights are converted to BF16. Every other tensor is
    copied through unchanged.
    """
    plan = ExecutionPlan(metadata={"command": "convert", "artifact_kind": "bf16"})
    for tensor in profile.tensors.values():
        if tensor.role != "weight":
            continue
        if _is_scaled_fp4(tensor):
            plan.add_action(
                tensor,
                input_format=FormatSpec("fp4_e2m1_e8m0"),
                output_format=_bf16_output(),
            )
        elif _is_scaled_fp8(tensor):
            plan.add_action(
                tensor,
                input_format=FormatSpec("fp8_block_e8m0", {"block_size": 128}),
                output_format=_bf16_output(),
            )
        else:
            plan.kept_tensors.append(tensor)
            if tensor.element_size == 1 and tensor.scale_name:
                plan.unmatched_quantized_tensors.append(tensor)
    return plan


def build_quantize_plan(
    profile: ModelProfile,
    policy: QuantizationPolicy,
    moe_layout: MoeLayout | None = None,
) -> ExecutionPlan:
    if policy.target_scheme_rules:
        return build_per_selector_plan(profile, policy, moe_layout)
    plan = ExecutionPlan(
        metadata={
            "command": "quantize",
            "artifact_kind": "compressed_tensors",
            "unselected_weights": {
                "strategy": policy.unselected.strategy,
                "format": policy.unselected.format,
            },
            "algorithm": {
                "mode": "uniform",
                "name": (
                    "mse_grid_search"
                    if policy.method == "mse"
                    else policy.method
                ),
                "num_bits": policy.num_bits,
                "activation_num_bits": policy.activation_num_bits,
                "scale_dtype": (
                    policy.scale_dtype if policy.is_w8a8 else "bfloat16"
                ),
                "strategy": policy.strategy,
                "group_size": policy.group_size,
                "n_candidates": policy.n_candidates,
                "chunk_size": policy.chunk_size,
            },
        }
    )
    selected_banks: list[TensorInfo] = []
    for tensor in profile.tensors.values():
        if tensor.role != "weight":
            continue

        if policy.selects(tensor):
            if (
                policy.strategy == "channel"
                and "moe.routed" in tensor.tags
                and not policy.is_w8a8
            ):
                raise ValueError(
                    f"Selected routed MoE tensor {tensor.name!r} cannot use "
                    "channel strategy in W8A16; vLLM WNA16 MoE requires group "
                    "strategy. Use --activation-bits 8 for W8A8"
                )
            proj_kind = _fused_expert_proj_kind(tensor)
            if _is_fused_moe_expert(tensor):
                if moe_layout is None:
                    raise ValueError(
                        f"Selected fused routed-expert bank {tensor.name!r} "
                        "requires a MoeLayout. Pass moe_layout=select_moe_layout"
                        "(config) so the expert axis order is known."
                    )
                selected_banks.append(tensor)
                _plan_fused_moe_expert(plan, tensor, policy, proj_kind, moe_layout)
                continue
            input_format = _input_format_for(tensor)
            if input_format is None:
                raise ValueError(
                    f"Selected tensor {tensor.name!r} cannot be quantized to "
                    f"INT{policy.num_bits} "
                    f"(dtype={tensor.dtype}, storage_format={tensor.storage_format}, shape={tensor.shape})"
                )
            logical_shape = tensor.effective_logical_shape
            if len(logical_shape) != 2:
                raise ValueError(
                    f"Selected tensor {tensor.name!r} has logical shape {logical_shape}; "
                    "expected a 2D weight"
                )
            if (
                policy.strategy == "group"
                and (
                    policy.group_size is None
                    or logical_shape[1] % policy.group_size
                )
            ):
                raise ValueError(
                    f"Selected tensor {tensor.name!r} has logical shape {logical_shape}; "
                    f"in_features must be divisible by group_size={policy.group_size}"
                )
            if policy.num_bits == 4 and logical_shape[1] % 8:
                raise ValueError(
                    f"Selected tensor {tensor.name!r} has in_features="
                    f"{logical_shape[1]}; INT4 pack-quantized storage requires "
                    "in_features divisible by 8"
                )
            plan.add_action(
                tensor,
                input_format=input_format,
                output_format=FormatSpec(
                    (
                        "compressed_tensors_w8a8_channelwise"
                        if policy.is_w8a8
                        else (
                            "compressed_tensors_int8_channelwise"
                            if policy.strategy == "channel"
                            else f"compressed_tensors_int{policy.num_bits}_groupwise"
                        )
                    ),
                    {
                        "quantizer": policy.method,
                        "num_bits": policy.num_bits,
                        "activation_num_bits": policy.activation_num_bits,
                        "scale_dtype": (
                            policy.scale_dtype
                            if policy.is_w8a8
                            else "bfloat16"
                        ),
                        "strategy": policy.strategy,
                        "group_size": policy.group_size,
                        "n_candidates": policy.n_candidates,
                        "chunk_size": policy.chunk_size,
                    },
                ),
                rule_name=f"user_selected_int{policy.num_bits}",
            )
        else:
            _plan_unselected_weight(plan, tensor, policy.unselected)
    validate_fusion_closure(
        (
            tensor.name
            for tensor in profile.tensors.values()
            if tensor.role == "weight" and tensor.name.endswith(".weight")
        ),
        (
            action.tensor.name
            for action in plan.actions
            if action.output_format.name
            in {
                "compressed_tensors_int4_groupwise",
                "compressed_tensors_int8_groupwise",
                "compressed_tensors_int8_channelwise",
                "compressed_tensors_w8a8_channelwise",
            }
        ),
    )
    _validate_fused_bank_closure(profile, selected_banks)
    return plan


def build_per_selector_plan(
    profile: ModelProfile,
    policy: QuantizationPolicy,
    moe_layout: MoeLayout | None = None,
) -> ExecutionPlan:
    """Build one artifact from ordered selector-local quantization rules."""
    if not policy.target_scheme_rules:
        raise ValueError("build_per_selector_plan requires target_scheme_rules")
    if policy.method != "mse":
        raise ValueError("Per-selector formats currently require MSE")

    plan = ExecutionPlan(
        metadata={
            "command": "quantize",
            "artifact_kind": "compressed_tensors",
            "unselected_weights": {
                "strategy": policy.unselected.strategy,
                "format": policy.unselected.format,
            },
            "algorithm": {
                "name": "per_selector_mse_grid_search",
                "mode": "per_selector",
                "n_candidates": policy.n_candidates,
                "rules": [
                    {
                        "selector": rule.label,
                        "scheme": rule.scheme,
                        "num_bits": rule.num_bits,
                        "activation_num_bits": rule.activation_num_bits,
                        "strategy": rule.strategy,
                        "group_size": rule.group_size,
                        "chunk_size": rule.chunk_size,
                        "scale_dtype": (
                            rule.scale_dtype if rule.is_w8a8 else "bfloat16"
                        ),
                    }
                    for rule in policy.target_scheme_rules
                ],
            },
        }
    )

    selected_by_scheme: dict[tuple[str, int | None], list[str]] = defaultdict(list)
    selected_banks_by_scheme: dict[
        tuple[str, int | None], list[TensorInfo]
    ] = defaultdict(list)
    for tensor in profile.tensors.values():
        if tensor.role != "weight":
            continue
        rule = policy.target_scheme_rule_for(tensor)
        if rule is None:
            _plan_unselected_weight(plan, tensor, policy.unselected)
            continue

        assert rule.chunk_size is not None
        num_bits = rule.num_bits
        group_size = rule.group_size
        scheme = (rule.scheme, group_size)
        effective_policy = replace(
            policy,
            target_scheme_rules=(),
            num_bits=num_bits,
            activation_num_bits=rule.activation_num_bits,
            scale_dtype=rule.scale_dtype or "float32",
            strategy=rule.strategy,
            group_size=group_size,
            chunk_size=rule.chunk_size,
        )
        proj_kind = _fused_expert_proj_kind(tensor)
        if _is_fused_moe_expert(tensor):
            if moe_layout is None:
                raise ValueError(
                    f"Selected fused routed-expert bank {tensor.name!r} "
                    "requires a MoeLayout"
                )
            _plan_fused_moe_expert(
                plan,
                tensor,
                effective_policy,
                proj_kind,
                moe_layout,
            )
            selected_banks_by_scheme[scheme].append(tensor)
            continue

        input_format = _input_format_for(tensor)
        if input_format is None:
            raise ValueError(
                f"Scheme selector {rule.label!r} selected unsupported tensor "
                f"{tensor.name!r} (dtype={tensor.dtype}, "
                f"storage_format={tensor.storage_format}, shape={tensor.shape})"
            )
        logical_shape = tensor.effective_logical_shape
        if len(logical_shape) != 2:
            raise ValueError(
                f"Per-selector input {tensor.name!r} has logical shape "
                f"{logical_shape}; expected a 2D weight"
            )
        if rule.strategy == "group" and (
            group_size is None or logical_shape[1] % group_size
        ):
            raise ValueError(
                f"Per-selector input {tensor.name!r} has in_features="
                f"{logical_shape[1]}; must be divisible by group_size={group_size}"
            )
        if num_bits == 4 and logical_shape[1] % 8:
            raise ValueError(
                f"Per-selector input {tensor.name!r} has in_features="
                f"{logical_shape[1]}; INT4 pack-quantized storage requires "
                "in_features divisible by 8"
            )
        if rule.is_w8a8:
            output_format = "compressed_tensors_w8a8_channelwise"
        elif rule.strategy == "channel":
            output_format = "compressed_tensors_int8_channelwise"
        else:
            output_format = f"compressed_tensors_int{num_bits}_groupwise"
        plan.add_action(
            tensor,
            input_format=input_format,
            output_format=FormatSpec(
                output_format,
                {
                    "quantizer": "mse",
                    "num_bits": num_bits,
                    "activation_num_bits": rule.activation_num_bits,
                    "scale_dtype": (
                        rule.scale_dtype if rule.is_w8a8 else "bfloat16"
                    ),
                    "strategy": rule.strategy,
                    "group_size": group_size,
                    "n_candidates": policy.n_candidates,
                    "chunk_size": rule.chunk_size,
                },
            ),
            rule_name=f"selected_{rule.label}_{rule.scheme}",
        )
        selected_by_scheme[scheme].append(tensor.name)

    all_logical_weights = [
        tensor.name
        for tensor in profile.tensors.values()
        if tensor.role == "weight" and tensor.name.endswith(".weight")
    ]
    for selected in selected_by_scheme.values():
        validate_fusion_closure(all_logical_weights, selected)
    for selected_banks in selected_banks_by_scheme.values():
        _validate_fused_bank_closure(profile, selected_banks)
    return plan


def build_plan(profile: ModelProfile) -> ExecutionPlan:
    """Backwards-compatible alias for the BF16 conversion planner."""
    return build_convert_plan(profile)
