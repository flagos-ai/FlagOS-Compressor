import pytest

from flagos_compressor.core.planner import build_quantize_plan
from flagos_compressor.core.policy import (
    QuantizationPolicy,
    TargetSchemeRule,
    UnselectedWeightsPolicy,
)
from flagos_compressor.core.profile import ModelProfile, TensorInfo


def _tensor(name, *, storage_format=None, dtype="bfloat16", shape=(4, 32), scale_name=None, tags=()):
    logical_shape = (shape[0], shape[1] * 2) if storage_format == "fp4_e2m1_e8m0" else shape
    return TensorInfo(
        name=name,
        shape=shape,
        logical_shape=logical_shape,
        dtype=dtype,
        shard="model.safetensors",
        element_size=1 if storage_format else 2,
        scale_name=scale_name,
        storage_format=storage_format,
        tags=tags,
    )


def _profile():
    profile = ModelProfile(model_path="/tmp/model")
    routed = "model.layers.0.mlp.experts.0.w1.weight"
    shared = "model.layers.0.mlp.shared_experts.w1.weight"
    attention = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors = {
        routed: _tensor(
            routed,
            storage_format="fp4_e2m1_e8m0",
            dtype="uint8",
            shape=(4, 16),
            scale_name=routed + ".source_scale",
            tags=("linear", "moe", "moe.routed"),
        ),
        shared: _tensor(shared, tags=("linear", "moe", "moe.shared")),
        attention: _tensor(attention, tags=("attention", "linear")),
    }
    return profile


def test_moe_selection_includes_routed_and_shared():
    plan = build_quantize_plan(_profile(), QuantizationPolicy(selections=("moe",)))
    assert plan.metadata["algorithm"]["mode"] == "uniform"
    assert plan.input_format_counts == {"fp4_e2m1_e8m0": 1, "bf16": 1}
    assert plan.output_format_counts == {
        "compressed_tensors_int4_groupwise": 2
    }
    assert len(plan.kept_tensors) == 1


def test_regex_can_select_arbitrary_linear():
    policy = QuantizationPolicy(include_names=(r"self_attn\.o_proj\.weight$",))
    plan = build_quantize_plan(_profile(), policy)
    assert plan.output_format_counts["compressed_tensors_int4_groupwise"] == 1
    assert plan.output_format_counts["bf16"] == 1


def test_builtin_exclude_removes_shared_experts():
    policy = QuantizationPolicy(selections=("moe",), exclude_selections=("moe.shared",))
    plan = build_quantize_plan(_profile(), policy)
    assert plan.output_format_counts["compressed_tensors_int4_groupwise"] == 1


def test_preserve_policy_has_an_explicit_runtime_contract_boundary():
    policy = QuantizationPolicy(
        selections=("moe",),
        unselected=UnselectedWeightsPolicy(strategy="preserve", format=None),
    )
    with pytest.raises(ValueError, match="runtime config exporter"):
        build_quantize_plan(_profile(), policy)


def test_selected_shape_must_align_to_group_size():
    profile = _profile()
    name = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[name] = _tensor(name, shape=(4, 30), tags=("attention", "linear"))
    with pytest.raises(ValueError, match="divisible"):
        build_quantize_plan(profile, QuantizationPolicy(selections=("attention",)))


def test_selected_shape_must_align_to_pack_word():
    profile = _profile()
    name = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[name] = _tensor(name, shape=(4, 6), tags=("attention", "linear"))
    with pytest.raises(ValueError, match="divisible by 8"):
        build_quantize_plan(
            profile,
            QuantizationPolicy(selections=("attention",), group_size=2),
        )


def test_int8_selection_uses_w8a16_format_without_int4_pack_constraint():
    profile = _profile()
    name = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[name] = _tensor(
        name,
        shape=(4, 12),
        tags=("attention", "linear"),
    )
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("attention",),
            num_bits=8,
            group_size=4,
        ),
    )
    assert plan.output_format_counts["compressed_tensors_int8_groupwise"] == 1
    action = next(
        action
        for action in plan.actions
        if action.output_format.name == "compressed_tensors_int8_groupwise"
    )
    assert action.output_format.params["num_bits"] == 8


def test_int8_channel_selection_does_not_require_group_alignment():
    profile = _profile()
    name = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[name] = _tensor(
        name,
        shape=(4, 13),
        tags=("attention", "linear"),
    )
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("attention",),
            num_bits=8,
            strategy="channel",
        ),
    )
    assert plan.output_format_counts == {
        "compressed_tensors_int8_channelwise": 1,
        "bf16": 1,
    }
    channel_action = next(
        action
        for action in plan.actions
        if action.output_format.name == "compressed_tensors_int8_channelwise"
    )
    assert channel_action.output_format.params["strategy"] == "channel"
    assert channel_action.output_format.params["group_size"] is None


def test_w8a8_plan_propagates_bf16_scale_dtype():
    profile = _profile()
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("attention",),
            num_bits=8,
            activation_num_bits=8,
            strategy="channel",
            scale_dtype="bf16",
        ),
    )
    action = next(
        action
        for action in plan.actions
        if action.output_format.name == "compressed_tensors_w8a8_channelwise"
    )
    assert action.output_format.params["scale_dtype"] == "bfloat16"


def test_int8_channel_rejects_routed_moe():
    with pytest.raises(ValueError, match="MoE.*group strategy"):
        build_quantize_plan(
            _profile(),
            QuantizationPolicy(
                selections=("moe.routed",),
                num_bits=8,
                strategy="channel",
            ),
        )


def test_per_selector_plan_routes_modules_to_requested_integer_formats():
    profile = _profile()
    attention = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[attention] = _tensor(
        attention,
        storage_format="fp8_block_e8m0",
        dtype="float8_e4m3fn",
        shape=(128, 128),
        scale_name=attention + ".source_scale",
        tags=("attention", "linear"),
    )

    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            target_scheme_rules=(
                TargetSchemeRule(
                    selection="moe",
                    weight_format="int4",
                    activation_num_bits=16,
                    strategy="group",
                ),
                TargetSchemeRule(
                    selection="attention",
                    weight_format="int8",
                    activation_num_bits=16,
                    strategy="group",
                ),
            ),
            n_candidates=8,
        ),
    )

    assert plan.input_format_counts == {
        "bf16": 1,
        "fp4_e2m1_e8m0": 1,
        "fp8_block_e8m0": 1,
    }
    assert plan.output_format_counts == {
        "compressed_tensors_int4_groupwise": 2,
        "compressed_tensors_int8_groupwise": 1,
    }
    params = {
        action.tensor.name: action.output_format.params
        for action in plan.actions
        if action.output_format.name.startswith("compressed_tensors_int")
    }
    assert params["model.layers.0.mlp.experts.0.w1.weight"]["group_size"] == 32
    assert params[attention]["group_size"] == 128
    assert plan.metadata["algorithm"]["mode"] == "per_selector"


def test_later_formatted_selector_overrides_a_broader_rule():
    profile = _profile()
    policy = QuantizationPolicy(
        target_scheme_rules=(
            TargetSchemeRule(
                selection="linear",
                weight_format="int4",
                activation_num_bits=16,
                strategy="group",
            ),
            TargetSchemeRule(
                name_pattern=r"self_attn\..*",
                weight_format="int8",
                activation_num_bits=16,
                strategy="group",
                group_size=32,
            ),
        )
    )

    plan = build_quantize_plan(profile, policy)

    attention = next(
        action
        for action in plan.actions
        if action.tensor.name.endswith("self_attn.o_proj.weight")
    )
    assert attention.output_format.name == "compressed_tensors_int8_groupwise"


def test_same_explicit_formats_still_use_per_selector_mode():
    plan = build_quantize_plan(
        _profile(),
        QuantizationPolicy(
            target_scheme_rules=(
                TargetSchemeRule(
                    selection="moe",
                    weight_format="int4",
                    activation_num_bits=16,
                    strategy="group",
                ),
                TargetSchemeRule(
                    selection="attention",
                    weight_format="int4",
                    activation_num_bits=16,
                    strategy="group",
                ),
            )
        ),
    )

    assert plan.metadata["algorithm"]["mode"] == "per_selector"
    assert plan.output_format_counts == {
        "compressed_tensors_int4_groupwise": 3
    }


def test_per_selector_plan_supports_w4a16_and_w8a8_together():
    profile = _profile()
    attention = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[attention] = _tensor(
        attention,
        storage_format="fp8_block_e8m0",
        dtype="float8_e4m3fn",
        shape=(128, 128),
        scale_name=attention + ".source_scale",
        tags=("attention", "linear"),
    )
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            target_scheme_rules=(
                TargetSchemeRule(
                    selection="moe",
                    weight_format="int4",
                    activation_num_bits=16,
                    strategy="group",
                ),
                TargetSchemeRule(
                    selection="attention",
                    weight_format="int8",
                    activation_num_bits=8,
                    strategy="channel",
                ),
            ),
            n_candidates=8,
        ),
    )

    assert plan.output_format_counts == {
        "compressed_tensors_int4_groupwise": 2,
        "compressed_tensors_w8a8_channelwise": 1,
    }
    attention_action = next(
        action for action in plan.actions if action.tensor.name == attention
    )
    assert attention_action.output_format.params == {
        "quantizer": "mse",
        "num_bits": 8,
        "activation_num_bits": 8,
        "scale_dtype": "float32",
        "strategy": "channel",
        "group_size": None,
        "n_candidates": 8,
        "chunk_size": 1024,
    }


def test_per_selector_distinguishes_groupwise_and_channelwise_w8a16():
    plan = build_quantize_plan(
        _profile(),
        QuantizationPolicy(
            target_scheme_rules=(
                TargetSchemeRule(
                    selection="moe",
                    weight_format="int8",
                    activation_num_bits=16,
                    strategy="group",
                    group_size=32,
                ),
                TargetSchemeRule(
                    selection="attention",
                    weight_format="int8",
                    activation_num_bits=16,
                    strategy="channel",
                ),
            )
        ),
    )

    assert plan.output_format_counts == {
        "compressed_tensors_int8_groupwise": 2,
        "compressed_tensors_int8_channelwise": 1,
    }
    assert [
        (rule["scheme"], rule["strategy"], rule["group_size"])
        for rule in plan.metadata["algorithm"]["rules"]
    ] == [
        ("int8-a16", "group", 32),
        ("int8-a16", "channel", None),
    ]
