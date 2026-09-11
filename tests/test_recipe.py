from argparse import Namespace
from pathlib import Path

import pytest

from flagos_compressor.cli.helpers import build_quantization_policy
from flagos_compressor.core.policy import UnselectedWeightsPolicy


def test_simple_recipe_builds_policy(tmp_path):
    recipe = tmp_path / "quantize.yaml"
    recipe.write_text(
        """
version: 1
bits: 8
method: mse
group_size: 128
unselected:
  strategy: convert
  format: bf16
select:
  - moe
  - name: '.*\\.o_proj\\.weight$'
exclude:
  - moe.shared
  - name: '.*\\.layers\\.0\\..*'
""",
        encoding="utf-8",
    )
    args = Namespace(
        recipe=str(recipe),
        select=None,
        exclude=None,
        select_name=None,
        exclude_name=None,
        method=None,
        bits=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.selections == ("moe",)
    assert policy.exclude_selections == ("moe.shared",)
    assert policy.num_bits == 8
    assert policy.group_size == 128
    assert policy.include_names == (r".*\.o_proj\.weight$",)
    assert policy.unselected == UnselectedWeightsPolicy(
        strategy="convert",
        format="bf16",
    )


def test_version_one_rejects_calibrated_recipe_fields(tmp_path):
    recipe = tmp_path / "legacy-gptq.yaml"
    recipe.write_text(
        """
version: 1
method: gptq
calibration:
  data: calibration.txt
select:
  - linear
""",
        encoding="utf-8",
    )
    args = Namespace(recipe=str(recipe))

    with pytest.raises(ValueError, match="version 2"):
        build_quantization_policy(args)


def test_int8_cli_uses_w8a16_default_group_size():
    args = Namespace(
        recipe=None,
        select=["attention"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.num_bits == 8
    assert policy.group_size == 128
    assert policy.chunk_size == 1024


def test_int8_channel_cli_omits_group_size():
    args = Namespace(
        recipe=None,
        select=["attention"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        strategy="channel",
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.strategy == "channel"
    assert policy.group_size is None


def test_int8_channel_rejects_explicit_group_size():
    args = Namespace(
        recipe=None,
        select=["attention"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        strategy="channel",
        method=None,
        group_size=128,
        n_candidates=None,
        chunk_size=None,
    )
    with pytest.raises(ValueError, match="group_size must be omitted"):
        build_quantization_policy(args)


def test_w8a8_cli_builds_dynamic_token_policy():
    args = Namespace(
        recipe=None,
        select=["moe.routed"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        activation_bits=8,
        scale_dtype="bf16",
        strategy="channel",
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    policy = build_quantization_policy(args)
    assert policy.is_w8a8
    assert policy.scale_dtype == "bfloat16"
    assert policy.group_size is None


def test_w8a8_recipe_supports_bf16_scale_dtype(tmp_path):
    recipe = tmp_path / "w8a8.yaml"
    recipe.write_text(
        """
version: 1
bits: 8
activation_bits: 8
strategy: channel
scale_dtype: bf16
select:
  - attention
""",
        encoding="utf-8",
    )
    args = Namespace(
        recipe=str(recipe),
        select=None,
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    policy = build_quantization_policy(args)

    assert policy.is_w8a8
    assert policy.scale_dtype == "bfloat16"


def test_activation_int8_requires_channel_w8():
    args = Namespace(
        recipe=None,
        select=["attention"],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        activation_bits=8,
        strategy="group",
        method=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )
    with pytest.raises(ValueError, match="W8A8 requires"):
        build_quantization_policy(args)


def test_scheme_select_cli_needs_no_global_bits():
    args = Namespace(
        recipe=None,
        select=[
            ["moe=int4", "activation-bits=16", "strategy=group"],
            [
                "attention=int8",
                "activation-bits=16",
                "strategy=group",
                "group-size=64",
            ],
        ],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        format=None,
        group_size=None,
        n_candidates=17,
        chunk_size=None,
    )

    policy = build_quantization_policy(args)

    assert [
        (rule.label, rule.scheme, rule.group_size)
        for rule in policy.target_scheme_rules
    ] == [("moe", "int4-a16", 32), ("attention", "int8-a16", 64)]
    assert policy.method == "mse"
    assert policy.n_candidates == 17


def test_scheme_select_recipe_needs_no_global_bits(tmp_path):
    recipe = tmp_path / "per-selector.yaml"
    recipe.write_text(
        """version: 1
select:
  - target: moe
    weight_format: int4
    activation_bits: 16
    strategy: group
  - name: '.*\\.self_attn\\..*'
    weight_format: int8
    activation_bits: 16
    strategy: group
    group_size: 64
""",
        encoding="utf-8",
    )
    args = Namespace(
        recipe=str(recipe),
        select=None,
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        format=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    policy = build_quantization_policy(args)

    assert [rule.scheme for rule in policy.target_scheme_rules] == [
        "int4-a16",
        "int8-a16",
    ]
    assert policy.target_scheme_rules[1].name_pattern == r".*\.self_attn\..*"
    assert policy.target_scheme_rules[1].group_size == 64
    assert policy.n_candidates == 200


def test_scheme_select_rejects_global_bit_override():
    args = Namespace(
        recipe=None,
        select=[
            ["attention=int8", "activation-bits=8", "strategy=channel"]
        ],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=8,
        activation_bits=None,
        strategy=None,
        method=None,
        format=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    with pytest.raises(ValueError, match="remove global: bits"):
        build_quantization_policy(args)


def test_scheme_and_legacy_selectors_cannot_be_mixed():
    args = Namespace(
        recipe=None,
        select=[
            ["moe=int4", "activation-bits=16", "strategy=group"],
            ["attention"],
        ],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        format=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    with pytest.raises(ValueError, match="cannot be mixed"):
        build_quantization_policy(args)


def test_scheme_selector_distinguishes_unsupported_fp8_from_int8():
    args = Namespace(
        recipe=None,
        select=[
            ["attention=fp8", "activation-bits=8", "strategy=channel"]
        ],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        format=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    with pytest.raises(ValueError, match="Unsupported target weight format 'fp8'"):
        build_quantization_policy(args)


def test_int8_a8_scheme_uses_channel_weights_and_allows_bf16_scales(tmp_path):
    recipe = tmp_path / "w8a8-per-selector.yaml"
    recipe.write_text(
        """version: 1
select:
  - target: attention
    weight_format: int8
    activation_bits: 8
    strategy: channel
    scale_dtype: bf16
""",
        encoding="utf-8",
    )
    args = Namespace(
        recipe=str(recipe),
        select=None,
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        format=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    policy = build_quantization_policy(args)
    rule = policy.target_scheme_rules[0]
    assert rule.scheme == "int8-a8"
    assert rule.activation_num_bits == 8
    assert rule.strategy == "channel"
    assert rule.group_size is None
    assert rule.scale_dtype == "bfloat16"


def test_int8_a8_scheme_rejects_group_size():
    args = Namespace(
        recipe=None,
        select=[
            [
                "attention=int8",
                "activation-bits=8",
                "strategy=channel",
                "group-size=128",
            ]
        ],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        format=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    with pytest.raises(ValueError, match="does not accept group-size"):
        build_quantization_policy(args)


def test_mixed_cli_rule_requires_explicit_granularity():
    args = Namespace(
        recipe=None,
        select=[["attention=int8", "activation-bits=8"]],
        exclude=None,
        select_name=None,
        exclude_name=None,
        bits=None,
        activation_bits=None,
        scale_dtype=None,
        strategy=None,
        method=None,
        format=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    with pytest.raises(ValueError, match="requires explicit local settings: strategy"):
        build_quantization_policy(args)


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
def test_calibrated_example_recipes_build_native_policies(method):
    args = Namespace(
        recipe=str(Path("examples/recipes") / f"{method}.yaml"),
        select=None,
        exclude=None,
        select_name=None,
        exclude_name=None,
        method=None,
        bits=None,
        strategy=None,
        group_size=None,
        n_candidates=None,
        chunk_size=None,
    )

    policy = build_quantization_policy(args)

    assert policy.method == method
    assert policy.format == ("gptq" if method == "autoround" else method)
    assert policy.group_size == 128
    assert policy.calibration.samples == 128
    assert policy.calibration.sequence_length == 512
