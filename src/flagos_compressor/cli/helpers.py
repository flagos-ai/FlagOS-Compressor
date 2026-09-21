from __future__ import annotations

import json
from pathlib import Path

from flagos_compressor.core.plan import ExecutionPlan
from flagos_compressor.core.policy import (
    AWQPolicy,
    AutoRoundPolicy,
    CalibrationPolicy,
    GPTQPolicy,
    QuantizationPolicy,
    TargetSchemeRule,
    UnselectedWeightsPolicy,
)


def print_plan(plan: ExecutionPlan) -> None:
    print("Execution plan")
    algorithm = plan.metadata.get("algorithm") or {}
    if algorithm:
        if algorithm.get("mode") == "per_selector":
            print("  quantization mode: selector-local settings (MSE)")
            for rule in algorithm.get("rules", ()):
                print(
                    "    "
                    f"{rule['selector']} -> {rule['scheme'].upper()} "
                    f"strategy={rule['strategy']}"
                    + (
                        f" group_size={rule['group_size']}"
                        if rule.get("group_size") is not None
                        else ""
                    )
                    + (
                        f" scale_dtype={rule['scale_dtype']}"
                        if rule.get("activation_num_bits") == 8
                        else ""
                    )
                )
        else:
            detail = (
                f"W{algorithm.get('num_bits')}A"
                f"{algorithm.get('activation_num_bits', 16)} "
                f"{algorithm.get('strategy')} "
                f"{algorithm.get('name')}"
            )
            if algorithm.get("group_size") is not None:
                detail += f" group_size={algorithm['group_size']}"
            if algorithm.get("activation_num_bits") == 8:
                detail += (
                    f" scale_dtype={algorithm.get('scale_dtype', 'float32')}"
                )
            print(f"  quantization mode: uniform ({detail})")
    print("  input formats")
    for input_format, count in sorted(plan.input_format_counts.items()):
        print(f"    {input_format}: {count}")
    print("  output formats")
    for output_format, count in sorted(plan.output_format_counts.items()):
        print(f"    {output_format}: {count}")
    print(f"  keep: {len(plan.kept_tensors)}")
    print(f"  unmatched quantized: {len(plan.unmatched_quantized_tensors)}")


def ensure_no_unmatched(plan: ExecutionPlan) -> None:
    if not plan.unmatched_quantized_tensors:
        return
    examples = ", ".join(t.name for t in plan.unmatched_quantized_tensors[:3])
    raise RuntimeError(
        f"Found {len(plan.unmatched_quantized_tensors)} scaled byte tensors with unsupported "
        f"layouts; refusing to guess their formats. Examples: {examples}"
    )


def load_quantize_recipe(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Recipe support requires PyYAML; install flagos-compressor dependencies"
        ) from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Quantize recipe must be a YAML mapping")
    allowed = {
        "version",
        "bits",
        "activation_bits",
        "scale_dtype",
        "strategy",
        "method",
        "format",
        "group_size",
        "n_candidates",
        "chunk_size",
        "select",
        "exclude",
        "unselected",
        "calibration",
        "gptq",
        "awq",
        "autoround",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown recipe keys: {', '.join(unknown)}")
    version = data.get("version", 1)
    if version not in {1, 2}:
        raise ValueError(f"Unsupported recipe version: {version}")
    calibrated_keys = {"format", "calibration", "gptq", "awq", "autoround"}
    calibrated_methods = {"gptq", "awq", "autoround"}
    if version == 1 and (
        calibrated_keys & set(data)
        or data.get("method") in calibrated_methods
    ):
        raise ValueError(
            "Recipe version 2 is required for calibrated GPTQ, AWQ, and "
            "AutoRound fields"
        )
    return data


def _parse_selector_items(items) -> tuple[list[str], list[str]]:
    groups: list[str] = []
    names: list[str] = []
    for item in items or []:
        if isinstance(item, str):
            groups.append(item)
        elif isinstance(item, dict) and set(item) == {"name"} and isinstance(item["name"], str):
            names.append(item["name"])
        else:
            raise ValueError("Selectors must be built-in names or mappings like {name: 'REGEX'}")
    return groups, names


def _selector_clause_tokens(item, *, label: str) -> list[str]:
    if isinstance(item, str):
        return [item]
    if (
        isinstance(item, (list, tuple))
        and item
        and all(isinstance(token, str) for token in item)
    ):
        return list(item)
    raise ValueError(f"{label} values must be non-empty strings")


def _parse_local_setting_tokens(
    tokens: list[str],
    *,
    label: str,
) -> dict[str, str]:
    allowed = {
        "activation-bits",
        "strategy",
        "group-size",
        "chunk-size",
        "scale-dtype",
    }
    settings: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(
                f"{label} local setting {token!r} must use KEY=VALUE"
            )
        key, value = token.split("=", 1)
        if key not in allowed:
            raise ValueError(
                f"Unknown {label} local setting {key!r}; supported: "
                + ", ".join(sorted(allowed))
            )
        if not value:
            raise ValueError(f"{label} local setting {key!r} cannot be empty")
        if key in settings:
            raise ValueError(f"Duplicate {label} local setting {key!r}")
        settings[key] = value
    missing = sorted({"activation-bits", "strategy"} - set(settings))
    if missing:
        raise ValueError(
            f"{label} mixed rule requires explicit local settings: "
            + ", ".join(missing)
        )
    return settings


def _target_rule_from_cli_clause(
    tokens: list[str],
    *,
    label: str,
    is_name: bool,
) -> TargetSchemeRule | None:
    header = tokens[0]
    if "=" not in header:
        if len(tokens) != 1:
            raise ValueError(
                f"{label} local settings require TARGET=WEIGHT_FORMAT"
            )
        return None
    target, weight_format = header.rsplit("=", 1)
    format_prefix = weight_format.lower().rstrip("0123456789")
    format_suffix = weight_format[len(format_prefix) :]
    looks_like_weight_format = (
        format_prefix in {"int", "fp"} and format_suffix.isdigit()
    )
    if is_name and not looks_like_weight_format and len(tokens) == 1:
        return None
    if not target:
        raise ValueError(f"{label} target cannot be empty")
    if not looks_like_weight_format:
        raise ValueError(
            f"{label} weight format must be explicit, for example int4, int8, "
            "or a future fp4/fp8 format"
        )
    settings = _parse_local_setting_tokens(tokens[1:], label=label)
    try:
        activation_num_bits = int(settings["activation-bits"])
        group_size = (
            int(settings["group-size"])
            if "group-size" in settings
            else None
        )
        chunk_size = (
            int(settings["chunk-size"])
            if "chunk-size" in settings
            else None
        )
    except ValueError as exc:
        raise ValueError(
            f"{label} activation-bits, group-size, and chunk-size must be integers"
        ) from exc
    kwargs = {
        "weight_format": weight_format,
        "activation_num_bits": activation_num_bits,
        "strategy": settings["strategy"],
        "group_size": group_size,
        "chunk_size": chunk_size,
        "scale_dtype": settings.get("scale-dtype"),
    }
    if is_name:
        kwargs["name_pattern"] = target
    else:
        kwargs["selection"] = target
    return TargetSchemeRule(**kwargs)


def _parse_scheme_cli_selectors(
    groups,
    names,
) -> tuple[list[str], list[str], list[TargetSchemeRule]]:
    legacy_groups: list[str] = []
    legacy_names: list[str] = []
    rules: list[TargetSchemeRule] = []
    for item in groups or ():
        tokens = _selector_clause_tokens(item, label="--select")
        rule = _target_rule_from_cli_clause(
            tokens,
            label="--select",
            is_name=False,
        )
        if rule is None:
            legacy_groups.append(tokens[0])
        else:
            rules.append(rule)

    for item in names or ():
        tokens = _selector_clause_tokens(item, label="--select-name")
        rule = _target_rule_from_cli_clause(
            tokens,
            label="--select-name",
            is_name=True,
        )
        if rule is None:
            legacy_names.append(tokens[0])
        else:
            rules.append(rule)
    return legacy_groups, legacy_names, rules


def _parse_recipe_selections(
    items,
) -> tuple[list[str], list[str], list[TargetSchemeRule]]:
    if items is None:
        return [], [], []
    if not isinstance(items, list) or not items:
        raise ValueError("select must be a non-empty list")
    groups: list[str] = []
    names: list[str] = []
    rules: list[TargetSchemeRule] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            groups.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"select item {index} must be a string or mapping")
        if set(item) == {"name"} and isinstance(item["name"], str):
            names.append(item["name"])
            continue
        allowed = {
            "target",
            "name",
            "weight_format",
            "activation_bits",
            "strategy",
            "group_size",
            "chunk_size",
            "scale_dtype",
        }
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown select item {index} keys: {', '.join(unknown)}"
            )
        if ("target" in item) == ("name" in item):
            raise ValueError(
                f"Per-selector item {index} requires exactly one of target or name"
            )
        missing = sorted(
            {"weight_format", "activation_bits", "strategy"} - set(item)
        )
        if missing:
            raise ValueError(
                f"Per-selector item {index} requires: " + ", ".join(missing)
            )
        selection = item.get("target")
        name_pattern = item.get("name")
        if selection is not None and not isinstance(selection, str):
            raise ValueError(f"select item {index} target must be a string")
        if name_pattern is not None and not isinstance(name_pattern, str):
            raise ValueError(f"select item {index} name must be a string")
        rules.append(
            TargetSchemeRule(
                weight_format=str(item["weight_format"]),
                activation_num_bits=int(item["activation_bits"]),
                strategy=str(item["strategy"]),
                selection=selection,
                name_pattern=name_pattern,
                group_size=(
                    int(item["group_size"])
                    if item.get("group_size") is not None
                    else None
                ),
                chunk_size=(
                    int(item["chunk_size"])
                    if item.get("chunk_size") is not None
                    else None
                ),
                scale_dtype=(
                    str(item["scale_dtype"])
                    if item.get("scale_dtype") is not None
                    else None
                ),
            )
        )
    return groups, names, rules


def _parse_unselected_policy(value) -> UnselectedWeightsPolicy:
    if value is None:
        return UnselectedWeightsPolicy()
    if not isinstance(value, dict):
        raise ValueError("unselected must be a mapping")
    unknown = sorted(set(value) - {"strategy", "format"})
    if unknown:
        raise ValueError(f"Unknown unselected keys: {', '.join(unknown)}")
    strategy = value.get("strategy", "convert")
    target_format = value.get(
        "format",
        None if strategy == "preserve" else "bf16",
    )
    return UnselectedWeightsPolicy(
        strategy=strategy,
        format=target_format,
    )


def _mapping(value, name: str, allowed: set[str]) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(unknown)}")
    return value


def _build_calibration_policy(
    args,
    recipe: dict,
    official_autoround: dict | None = None,
) -> CalibrationPolicy:
    config = _mapping(
        recipe.get("calibration"),
        "calibration",
        {
            "data",
            "samples",
            "sequence_length",
            "seed",
            "split",
            "text_column",
            "trust_remote_code",
        },
    )

    def value(cli_name: str, config_name: str, default):
        cli_value = getattr(args, cli_name, None)
        if cli_value is not None:
            return cli_value
        if config_name in config:
            return config[config_name]
        official_names = {
            "data": "dataset",
            "samples": "nsamples",
            "sequence_length": "seqlen",
            "seed": "seed",
        }
        official_name = official_names.get(config_name)
        if official_name and official_autoround is not None:
            return official_autoround.get(official_name, default)
        return default

    data = value("calibration_data", "data", None)
    if isinstance(data, list):
        data = tuple(data)
    if data is not None and not isinstance(data, (str, tuple)):
        raise ValueError("calibration.data must be a path, dataset name, or list of text")
    return CalibrationPolicy(
        data=data,
        samples=int(value("calibration_samples", "samples", 128)),
        sequence_length=int(
            value("calibration_seq_length", "sequence_length", 512)
        ),
        seed=int(value("calibration_seed", "seed", 42)),
        split=str(value("calibration_split", "split", "train")),
        text_column=str(value("calibration_text_column", "text_column", "text")),
        trust_remote_code=bool(
            value("trust_remote_code", "trust_remote_code", False)
        ),
    )


def _build_gptq_policy(
    args,
    recipe: dict,
    official_autoround: dict | None = None,
) -> GPTQPolicy:
    config = _mapping(
        recipe.get("gptq"),
        "gptq",
        {
            "block_size",
            "damp_percent",
            "desc_act",
            "static_groups",
            "true_sequential",
            "symmetric",
        },
    )

    def value(cli_name: str, config_name: str, default):
        cli_value = getattr(args, cli_name, None)
        if cli_value is not None:
            return cli_value
        if config_name in config:
            return config[config_name]
        if config_name == "symmetric" and official_autoround is not None:
            return official_autoround.get("sym", default)
        return default

    return GPTQPolicy(
        block_size=int(value("gptq_block_size", "block_size", 128)),
        damp_percent=float(value("damp_percent", "damp_percent", 0.01)),
        desc_act=bool(value("desc_act", "desc_act", True)),
        static_groups=bool(value("static_groups", "static_groups", False)),
        true_sequential=bool(value("true_sequential", "true_sequential", True)),
        symmetric=bool(value("symmetric", "symmetric", True)),
    )


def _build_awq_policy(args, recipe: dict) -> AWQPolicy:
    config = _mapping(
        recipe.get("awq"),
        "awq",
        {
            "zero_point",
            "version",
            "duo_scaling",
            "apply_clip",
            "n_grid",
            "max_chunk_memory",
        },
    )

    def value(cli_name: str, config_name: str, default):
        cli_value = getattr(args, cli_name, None)
        return cli_value if cli_value is not None else config.get(config_name, default)

    return AWQPolicy(
        zero_point=bool(value("awq_zero_point", "zero_point", True)),
        version=str(value("awq_version", "version", "gemm")),
        duo_scaling=bool(value("awq_duo_scaling", "duo_scaling", True)),
        apply_clip=bool(value("awq_apply_clip", "apply_clip", True)),
        n_grid=int(value("awq_n_grid", "n_grid", 20)),
        max_chunk_memory=int(
            value("awq_max_chunk_memory", "max_chunk_memory", 1024 * 1024 * 1024)
        ),
    )


def _build_autoround_policy(
    args,
    recipe: dict,
    official_autoround: dict | None = None,
) -> AutoRoundPolicy:
    config = _mapping(
        recipe.get("autoround"),
        "autoround",
        {
            "iters",
            "lr",
            "minmax_lr",
            "batch_size",
            "gradient_accumulate_steps",
            "momentum",
            "enable_minmax_tuning",
            "enable_quantized_input",
            "official_config",
        },
    )

    def value(cli_name: str, config_name: str, default):
        cli_value = getattr(args, cli_name, None)
        if cli_value is not None:
            return cli_value
        if config_name in config:
            return config[config_name]
        if official_autoround is None:
            return default
        official_name = (
            "enable_quanted_input"
            if config_name == "enable_quantized_input"
            else config_name
        )
        return official_autoround.get(official_name, default)

    learning_rate = value("autoround_lr", "lr", None)
    minmax_learning_rate = value("autoround_minmax_lr", "minmax_lr", None)
    return AutoRoundPolicy(
        iters=int(value("autoround_iters", "iters", 200)),
        lr=float(learning_rate) if learning_rate is not None else None,
        minmax_lr=(
            float(minmax_learning_rate)
            if minmax_learning_rate is not None
            else None
        ),
        batch_size=int(value("autoround_batch_size", "batch_size", 8)),
        gradient_accumulate_steps=int(
            value(
                "autoround_gradient_accumulate_steps",
                "gradient_accumulate_steps",
                1,
            )
        ),
        momentum=float(value("autoround_momentum", "momentum", 0.0)),
        enable_minmax_tuning=bool(
            value(
                "autoround_minmax_tuning",
                "enable_minmax_tuning",
                True,
            )
        ),
        enable_quantized_input=bool(
            value(
                "autoround_quantized_input",
                "enable_quantized_input",
                True,
            )
        ),
    )


def build_quantization_policy(args) -> QuantizationPolicy:
    recipe = load_quantize_recipe(args.recipe) if args.recipe else {}
    autoround_recipe = recipe.get("autoround") or {}
    if autoround_recipe and not isinstance(autoround_recipe, dict):
        raise ValueError("autoround must be a mapping")
    official_config_source = getattr(args, "autoround_config", None)
    if official_config_source is None:
        official_config_source = autoround_recipe.get("official_config")
    official_autoround = None
    if official_config_source is not None:
        from flagos_compressor.integrations.autoround import (
            load_official_autoround_config,
        )

        official_autoround = load_official_autoround_config(
            official_config_source
        )

    recipe_groups, recipe_names, recipe_scheme_rules = _parse_recipe_selections(
        recipe.get("select")
    )
    cli_groups, cli_names, cli_scheme_rules = _parse_scheme_cli_selectors(
        getattr(args, "select", None),
        getattr(args, "select_name", None),
    )
    exclude_groups, recipe_excludes = _parse_selector_items(recipe.get("exclude"))
    selections = tuple(recipe_groups + cli_groups)
    exclude_selections = tuple(exclude_groups + list(args.exclude or ()))
    include_names = tuple(recipe_names + cli_names)
    exclude_names = tuple(recipe_excludes + list(args.exclude_name or ()))
    target_scheme_rules = tuple(recipe_scheme_rules + cli_scheme_rules)
    if target_scheme_rules and (selections or include_names):
        raise ValueError(
            "Selector-local and legacy --select/--select-name values cannot be mixed"
        )
    if not target_scheme_rules and not selections and not include_names:
        raise ValueError(
            "No weights selected; use --select/--select-name or a recipe"
        )

    if target_scheme_rules:
        incompatible = {
            "bits": getattr(args, "bits", None) is not None or "bits" in recipe,
            "activation_bits": (
                getattr(args, "activation_bits", None) is not None
                or "activation_bits" in recipe
            ),
            "scale_dtype": (
                getattr(args, "scale_dtype", None) is not None
                or "scale_dtype" in recipe
            ),
            "strategy": (
                getattr(args, "strategy", None) is not None
                or "strategy" in recipe
            ),
            "group_size": (
                getattr(args, "group_size", None) is not None
                or "group_size" in recipe
            ),
            "chunk_size": (
                getattr(args, "chunk_size", None) is not None
                or "chunk_size" in recipe
            ),
        }
        conflicts = sorted(name for name, present in incompatible.items() if present)
        if conflicts:
            raise ValueError(
                "Selector-local rules own these settings; remove global: "
                + ", ".join(conflicts)
            )
        requested_method = getattr(args, "method", None) or recipe.get("method")
        if requested_method not in (None, "mse"):
            raise ValueError("Selector-local rules currently support only --method mse")
        requested_format = getattr(args, "format", None) or recipe.get("format")
        if requested_format not in (None, "compressed-tensors"):
            raise ValueError(
                "Selector-local rules require --format compressed-tensors"
            )

    def value(name: str, default):
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            return cli_value
        if name in recipe:
            return recipe[name]
        if official_autoround is not None:
            official_name = {
                "bits": "bits",
                "group_size": "group_size",
            }.get(name)
            if official_name is not None:
                return official_autoround.get(official_name, default)
        return default

    method = value(
        "method",
        "autoround" if official_autoround is not None else "mse",
    )
    num_bits = int(value("bits", 4))
    activation_num_bits = int(value("activation_bits", 16))
    strategy = value("strategy", "group")
    requested_group_size = value("group_size", None)
    if strategy == "group" and requested_group_size is None:
        requested_group_size = (
            128
            if method in {"gptq", "awq", "autoround"}
            else (32 if num_bits == 4 else 128)
        )
    default_chunk_size = 4096 if num_bits == 4 else 1024
    return QuantizationPolicy(
        selections=selections,
        exclude_selections=exclude_selections,
        include_names=include_names,
        exclude_names=exclude_names,
        target_scheme_rules=target_scheme_rules,
        method=method,
        format=value("format", None),
        num_bits=num_bits,
        activation_num_bits=activation_num_bits,
        scale_dtype=value("scale_dtype", "float32"),
        strategy=strategy,
        group_size=(
            int(requested_group_size)
            if requested_group_size is not None
            else None
        ),
        n_candidates=int(value("n_candidates", 200)),
        chunk_size=int(value("chunk_size", default_chunk_size)),
        calibration=_build_calibration_policy(args, recipe, official_autoround),
        gptq=_build_gptq_policy(args, recipe, official_autoround),
        awq=_build_awq_policy(args, recipe),
        autoround=_build_autoround_policy(args, recipe, official_autoround),
        unselected=_parse_unselected_policy(recipe.get("unselected")),
    )


def dump_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))
