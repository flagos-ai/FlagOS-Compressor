from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
from contextlib import contextmanager

from flagos_compressor.backends.registry import build_backend
from flagos_compressor.cli.helpers import build_quantization_policy, ensure_no_unmatched, print_plan
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.moe_layout import select_moe_layout
from flagos_compressor.core.planner import build_quantize_plan
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors

logger = logging.getLogger(__name__)


@contextmanager
def _calibration_source(model_path: str, backend):
    """Materialize source FP4/FP8 weights as BF16 before Transformers loading."""
    from flagos_compressor.core.planner import build_convert_plan

    profile = scan_hf_safetensors(model_path)
    needs_conversion = any(
        tensor.role == "weight"
        and tensor.storage_format in {"fp4_e2m1_e8m0", "fp8_block_e8m0"}
        for tensor in profile.tensors.values()
    )
    if not needs_conversion:
        yield model_path
        return
    with tempfile.TemporaryDirectory(prefix="flagos-calibration-bf16-") as directory:
        plan = build_convert_plan(profile)
        ensure_no_unmatched(plan)
        logger.info("Staging source FP4/FP8 weights as BF16 for model calibration")
        execute_plan(model_path, directory, plan, backend)
        yield directory


def _run_calibrated(args, policy) -> None:
    from flagos_compressor.calibration.data import build_calibration_batches
    from flagos_compressor.calibration.modeling import (
        capture_first_layer_inputs,
        decoder_layers,
        load_transformers_model,
    )
    from flagos_compressor.calibration.runner import quantize_model_sequential
    from flagos_compressor.formats.native_quantized import save_native_quantized_model

    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        raise RuntimeError(
            f"{backend.name.upper()} calibration requested but the backend is unavailable"
        )
    device = backend.device
    with _calibration_source(args.input, backend) as model_source:
        model, tokenizer = load_transformers_model(
            model_source,
            trust_remote_code=policy.calibration.trust_remote_code,
        )
        layers = decoder_layers(model)
        batches = build_calibration_batches(tokenizer, policy.calibration)
        samples = capture_first_layer_inputs(
            model,
            layers[0][1],
            batches,
            device=device,
        )
        quantized = quantize_model_sequential(
            layers,
            samples,
            policy,
            device=device,
        )
        autoround_config = None
        if policy.method == "autoround":
            from flagos_compressor.integrations.autoround import (
                official_autoround_export_config,
            )

            autoround_config = official_autoround_export_config(policy)
        save_native_quantized_model(
            model_source,
            args.output,
            model,
            quantized,
            method=policy.method,
            bits=policy.num_bits,
            group_size=int(policy.group_size or -1),
            desc_act=policy.gptq.desc_act,
            damp_percent=policy.gptq.damp_percent,
            true_sequential=policy.gptq.true_sequential,
            static_groups=policy.gptq.static_groups,
            symmetric=policy.gptq.symmetric,
            awq_zero_point=policy.awq.zero_point,
            awq_version=policy.awq.version,
            autoround_config=autoround_config,
        )
    logger.info(
        "Done. %s-quantized Linear modules: %d",
        policy.method.upper(),
        len(quantized),
    )
    logger.info("Checkpoint format: %s", policy.format)


def _selected_fused_expert(profile, policy) -> bool:
    """Whether the policy selects any fused 3D routed-expert bank."""
    return any(
        tensor.module_kind == "moe_routed_fused" and policy.selects(tensor)
        for tensor in profile.tensors.values()
    )


def _load_moe_layout(model_path: str, profile, policy):
    """Select a fused-expert layout only when fused experts are quantized."""
    if not _selected_fused_expert(profile, policy):
        return None
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "Quantizing fused routed experts requires config.json to select a "
            f"layout adapter, but {config_path} is missing"
        )
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    fused_banks = (
        (tensor.name, tensor.effective_logical_shape)
        for tensor in profile.tensors.values()
        if tensor.module_kind == "moe_routed_fused"
    )
    layout = select_moe_layout(config, fused_banks)
    logger.info("Using fused routed-expert layout: %s", layout.name)
    return layout


def run(args) -> None:
    import flagos_compressor.formats.register  # noqa: F401
    import flagos_compressor.quantizers.register  # noqa: F401

    policy = build_quantization_policy(args)
    if policy.method in {"gptq", "awq", "autoround"}:
        if args.dry_run:
            profile = scan_hf_safetensors(args.input)
            selected = [
                tensor
                for tensor in profile.tensors.values()
                if policy.selects(tensor) and len(tensor.effective_logical_shape) == 2
            ]
            fused = [
                tensor
                for tensor in profile.tensors.values()
                if policy.selects(tensor) and len(tensor.effective_logical_shape) == 3
            ]
            print(f"Calibration quantization: {policy.method} -> {policy.format}")
            print(f"  selected 2D weights: {len(selected)}")
            print(f"  selected fused expert banks: {len(fused)}")
            print(f"  calibration samples: {policy.calibration.samples}")
            print(f"  calibration sequence length: {policy.calibration.sequence_length}")
            return
        _run_calibrated(args, policy)
        return
    profile = scan_hf_safetensors(args.input)
    moe_layout = _load_moe_layout(args.input, profile, policy)
    plan = build_quantize_plan(profile, policy, moe_layout)
    ensure_no_unmatched(plan)
    if policy.target_scheme_rules:
        quantized_counts = {
            name: count
            for name, count in plan.output_format_counts.items()
            if name.startswith("compressed_tensors_")
        }
        if not quantized_counts:
            raise RuntimeError(
                "The per-selector schemes did not match any supported tensors"
            )
        if args.dry_run:
            print_plan(plan)
            return
        backend = build_backend(args.backend, args.device)
        if not backend.is_available():
            logger.warning(
                "backend %r is unavailable; CPU fallback may be used.",
                backend.name,
            )
        report = execute_plan(args.input, args.output, plan, backend)
        logger.info("Done.")
        for name, count in sorted(quantized_counts.items()):
            logger.info("Selected %s tensors/banks: %d", name, count)
        logger.info("Kept tensors: %d", report.kept)
        logger.info("Manifest: %s/quantization_manifest.json", args.output)
        logger.info("Report: %s/quantization_report.json", args.output)
        return
    num_bits = policy.num_bits
    linear_format = (
        "compressed_tensors_w8a8_channelwise"
        if policy.is_w8a8
        else (
            "compressed_tensors_int8_channelwise"
            if policy.strategy == "channel"
            else f"compressed_tensors_int{num_bits}_groupwise"
        )
    )
    quantized_count = plan.output_format_counts.get(
        linear_format,
        0,
    )
    fused_moe_count = plan.output_format_counts.get(
        (
            "compressed_tensors_w8a8_channelwise_moe_fused"
            if policy.is_w8a8
            else f"compressed_tensors_int{num_bits}_moe_fused"
        ),
        0,
    )
    if quantized_count == 0 and fused_moe_count == 0:
        raise RuntimeError(
            f"The INT{num_bits} selectors did not match any supported tensors"
        )
    if args.dry_run:
        print_plan(plan)
        return

    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        logger.warning("backend %r is unavailable; CPU fallback may be used.", backend.name)
    report = execute_plan(args.input, args.output, plan, backend)
    logger.info("Done.")
    logger.info("Strategy: %s", policy.strategy)
    if policy.is_w8a8:
        logger.info("W8A8 scale dtype: %s", policy.scale_dtype)
    logger.info(
        "W%dA%d tensors: %d",
        num_bits,
        policy.activation_num_bits,
        quantized_count,
    )
    logger.info(
        "W%dA%d fused MoE banks: %d",
        num_bits,
        policy.activation_num_bits,
        fused_moe_count,
    )
    logger.info("Converted tensors: %d", report.converted)
    logger.info("Kept tensors: %d", report.kept)
    logger.info("Manifest: %s/quantization_manifest.json", args.output)
    logger.info("Report: %s/quantization_report.json", args.output)
