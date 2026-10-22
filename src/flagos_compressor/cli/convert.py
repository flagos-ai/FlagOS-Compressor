from __future__ import annotations

import logging

from flagos_compressor.cli.execution import execute_checkpoint, parallel_devices, print_parallel_plan
from flagos_compressor.cli.helpers import ensure_no_unmatched, print_plan
from flagos_compressor.core.planner import build_convert_plan
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors

logger = logging.getLogger(__name__)


def run(args) -> None:
    import flagos_compressor.formats.register  # noqa: F401

    devices = parallel_devices(args)
    profile = scan_hf_safetensors(args.input)
    plan = build_convert_plan(profile)
    ensure_no_unmatched(plan)
    if not plan.actions:
        raise RuntimeError("No supported FP8/FP4 tensors were found")
    if getattr(args, "dry_run", False):
        print_plan(plan)
        print_parallel_plan(args, devices)
        return
    report = execute_checkpoint(args, plan, devices)
    logger.info("Done.")
    logger.info("Converted tensors: %d", report.converted)
    logger.info("Kept tensors: %d", report.kept)
    logger.info("Skipped source scale tensors: %d", report.skipped_scales)
    logger.info("BF16 output tensors: %d", plan.output_format_counts.get("bf16", 0))
    logger.info("Report: %s/conversion_report.json", args.output)
