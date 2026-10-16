from __future__ import annotations

import argparse
import logging
import sys

from flagos_compressor.core.policy import BUILTIN_SELECTIONS


COMMANDS = {"convert", "quantize", "inspect", "validate"}


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="cpu",
        help="PyTorch device backend, for example cpu, cuda, npu, mlu, or musa.",
    )
    parser.add_argument("--device")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flagos-compressor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert FP4/FP8 checkpoint weights to BF16.")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--to", default="bf16", choices=["bf16"])
    convert.add_argument("--dry-run", action="store_true")
    _add_backend_args(convert)

    quantize = subparsers.add_parser(
        "quantize",
        help="Quantize selected weights to INT4 or INT8.",
    )
    quantize.add_argument("--input", required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--recipe")
    quantize.add_argument(
        "--select",
        action="append",
        nargs="+",
        metavar=("TARGET[=WEIGHT_FORMAT]", "KEY=VALUE"),
        help=(
            "Select a built-in target. For mixed quantization, append local "
            "settings using existing option names, for example: --select "
            "moe=int4 activation-bits=16 strategy=group group-size=32."
        ),
    )
    quantize.add_argument("--exclude", action="append", choices=sorted(BUILTIN_SELECTIONS))
    quantize.add_argument(
        "--select-name",
        action="append",
        nargs="+",
        metavar=("REGEX[=WEIGHT_FORMAT]", "KEY=VALUE"),
    )
    quantize.add_argument("--exclude-name", action="append", metavar="REGEX")
    quantize.add_argument("--bits", type=int, choices=[4, 8], default=None)
    quantize.add_argument(
        "--activation-bits",
        type=int,
        choices=[8, 16],
        default=None,
        help=(
            "Activation bit width. Use 8 with --bits 8 --strategy channel "
            "for dynamic per-token W8A8; defaults to 16."
        ),
    )
    quantize.add_argument(
        "--scale-dtype",
        choices=["fp32", "bf16"],
        default=None,
        help="W8A8 weight scale dtype; defaults to fp32.",
    )
    quantize.add_argument(
        "--strategy",
        choices=["group", "channel"],
        default=None,
        help="Weight granularity: groupwise or per-channel.",
    )
    quantize.add_argument(
        "--method",
        choices=["mse", "gptq", "awq", "autoround"],
        default=None,
    )
    quantize.add_argument(
        "--format",
        choices=["compressed-tensors", "gptq", "awq"],
        default=None,
        help="Checkpoint ABI; inferred from method when omitted.",
    )
    quantize.add_argument("--group-size", type=int, default=None)
    quantize.add_argument("--n-candidates", type=int, default=None)
    quantize.add_argument("--chunk-size", type=int, default=None)
    quantize.add_argument(
        "--calibration-data",
        help="Local txt/json/jsonl calibration file or Hugging Face dataset name.",
    )
    quantize.add_argument("--calibration-samples", type=int)
    quantize.add_argument("--calibration-seq-length", type=int)
    quantize.add_argument("--calibration-seed", type=int)
    quantize.add_argument("--calibration-split")
    quantize.add_argument("--calibration-text-column")
    quantize.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument("--gptq-block-size", type=int)
    quantize.add_argument("--damp-percent", type=float)
    quantize.add_argument(
        "--desc-act",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument(
        "--static-groups",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument(
        "--true-sequential",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument("--awq-version", choices=["gemm"])
    quantize.add_argument(
        "--awq-zero-point",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument(
        "--awq-duo-scaling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument(
        "--awq-apply-clip",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument("--awq-n-grid", type=int)
    quantize.add_argument("--awq-max-chunk-memory", type=int)
    quantize.add_argument("--autoround-iters", type=int)
    quantize.add_argument(
        "--autoround-config",
        help="Official AutoRound config.json or quantization config to import.",
    )
    quantize.add_argument("--autoround-lr", type=float)
    quantize.add_argument("--autoround-minmax-lr", type=float)
    quantize.add_argument("--autoround-batch-size", type=int)
    quantize.add_argument("--autoround-gradient-accumulate-steps", type=int)
    quantize.add_argument("--autoround-momentum", type=float)
    quantize.add_argument(
        "--autoround-minmax-tuning",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument(
        "--autoround-quantized-input",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    quantize.add_argument("--dry-run", action="store_true")
    _add_backend_args(quantize)

    inspect = subparsers.add_parser("inspect", help="Inspect formats and selectable weight groups.")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--json", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate a converted or quantized artifact.")
    validate.add_argument("--input", required=True)
    validate.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Backwards compatibility: the old CLI accepted --input/--output without
    # an explicit command and always performed BF16 conversion.
    if raw and raw[0] not in COMMANDS and raw[0] not in {"-h", "--help"}:
        raw.insert(0, "convert")
    parser = build_parser()
    args = parser.parse_args(raw)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # Keep operational CLI messages visible by default without enabling noisy
    # INFO logs from third-party dependencies.
    logging.getLogger("flagos_compressor").setLevel(logging.INFO)

    if args.command == "convert":
        from flagos_compressor.cli import convert
        convert.run(args)
    elif args.command == "quantize":
        from flagos_compressor.cli import quantize
        quantize.run(args)
    elif args.command == "inspect":
        from flagos_compressor.cli import inspect_model
        inspect_model.run(args)
    else:
        from flagos_compressor.cli import validate
        validate.run(args)


if __name__ == "__main__":
    main()
