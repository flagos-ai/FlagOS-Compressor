"""Compatibility entry point; use flagos-compressor quantize --devices instead."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import warnings

from flagos_compressor.cli.main import main as compressor_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--profile-cache", help="Deprecated; source metadata is scanned by the CLI")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--include-name", action="append", default=[])
    parser.add_argument("--exclude-name", action="append", default=[])
    parser.add_argument("--n-candidates", type=int, default=200)
    args = parser.parse_args()
    if args.profile_cache:
        warnings.warn("--profile-cache is deprecated and ignored; the CLI scans current source metadata.",
                      FutureWarning, stacklevel=2)
    command = ["quantize", "--input", args.input, "--output", args.output,
               "--select", "linear", "--bits", "8", "--activation-bits", "8",
               "--strategy", "channel", "--n-candidates", str(args.n_candidates),
               "--backend", "cuda", "--devices", args.gpus]
    for pattern in args.include_name:
        command.extend(["--select-name", pattern])
    for pattern in args.exclude_name:
        command.extend(["--exclude-name", pattern])
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=False)
    (work / "command.json").write_text(json.dumps(command, indent=2))
    compressor_main(command)
    shutil.copy2(Path(args.output) / "quantization_report.json", work / "quantization_report.json")


if __name__ == "__main__":
    main()
