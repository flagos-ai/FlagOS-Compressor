import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from flagos_compressor.core.validation import validate_artifact


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="requires two CUDA GPUs")
def test_parallel_shards_merge_into_one_valid_artifact(tmp_path):
    source, output, work = (tmp_path / p for p in ("source", "output", "work"))
    source.mkdir()
    mapping = {}
    for layer in range(2):
        state = {}
        for proj in ("w1", "w2", "w3"):
            base = f"model.layers.{layer}.mlp.experts.0.{proj}"
            state[base + ".weight_packed"] = torch.full((2, 16), 0x21, dtype=torch.uint8)
            state[base + ".weight_scale"] = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
            state[base + ".weight_global_scale"] = torch.tensor(2.)
        shard = f"model-{layer + 1:05d}-of-00002.safetensors"
        save_file(state, str(source / shard))
        mapping.update({name: shard for name in state})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (source / "config.json").write_text("{}")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / "examples/parallel_w8a8.py"),
        "--input", str(source), "--output", str(output), "--work-dir", str(work),
        "--gpus", "0,1", "--n-candidates", "8",
    ], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    assert validation["int8_tensors"] == 6
    assert not (output / "_INCOMPLETE").exists()
    report = json.loads((output / "quantization_report.json").read_text())
    assert report["converted"] == 6
    assert report["skipped_scales"] == 12
    assert all(op["fallback"] == 0 for op in report["ops"].values())
