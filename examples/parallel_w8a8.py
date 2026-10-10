"""Quantize independent checkpoint shards on multiple GPUs using the core API.

Source weights and their auxiliary tensors must live in the same shard.
Each shard gets an isolated input view and output directory. The final index
and metadata are published only after every worker has completed successfully.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import torch
import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.cli.helpers import ensure_no_unmatched
from flagos_compressor.core.executor import (
    _patch_compressed_tensors_config, _write_quantization_manifest, execute_plan,
)
from flagos_compressor.core.plan import ExecutionPlan
from flagos_compressor.core.planner import build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy
from flagos_compressor.core.profile import ModelProfile
from flagos_compressor.core.report import ConversionReport, OpEvent
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


def initialize(profile_path, policy_dict, gpu_queue):
    global PROFILE, PLAN, DEVICE
    torch.set_num_threads(2)
    PROFILE = ModelProfile.from_dict(json.loads(Path(profile_path).read_text()))
    PLAN = build_quantize_plan(PROFILE, QuantizationPolicy(**policy_dict))
    DEVICE = gpu_queue.get()
    torch.cuda.set_device(DEVICE)


def process_shard(shard, source, output, work):
    started = time.time()
    source, output, work = Path(source), Path(output), Path(work)
    view = work / "views" / shard.removesuffix(".safetensors")
    part = output / ".parts" / shard.removesuffix(".safetensors")
    view.mkdir(parents=True)
    (view / shard).symlink_to(source / shard)
    shutil.copy2(source / "config.json", view / "config.json")
    mapping = {n: t.shard for n, t in PROFILE.tensors.items() if t.shard == shard}
    (view / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    plan = ExecutionPlan(metadata=PLAN.metadata.copy())
    for action in PLAN.actions:
        if action.tensor.shard == shard:
            plan.add_action(action.tensor, input_format=action.input_format,
                            output_format=action.output_format, rule_name=action.rule_name)
    plan.kept_tensors = [t for t in PLAN.kept_tensors if t.shard == shard]
    backend = build_backend("cuda", f"cuda:{DEVICE}", fallback_policy="error")
    # Some checkpoints have shards containing only unquantized parameters.
    if not plan.actions:
        plan.metadata["artifact_kind"] = "bf16"
    report = execute_plan(view, part, plan, backend)
    index = json.loads((part / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {shard}
    destination = output / shard
    if destination.exists():
        raise FileExistsError(destination)
    os.rename(part / shard, destination)
    result = {"shard": shard, "gpu": DEVICE, "elapsed_s": time.time()-started,
              "index": index, "report": report.to_dict()}
    (work / (shard + ".result.json")).write_text(json.dumps(result))
    shutil.rmtree(view)
    shutil.rmtree(part)
    print(json.dumps({k: result[k] for k in ("shard", "gpu", "elapsed_s")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--profile-cache")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--include-name", action="append", default=[])
    parser.add_argument("--exclude-name", action="append", default=[])
    parser.add_argument("--n-candidates", type=int, default=200)
    args = parser.parse_args()
    source, output, work = map(Path, (args.input, args.output, args.work_dir))
    source = source.resolve()
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    (output / "_INCOMPLETE").write_text("Quantization is in progress. See " + str(work))
    (work / "driver.pid").write_text(str(os.getpid()))
    index_hash = hashlib.sha256((source / "model.safetensors.index.json").read_bytes()).hexdigest()
    profile = (ModelProfile.from_dict(json.loads(Path(args.profile_cache).read_text()))
               if args.profile_cache else scan_hf_safetensors(source))
    assert Path(profile.model_path).resolve() == source
    checkpoint = HfSafetensorsCheckpoint(source)
    assert {n: t.shard for n, t in profile.tensors.items()} == checkpoint.weight_map
    policy = QuantizationPolicy(
        selections=("linear",), include_names=tuple(args.include_name),
        exclude_names=tuple(args.exclude_name), num_bits=8, activation_num_bits=8,
        strategy="channel", n_candidates=args.n_candidates,
    )
    plan = build_quantize_plan(profile, policy)
    ensure_no_unmatched(plan)
    for action in plan.actions:
        for aux in (action.tensor.scale_name, *action.tensor.auxiliary_names):
            if aux and checkpoint.weight_map[aux] != action.tensor.shard:
                raise ValueError(f"Cross-shard auxiliary tensor: {aux}")
    (work / "profile.json").write_text(json.dumps(profile.to_dict()))
    (work / "policy.json").write_text(json.dumps(asdict(policy), indent=2))
    print("PLAN", json.dumps(plan.summary()), flush=True)
    # Largest shards first so workers finish at roughly the same time.
    shards = sorted(checkpoint.shard_files(), key=lambda s: (source / s).stat().st_size, reverse=True)
    gpus = [int(s) for s in args.gpus.split(",")]
    context = mp.get_context("spawn")
    queue = context.Queue()
    for gpu in gpus:
        queue.put(gpu)
    policy_dict = {"selections": policy.selections, "include_names": policy.include_names,
                   "exclude_names": policy.exclude_names, "num_bits": 8,
                   "activation_num_bits": 8, "strategy": "channel",
                   "n_candidates": policy.n_candidates}
    combined = ConversionReport(backend="cuda")
    mapping, size = {}, 0
    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=context,
                             initializer=initialize,
                             initargs=(str(work / "profile.json"), policy_dict, queue)) as pool:
        futures = [pool.submit(process_shard, s, str(source), str(output), str(work)) for s in shards]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            assert not (mapping.keys() & result["index"]["weight_map"].keys())
            mapping.update(result["index"]["weight_map"])
            size += result["index"]["metadata"]["total_size"]
            r = result["report"]
            combined.converted += r["converted"]
            combined.kept += r["kept"]
            combined.skipped_scales += r["skipped_scales"]
            combined.op_events.extend(OpEvent(**event) for event in r["op_events"])
            print(f"PROGRESS {count}/{len(shards)}", flush=True)
    assert hashlib.sha256((source / "model.safetensors.index.json").read_bytes()).hexdigest() == index_hash
    checkpoint.copy_auxiliary_files(output)
    checkpoint.write_index(output, mapping, total_size=size)
    config = _patch_compressed_tensors_config(output, plan)
    _write_quantization_manifest(output, plan, config)
    combined.output_tensors = len(mapping)
    combined.extras = {"source_index_sha256": index_hash, "gpus": gpus,
                       "input_formats": plan.input_format_counts,
                       "output_formats": plan.output_format_counts,
                       "total_size": size}
    combined.finish()
    combined.save(output / "quantization_report.json")
    shutil.rmtree(output / ".parts", ignore_errors=False)
    (output / "_INCOMPLETE").unlink()
    print("COMPLETE", json.dumps({"output": str(output), "bytes": size,
                                  "converted": combined.converted,
                                  "elapsed_s": combined.finished_at-combined.started_at}), flush=True)


if __name__ == "__main__":
    main()
