# Run checkpoint quantization on multiple devices

Use `--devices` on the regular `quantize` or `convert` command. The CLI builds
one complete plan, starts a spawned process for each selected device, and
assigns source safetensors shards from a shared queue, largest first. Workers
use the same tensor executor as serial execution. The parent merges the index,
configuration, manifest, and report once all workers finish.

## Supported methods

| Operation | Parallel support | Scope |
|-----------|------------------|-------|
| MSE W8A8 | Yes | Channel-wise INT8 weights, dynamic token INT8 activations, FP32 or BF16 scales |
| MSE W8A16 | Yes | Group-wise weights; channel strategy follows the existing routed-MoE restrictions |
| MSE W4A16 | Yes | Group-wise INT4 weights |
| MSE selector-local mixtures | Yes | Existing CLI or recipe rules, exclusions, and unselected-weight policy |
| FP4/FP8 to BF16 conversion | Yes | Recognized source layouts supported by serial `convert` |
| GPTQ | No | Sequential calibration; use `--device` |
| AWQ | No | Sequential calibration; use `--device` |
| Native AutoRound | No | Sequential calibration; use `--device` |

Calibrated methods propagate activations between layers, and some methods also
depend on previously quantized projection groups or block reconstruction.
Checkpoint shard boundaries do not express those dependencies. Passing
`--devices` with a calibrated method is rejected before loading the model,
including during a dry-run.

## Run W8A8

```bash
flagos-compressor quantize \
  --input /data/model --output /data/model-w8a8 \
  --select linear --method mse --bits 8 --activation-bits 8 \
  --strategy channel --scale-dtype fp32 \
  --backend cuda --devices 0,1,2,3,4,5,6,7 \
  --dry-run
```

Remove `--dry-run` to execute. Dry-run prints the plan, source shard count,
devices, and worker count without creating output or checking device availability.
`--devices cuda:0 cuda:1` and `--devices 0 1` are also accepted with
`--backend cuda`. Device names must match the backend and must be unique;
`--device` and `--devices` are mutually exclusive. CUDA indices refer to the
devices visible to the process, including any `CUDA_VISIBLE_DEVICES` mapping.

Use the existing selectors for the desired model scope, for example
`--exclude-name '^vit\.'` to preserve a `vit` vision tower. Fused MoE layouts
are resolved from the full checkpoint before workers start, even when paired
banks occupy different shards. Source scales and other dependencies may also
reside in different shards: workers read them through the full source index.

## Run mixed quantization or conversion

```bash
flagos-compressor quantize \
  --input /data/model --output /data/model-mixed \
  --select moe=int4 activation-bits=16 strategy=group group-size=32 \
  --select attention=int8 activation-bits=8 strategy=channel \
  --backend cuda --devices 0,1,2,3

flagos-compressor convert \
  --input /data/model-fp8 --output /data/model-bf16 \
  --backend cuda --devices 0,1,2,3
```

Devices are execution settings, so the same `--devices` option also works with
existing YAML quantization recipes without changing the recipe schema.

## Resource requirements and completion

- Each process handles one shard at a time. Each selected weight and its
  quantization workspace must fit the memory requirements of the serial
  executor on one device. Host memory usage grows with concurrent shards.
- A shard or fused expert bank is not split across devices. A single-file
  checkpoint uses one worker. If there are fewer shards than devices, the CLI
  reduces the worker count and prints a warning.
- Workers use the same backend. CUDA has hardware tests; other registered
  PyTorch accelerator backends require validation in their own environments.
- Parallel execution requires available devices and disables implicit CPU
  fallback. Explicit CPU operations already specified by a format retain their
  existing placement. Inspect the report for actual operation placement.
- Use a new or empty output directory outside the source directory. The output
  contains `_INCOMPLETE` until every worker, metadata merge, and structural
  validation succeeds. `flagos-compressor validate` rejects that marker.
- A worker failure stops the remaining workers, leaves `_INCOMPLETE`, and
  records the error in `execution_error.json`. Resume is not implemented;
  diagnose the error and use a new output directory for the next run.
- Source metadata hashes and shard size/modification times are checked again
  before publishing the index. Keep the source checkpoint immutable during a run.

The usual `quantization_report.json` or `conversion_report.json` includes
`extras.parallel` with requested and active devices, worker count, and per-device
shard/action counts. `extras.validation` records the completion check. Structural
validation does not establish inference accuracy or runtime compatibility.

Speedup depends on selected tensor sizes, shard balance, host memory, and shared
storage throughput. Small checkpoints can be dominated by process startup.
Use identical quantization settings and include output validation when comparing
serial and parallel elapsed time.

## Existing example command

`examples/parallel_w8a8.py` remains a compatibility wrapper. Its existing
`--gpus`, `--include-name`, `--exclude-name`, and `--n-candidates` options now
call the formal CLI. `--work-dir` stores the resolved command and final report.
`--profile-cache` is deprecated and ignored with a warning; the formal CLI scans
current source metadata. New workflows should call `flagos-compressor` directly.
