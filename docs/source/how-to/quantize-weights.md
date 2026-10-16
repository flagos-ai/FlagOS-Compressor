# Quantize weights

Quantize selected checkpoint weights when you need W4A16, W8A16, W8A8, GPTQ, AWQ, or AutoRound artifacts for inference-oriented testing.

## Prerequisites

Before quantization, prepare:

- a local HuggingFace `safetensors` input directory
- an empty output directory
- enough disk space for generated shards and metadata
- a selected quantization method, weight bit width, activation bit width, and output ABI
- calibration data if using GPTQ, AWQ, AutoRound, or any workflow that requires calibration

Run `inspect` before choosing selectors:

```bash
flagos-compressor inspect --input "$INPUT_MODEL"
```

## Choose a quantization family

| Family | Typical command options | Output purpose |
|--------|-------------------------|----------------|
| MSE W4A16 | `--method mse --bits 4 --activation-bits 16 --strategy group --group-size 32` | maximum weight compression candidate |
| MSE W8A16 | `--method mse --bits 8 --activation-bits 16 --strategy group --group-size 128` | conservative baseline |
| MSE W8A16 channel | `--method mse --bits 8 --activation-bits 16 --strategy channel` | ordinary Linear or attention channel-weight test |
| MSE W8A8 | `--method mse --bits 8 --activation-bits 8 --strategy channel` | INT8 weight and dynamic per-token activation testing |
| GPTQ | `--method gptq --format gptq` or inferred format | calibrated GPTQ-compatible export |
| AWQ | `--method awq --format awq` or inferred format | calibrated AWQ-compatible export |
| AutoRound | `--method autoround` | calibrated AutoRound workflow |

```{important}
For W8A8, do not pass `--group-size`. Use `--bits 8 --activation-bits 8 --strategy channel`.
```

W8A16 channel is also a valid INT8 policy for ordinary Linear and attention weights. Do not use it for routed MoE weights; use W8A16 group or the complete W8A8 policy instead. Channel strategies never accept `--group-size`.

## Run a conservative W8A16 dry-run

Use W8A16 group quantization as the first baseline:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128" \
  --select linear \
  --bits 8 \
  --activation-bits 16 \
  --strategy group \
  --group-size 128 \
  --method mse \
  --backend cuda \
  --dry-run
```

Verify before generation:

- the selected weights match the intended layer groups
- no selector is unmatched
- each selected layer satisfies the group-size constraint
- fused pairs and routed expert banks are not split
- the reported output format and quantization scheme match the intended runtime

## Generate the W8A16 artifact

Remove `--dry-run`:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128" \
  --select linear \
  --bits 8 \
  --activation-bits 16 \
  --strategy group \
  --group-size 128 \
  --method mse \
  --backend cuda
```

## Generate a W4A16 artifact

Use W4A16 when you need a maximum compression candidate:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w4a16-g32" \
  --select linear \
  --bits 4 \
  --activation-bits 16 \
  --strategy group \
  --group-size 32 \
  --method mse \
  --backend cuda
```

Focus regression testing on accuracy-sensitive workloads before treating a W4A16 artifact as deployable.

## Generate a W8A8 artifact

Use W8A8 only when the target runtime supports dynamic per-token INT8 activation handling:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a8" \
  --select linear \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --method mse \
  --backend cuda
```

For fused routed-expert banks, W8A8 expands loader-facing tensors such as
`experts.<id>.<projection>.weight` and
`experts.<id>.<projection>.weight_scale`.

Do not infer speedup from weight size alone. Runtime kernel support, graph compilation, KV cache, and framework overhead affect actual memory and throughput.

## Use a recipe

Move stable quantization settings into YAML when you need reproducibility:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_MOE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-moe-w8a8-recipe" \
  --recipe /data/recipes/w8a8-linear.yaml \
  --backend cuda
```

CLI values that are explicitly passed override recipe values. Selectors from the CLI and recipe are merged.

## Validate the output

Validate every generated artifact:

```bash
flagos-compressor validate --input "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128"
```

Expected result:

```text
Valid
```

## Fallbacks

If quantization fails:

- run `inspect` again and confirm selector names
- rerun with `--dry-run` and a new empty output directory
- reduce selection scope if one layer violates constraints
- adjust `--group-size` only after checking selected layer shapes and runtime support
- use `--backend cpu` only as a fallback, not as proof that the target accelerator runtime works
