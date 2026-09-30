# Complete a first quantization run

This tutorial walks through one complete FlagOS-Compressor path: inspect a local checkpoint, verify the quantization plan with a dry-run, generate a W8A16 artifact, and validate the output directory.

## Prerequisites

Before you begin, make sure you have:

- Python 3.10 or later
- PyTorch installed for the backend you plan to use
- FlagOS-Compressor installed. See the [installation guide](../how-to/install.md)
- a local HuggingFace `safetensors` checkpoint directory
- an empty output location with enough space for generated shards

The input checkpoint should include at least:

```text
config.json
model.safetensors.index.json
model-00001-of-000xx.safetensors
...
```

```{warning}
Do not write output into the source model directory. Use a new empty directory for each test so old shards and old configs cannot mix with new output.
```

## Set paths

Set environment variables for the source model and output root:

```bash
MODEL_QWEN_DENSE=/data/models/qwen3.5-dense
MODEL_OUTPUT_ROOT=/data/models/flagos-compressor-tests
```

Replace these paths with directories that exist in your environment.

## Inspect the source checkpoint

Run `inspect` before conversion or quantization:

```bash
flagos-compressor inspect --input "$MODEL_QWEN_DENSE"
```

Check the inspection result before continuing:

- recognized linear layer counts match the model structure
- source weight formats are recognized as BF16, FP16, FP32, MXFP4, or supported block FP8
- Dense models do not show routed expert counts
- MoE models show expected routed or shared expert groups

For a structured record, save JSON output:

```bash
flagos-compressor inspect \
  --input "$MODEL_QWEN_DENSE" \
  --json > qwen3.5-dense-inspect.json
```

## Run a dry-run

Run the same quantization command with `--dry-run` first:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128" \
  --select linear \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

Review the plan before writing weights:

- selected weight counts match expectations
- W/A bits, strategy, and group size are correct
- no selector is reported as unmatched
- no fused unit or routed expert bank is split
- `unmatched quantized` is `0` for supported source-quantized inputs

If CUDA is not available, use `--backend cpu` for a functional check.

## Generate the artifact

Remove `--dry-run` and run the command again:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128" \
  --select linear \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda
```

During generation, monitor for:

- unexpected backend fallback
- shape or group-size errors
- output file creation failures

## Validate the output

Run validation on the output directory:

```bash
flagos-compressor validate \
  --input "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128"
```

Expected result:

```text
Valid
```

Save a structured validation report when you need an audit trail:

```bash
flagos-compressor validate \
  --input "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128" \
  --json > qwen3.5-dense-w8a16-g128-validate.json
```

## Result

You now have a validated W8A16 output directory containing model config, shard index, weight shards, quantization metadata, report files, and tokenizer files copied from the source model.

## Next step

Use the [output validation guide](../how-to/validate-outputs.md) to continue with runtime loading checks and platform-specific validation.
