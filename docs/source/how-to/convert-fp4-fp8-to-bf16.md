# Convert FP4 and FP8 checkpoints to BF16

Convert source-quantized checkpoint weights to BF16 when you need a float baseline or a runtime that does not consume the original FP4 or FP8 layout.

## Prerequisites

Before conversion, prepare:

- a local HuggingFace `safetensors` checkpoint directory
- `config.json` and `model.safetensors.index.json` in the input directory
- source weights in recognized MXFP4 or strict 128×128 block FP8 layout
- a new empty output directory

```{warning}
Do not use the input directory as the output directory. Conversion writes a new checkpoint directory and must not mix with source shards.
```

## Inspect the input

Run inspection first:

```bash
flagos-compressor inspect --input "$INPUT_MODEL"
```

Confirm that the source contains recognized MXFP4 or block FP8 tensors. If the input is already BF16, FP16, or FP32, conversion may have nothing to do.

## Run a dry-run

Preview the conversion plan without writing weights:

```bash
flagos-compressor convert \
  --input "$INPUT_MODEL" \
  --output "$OUTPUT_BF16_MODEL" \
  --to bf16 \
  --backend cuda \
  --dry-run
```

Check that:

- supported low-precision tensors are recognized
- output points to a new directory
- the selected backend is available or an intentional fallback is used

## Generate BF16 output

Remove `--dry-run` to write the converted checkpoint:

```bash
flagos-compressor convert \
  --input "$INPUT_MODEL" \
  --output "$OUTPUT_BF16_MODEL" \
  --to bf16 \
  --backend cuda
```

`--to` currently accepts `bf16`.

## Validate the converted artifact

Run validation after conversion:

```bash
flagos-compressor validate --input "$OUTPUT_BF16_MODEL"
```

Expected result:

```text
Valid
```

## Result

The output directory contains BF16-compatible checkpoint shards, the shard index, model configuration, tokenizer files, and conversion metadata.

## Fallbacks

If conversion reports that no supported FP4 or FP8 tensors were found:

- verify that the input is the intended low-precision source checkpoint
- use `inspect` output to confirm detected formats
- treat already-float inputs as a float baseline instead of forcing conversion
- do not guess unknown low-precision layouts.
