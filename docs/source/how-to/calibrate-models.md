# Calibrate models

Use calibration data for GPTQ, AWQ, AutoRound, or another method that needs representative activations. Calibrated execution uses the original Transformers model definition and requires `transformers>=5,<6` plus an available selected backend. Unlike the non-calibrated path, it does not silently use CPU fallback when a requested accelerator is unavailable.

## Prerequisites

Before calibration, prepare:

- a tokenizer-compatible source model directory
- a local `.txt`, `.text`, JSON, or JSONL calibration file, or a Hugging Face dataset name
- an appropriate text column for structured data
- the optional `datasets` extra for a Hugging Face dataset name
- a sample count and sequence length that fit available memory
- an available backend and device
- `--trust-remote-code` only when the Transformers model or tokenizer requires it and you have reviewed that requirement

The `--trust-remote-code` flag applies to model and tokenizer loading. It does not pass a trust setting to Hugging Face dataset loading.

The optional `datasets` dependency is loaded only for Hugging Face dataset names. Local files do not require that extra. Local text files contain one non-empty sample per line. JSONL accepts a string per line or an object with a text field. JSON accepts a top-level list, `{"data": [...]}`, or `{"text": [...]}` (and an object whose selected text column contains the records). A list or tuple of text strings is also accepted by the policy interface. Empty and over-length examples are skipped.

Ready-made calibration recipes are available under `examples/recipes/`. Use
them as starting points, then review the model selectors, backend, and
calibration inputs for the target checkpoint.

The default calibration policy is:

| Setting | Default |
|---------|---------|
| Maximum valid samples | `128` |
| Sequence length | `512` |
| Seed | `42` |
| Dataset split | `train` |
| Text column | `text` |
| Trust remote code | `false` |

## Prepare representative data

Calibration packs tokenized text into fixed-size 512-token blocks by default. The packer does not split a single source sample across unrelated examples. Select data that represents the target workload. For MoE models, the data should route tokens through the selected experts. The implementation routes selected tokens to experts, but it does not reject a run merely because a corpus failed to hit every selected expert.

Record the source, split, number of valid samples, sequence length, seed, model revision, and backend for every run. See the [recipe reference](../reference/recipes.md) for repeatable configuration.

## Run a local-file calibration

```bash
flagos-compressor quantize \
  --input "$MODEL" \
  --output "$OUTPUT" \
  --select linear \
  --bits 4 \
  --strategy group \
  --group-size 32 \
  --method gptq \
  --calibration-data /data/calibration/prompts.jsonl \
  --calibration-text-column text \
  --calibration-samples 128 \
  --calibration-seq-length 512 \
  --calibration-seed 42 \
  --backend cuda \
  --device cuda:0 \
  --dry-run
```

The calibration dry-run prints the plan and does not write weights. It does not replace a completed calibration or runtime test.

## Run a dataset calibration

```bash
flagos-compressor quantize \
  --input "$MODEL" \
  --output "$OUTPUT" \
  --select linear \
  --bits 4 \
  --strategy group \
  --group-size 32 \
  --method awq \
  --calibration-data wikitext \
  --calibration-split train \
  --calibration-text-column text \
  --calibration-samples 128 \
  --calibration-seq-length 512 \
  --backend cuda \
  --device cuda:0 \
  --dry-run
```

Use `--trust-remote-code` only for model loading when the model or tokenizer requires custom code. Review the code before enabling it.

## Run calibrated quantization

After the dry-run is correct, remove `--dry-run`:

```bash
flagos-compressor quantize \
  --input "$MODEL" \
  --output "$OUTPUT" \
  --select linear \
  --bits 4 \
  --strategy group \
  --group-size 32 \
  --method gptq \
  --calibration-data /data/calibration/prompts.jsonl \
  --calibration-text-column text \
  --calibration-samples 128 \
  --calibration-seq-length 512 \
  --calibration-seed 42 \
  --backend cuda \
  --device cuda:0
```

Source FP4 or FP8 weights are staged as BF16 before Transformers loads the calibration model. In Transformers-v5 scenarios, fused experts may be temporarily exposed as per-expert `nn.Linear` modules for calibration.

## Control sampling

Use these options to make calibration reproducible:

| Option | Purpose |
|--------|---------|
| `--calibration-samples` | Maximum number of valid samples |
| `--calibration-seq-length` | Maximum token sequence length |
| `--calibration-seed` | Sampling seed |
| `--calibration-split` | Dataset split |
| `--calibration-text-column` | Text field for structured local data or dataset rows |

## Validate the result

Run structural validation first:

```bash
flagos-compressor validate --input "$OUTPUT"
```

Native GPTQ, AWQ, and AutoRound exports are checked through their native configuration and tensors. They do not use the compressed-tensors manifest/report contract in the same way. Then run runtime loading and task-level quality checks in the target inference framework.

## Fallbacks

If calibration fails:

- reduce `--calibration-samples` or `--calibration-seq-length` to fit memory
- verify that `--calibration-text-column` exists in structured data
- use a small local JSONL file to separate data issues from dataset access issues
- confirm that the selected backend is available instead of expecting calibrated CPU fallback
- check whether the model or tokenizer requires `--trust-remote-code`
- archive the calibration command before rerunning with changed inputs
