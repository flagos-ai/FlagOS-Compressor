# Troubleshoot common failures

Use this page by searching for the complete error message. Fix the input or
selection described by the matching entry, then rerun `inspect` and
`--dry-run` before writing a new output directory.

## `Missing model.safetensors.index.json`

The input path is incorrect, or the checkpoint is not an indexed sharded
`safetensors` export.

Check that the input directory is the exported model root and that the index
and all referenced shards are present together. A supported single-file
checkpoint can use `model.safetensors` instead of the index and shard set.
Rerun `inspect` after correcting the directory.

## `No supported FP8/FP4 tensors were found`

`convert` did not find recognized MXFP4 or strict 128 x 128 block FP8 tensors.
If the input is already BF16 or FP16, this result is expected; use it as the
floating-point baseline instead of running `convert`.

For a low-precision input, verify the layout with `inspect` and confirm that
the scale tensors satisfy the supported format rules. Do not bypass an
unrecognized layout check.

## `No weights selected`

No `--select` or `--select-name` option was provided, and the recipe does not
contain a `select` entry.

Add at least one built-in selector or name rule. Recipe and CLI selectors can
be combined; use the [selector reference](../reference/selectors.md) for the
available built-in groups.

## `selectors did not match any supported tensors`

The resolved selector set did not match any supported quantizable weights.
The CLI may report this as `The INT4 selectors did not match any supported
tensors` or `The INT8 selectors did not match any supported tensors`.

Run `inspect` first, check the model's actual weight names and recognized
groups, then rerun the same command with `--dry-run`. Do not infer selector
names from the model name alone.

## `unmatched quantized` is not `0`

The checkpoint contains scaled byte weights whose layout is not a recognized
MXFP4 or block FP8 format. The tool refuses to guess the format.

Do not publish the output or bypass this check. Add and validate support for
the source format first, or use a checkpoint whose low-precision layout is
recognized.

## `in_features must be divisible by group_size`

At least one selected layer's input dimension is not divisible by the
requested group size.

Use `inspect` to review the selected layer shapes, then narrow the selection
or choose a group size supported by every selected layer and the target
runtime. Do not change the group size only to make the command pass without
checking runtime support.

## `INT4 pack-quantized storage requires in_features divisible by 8`

At least one selected layer does not satisfy the input-dimension requirement
for packed INT4 storage.

Change the selection scope or use a quantization scheme whose storage
constraints fit the selected layers.

## `channel strategy cannot be used for routed MoE`

W8A16 channel quantization is not valid for routed-expert weights.

Use W8A16 group for routed experts, or use the complete W8A8 policy:

```bash
--bits 8 --activation-bits 8 --strategy channel
```

Do not use W8A16 channel for routed experts.

## `group_size must be omitted for channel strategy`

Channel strategy does not accept `--group-size`. Remove the option for both
W8A16 channel and W8A8 channel commands.

## `The selection splits fused inference units`

A selector or exclusion matched only part of an inference-fused unit.

Use a built-in selector for the model family, or select the complete fused
pair or expert bank. Gate, up, and down projections in a fused expert module
must remain together, and routed expert banks must not be partially selected.
Rerun `--dry-run` and confirm the selected counts before generation.

## A model with `self_attn.indexer` fails during attention or MoE quantization

A broad `--select attention` or `--select moe` rule may include auxiliary
indexer weights that are not supported by the generic attention quantization
path.

Exclude the indexer weights and rerun the dry-run:

```text
--exclude-name '.*\.self_attn\.indexer\..*\.weight$'
```

The exclusion changes quantization selection only. Unselected floating-point
weights remain unchanged; unselected source low-precision weights follow the
configured unselected-weight policy.

## `Fused routed-expert quantization is not supported for this model`

The model configuration has no registered fused 3D expert layout, and the
complete `gate_up_proj` / `down_proj` shapes did not provide a consistent axis
order for automatic inference.

First check that the expert banks are complete and inspect their shapes. If
the layout is still unknown, add and validate a model-specific `MoeLayout`
before quantizing. Do not apply another model's layout by assumption.

## `backend 'cuda' is unavailable`

The local PyTorch environment cannot see CUDA.

Check the NVIDIA driver, CUDA runtime, and PyTorch installation. On PPU or
Haiguang environments, use the backend name actually registered by the
installed PyTorch extension; do not infer the argument from the vendor name.
You may use `--backend cpu` for conversion or non-calibrated functional smoke
tests, but CPU execution does not establish accelerator support or
performance.

## `validate` returns `Valid`, but runtime loading fails

`validate` checks artifact structure and metadata. It does not verify the
target runtime, device kernel, graph compilation, memory capacity, or
tensor-parallel configuration.

Check all of the following in the target environment:

- the inference framework version supports the declared `compressed-tensors`,
  GPTQ, AWQ, or AutoRound format;
- the loader supports the model architecture and exact tensor names;
- the hardware has kernels for the selected W4A16, W8A16, W8A8, GPTQ, AWQ, or
  AutoRound path;
- tensor parallelism, KV cache, workspace, communication buffers, and
  framework overhead fit in memory;
- logs do not show unexpected fallback, dequantization, or graph compilation
  failure.

For compressed-tensors output, review `quantization_manifest.json` and
`quantization_report.json` when present. For native GPTQ or AutoRound, review
`quantize_config.json` and the native tensors. For native AWQ, review
`quant_config.json` and the native tensors.

## Calibration fails after generation starts

Check the calibration file shape, text column, sample count, sequence length,
model and tokenizer loading requirements, and backend availability. Prefer a
small local text, JSON, or JSONL file to separate data parsing issues from
dataset access issues.

The `datasets` extra is required only for Hugging Face dataset names.
`--trust-remote-code` applies to Transformers model and tokenizer loading,
not to Hugging Face dataset loading. Calibrated GPTQ, AWQ, and AutoRound
execution requires an available backend and does not silently fall back to
CPU.

## What to record

When reporting a failure, keep:

- the exact command line;
- the recipe file, if one was used;
- source model name and revision;
- FlagOS-Compressor version or source commit;
- backend and device;
- `inspect`, dry-run, and `validate --json` output when available;
- the relevant manifest/report or native method configuration.

The normal release path is:

```text
inspect source
  -> dry-run plan
  -> generate output
  -> validate output structure
  -> load in runtime framework
  -> run accuracy, memory, and performance tests
```

FlagOS-Compressor owns the first four steps. Runtime and quality checks must
be completed in the deployment environment.
