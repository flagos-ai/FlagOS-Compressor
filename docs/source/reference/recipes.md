# Recipe reference

Recipes are YAML files for repeatable quantization settings. They keep method, bit width, strategy, selectors, calibration inputs, and output behavior together for review and reruns.

Version 1 recipes also support per-selector MSE policies. Use a per-selector entry when different parts of one checkpoint need different weight formats or activation policies.

## Recipe versions

`version: 1` is the original checkpoint-only MSE schema. `version: 2` is the extended schema for calibrated methods, nested calibration settings, method-specific mappings, and output-format settings. If `version` is omitted, the loader uses version 1. Version 1 rejects version-2-only keys such as `format`, `calibration`, `gptq`, `awq`, and `autoround`.

Use version 1 for checkpoint-only MSE recipes. Use version 2 for GPTQ, AWQ, AutoRound, nested calibration, or explicit output ABI settings.

```yaml
version: 2
bits: 4
activation_bits: 16
scale_dtype: fp32
strategy: group
method: gptq
format: gptq
group_size: 128
n_candidates: 200
chunk_size: 4096
calibration:
  data: /data/calibration/prompts.jsonl
  samples: 128
  sequence_length: 512
  seed: 42
  split: train
  text_column: text
  trust_remote_code: false
gptq:
  block_size: 128
  damp_percent: 0.01
  desc_act: true
  static_groups: false
  true_sequential: true
  symmetric: true
select:
  - linear
exclude:
  - moe.shared
unselected:
  strategy: convert
  format: bf16
```

## Common fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | integer | required by schema | Recipe schema version |
| `bits` | integer | `4` | Weight bit width, either `4` or `8` |
| `activation_bits` | integer | `16` | Activation bit width, either `8` or `16` |
| `scale_dtype` | string | `fp32` | W8A8 scale dtype, `fp32` or `bf16` |
| `strategy` | string | `group` | `group` or `channel` |
| `method` | string | `mse` | `mse`, `gptq`, `awq`, or `autoround` |
| `format` | string | inferred | `compressed-tensors`, `gptq`, or `awq`, inferred from method when omitted |
| `group_size` | integer | method and bit dependent | Required for group quantization and omitted for channel strategy |
| `n_candidates` | positive integer | `200` | MSE candidate count |
| `chunk_size` | positive integer | `4096` for INT4, `1024` for INT8 | Processing chunk size |
| `select` | list | none | Built-in selectors or exact name mappings to include |
| `exclude` | list | none | Built-in selectors or exact name mappings to exclude |
| `unselected.strategy` | string | `convert` | Handling for source-quantized weights outside the selected set |
| `unselected.format` | string | `bf16` | Format used with `unselected.strategy: convert` |

The policy layer derives group size as `32` for MSE INT4, `128` for MSE INT8, and `128` for GPTQ, AWQ, or AutoRound when group strategy is used. It derives chunk size as `4096` for INT4 and `1024` for INT8. W8A8 requires `bits: 8`, `activation_bits: 8`, and `strategy: channel`, with no `group_size`.

The method and format must agree:

| Method | Required format |
|--------|-----------------|
| `mse` | `compressed-tensors` |
| `gptq` | `gptq` |
| `awq` | `awq` |
| `autoround` | `gptq` |

## Selector fields

Recipe selectors accept either a built-in selector string or an exact mapping with one `name` key:

```yaml
select:
  - linear
  - name: .*\.self_attn\.o_proj\.weight$
exclude:
  - name: .*\.self_attn\.indexer\..*\.weight$
```

The mapping form is the recipe equivalent of the CLI `--select-name` and `--exclude-name` options. The CLI `--select` and `--exclude` options accept only built-in selector names. Recipe and CLI selectors are merged as a union before exclusions are applied. Regex patterns match checkpoint weight names.

Do not use a regex to split a fused inference unit. A fused attention pair or fused expert bank must be selected as a complete unit. A routed expert bank must include every selected expert and its complete gate, up, and down projection set.

## Per-selector fields

Per-selector entries use exactly one `target` built-in selector or `name` regex, plus an explicit `weight_format`, `activation_bits`, and `strategy`:

```yaml
version: 1
select:
  - target: moe
    weight_format: int4
    activation_bits: 16
    strategy: group
    group_size: 32
  - target: attention
    weight_format: int8
    activation_bits: 8
    strategy: channel
    scale_dtype: fp32
  - name: '^model\.layers\.0\.'
    weight_format: int8
    activation_bits: 16
    strategy: group
    group_size: 64
```

Per-selector entries may use `group_size`, `chunk_size`, and W8A8 `scale_dtype`. The supported `weight_format` values are `int4` and `int8`. Rules are evaluated in order and the last matching rule wins. CLI rules follow recipe rules, and CLI `--select-name` rules follow CLI `--select` rules. Global exclusions are applied after local rules.

Per-selector mode supports MSE W4A16, W8A16, and W8A8. Do not combine per-selector entries with the top-level global `bits`, `activation_bits`, `scale_dtype`, `strategy`, `group_size`, or `chunk_size` settings for the same policy. The legacy homogeneous recipe form remains available.

## Calibration mapping

Version 2 nests these fields under `calibration`:

| Field | Description |
|-------|-------------|
| `calibration.data` | Local `.txt`, `.text`, JSON, or JSONL path, or a Hugging Face dataset name |
| `calibration.samples` | Maximum number of valid samples, default `128` |
| `calibration.sequence_length` | Sequence length, default `512` |
| `calibration.seed` | Sampling seed, default `42` |
| `calibration.split` | Dataset split, default `train` |
| `calibration.text_column` | Text field or dataset column, default `text` |
| `calibration.trust_remote_code` | Whether remote code may be loaded for Transformers model and tokenizer loading, default `false` |

The corresponding CLI options are `--calibration-data`, `--calibration-samples`, `--calibration-seq-length`, `--calibration-seed`, `--calibration-split`, `--calibration-text-column`, and `--trust-remote-code`. See [calibration](../how-to/calibrate-models.md) for accepted file shapes and routing coverage requirements.

## Method mappings

Version 2 can contain these mappings:

```yaml
gptq:
  block_size: 128
  damp_percent: 0.01
  desc_act: true
  static_groups: false
  true_sequential: true
  symmetric: true
awq:
  version: gemm
  zero_point: true
  duo_scaling: true
  apply_clip: true
  n_grid: 20
  max_chunk_memory: 1073741824
autoround:
  iters: 200
  lr: null
  minmax_lr: null
  batch_size: 8
  gradient_accumulate_steps: 1
  momentum: 0.0
  enable_minmax_tuning: true
  enable_quantized_input: true
  official_config: /data/autoround/config.json
```

`official_config` imports an official AutoRound JSON configuration at lower priority. The bridge recognizes `scheme`, `bits`, `group_size`, `iters`, `nsamples`, `seqlen`, and both `enable_quantized_input` and the historical `enable_quanted_input` spelling. Explicit CLI values override recipe values. Recipe values override imported values. Built-in policy defaults fill remaining fields. The output retains AutoRound algorithm provenance even though its tensor ABI is GPTQ-compatible and identifies `flagos-compressor` as the provider. Ready-made recipes are available under `examples/recipes/`.

## Unselected weights

The currently executable unselected-weight policy is:

```yaml
unselected:
  strategy: convert
  format: bf16
```

This policy converts unselected source MXFP4 or block FP8 weights to BF16. Unselected ordinary floating-point weights remain unchanged.

Do not use this as a current workflow:

```yaml
unselected:
  strategy: preserve
```

The policy parser accepts `preserve` as a policy shape, but the planner rejects it because preserving a source low-precision format requires a compatible runtime configuration and inference kernel. `preserve` is therefore not an executable output policy in the current implementation. `convert` requires `format: bf16`.

## MSE recipes

### W4A16 linear recipe

```yaml
version: 1
bits: 4
activation_bits: 16
strategy: group
method: mse
group_size: 32
n_candidates: 200
chunk_size: 4096
select:
  - linear
unselected:
  strategy: convert
  format: bf16
```

### W8A16 routed-expert and attention recipe

```yaml
version: 1
bits: 8
activation_bits: 16
strategy: group
method: mse
group_size: 128
n_candidates: 200
chunk_size: 1024
select:
  - moe.routed
  - attention
unselected:
  strategy: convert
  format: bf16
```

### W8A8 linear recipe

Do not include `group_size` for this channel strategy recipe.

```yaml
version: 1
bits: 8
activation_bits: 8
strategy: channel
method: mse
n_candidates: 200
chunk_size: 1024
select:
  - linear
unselected:
  strategy: convert
  format: bf16
```

## Precedence and selection

When a recipe is used with CLI options:

1. Explicit CLI scalar values override recipe scalar values.
1. Recipe scalar values override imported official AutoRound configuration values.
1. Remaining fields use policy defaults.
1. CLI and recipe selectors are merged as a union.
1. Built-in and name exclusions subtract from the selected set.

Use [dry-run](../how-to/quantize-weights.md) to inspect the resolved policy and selected tensor counts before writing output.

## Reproducibility

Archive the recipe, complete command line, source model revision, FlagOS-Compressor version or commit, backend and device, inspection JSON, dry-run output, and validation JSON together.
