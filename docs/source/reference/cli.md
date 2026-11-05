# CLI reference

Use this page to look up commands, option names, defaults, and applicability rules. Run `flagos-compressor <command> --help` in the installed checkout for the parser's live help text.

## Commands

| Command | Purpose |
|---------|---------|
| `convert` | Convert recognized FP4 or FP8 checkpoint weights to BF16 |
| `quantize` | Quantize selected weights to INT4 or INT8 |
| `inspect` | Inspect checkpoint formats and selectable weight groups |
| `validate` | Validate a generated conversion or quantization artifact |

All paths are local model directories. Keep input and output directories separate. Use a new empty output directory for each generation run.

## `convert`

```bash
flagos-compressor convert [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | path | required | Source checkpoint directory |
| `--output` | path | required | Destination directory |
| `--to` | `bf16` | `bf16` | Output dtype. BF16 is currently the only accepted value |
| `--dry-run` | flag | false | Print the conversion plan without writing weights |
| `--backend` | string | `cpu` | PyTorch backend, such as `cpu`, `cuda`, `npu`, `mlu`, or `musa` |
| `--device` | string | backend default | Optional device identifier, such as `cuda:0` |

Example:

```bash
flagos-compressor convert \
  --input /data/source-low-precision \
  --output /data/output-bf16 \
  --to bf16 \
  --backend cuda \
  --device cuda:0
```

For backward compatibility, a command line beginning with legacy `--input` or another non-command option is treated as `convert`. The explicit `convert` form is preferred.

## `quantize`

```bash
flagos-compressor quantize [OPTIONS]
```

### Input, output, and selection options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | path | required | Source checkpoint directory |
| `--output` | path | required | Destination directory |
| `--recipe` | path | none | YAML recipe |
| `--select` | repeatable selector | none | Include a built-in selector |
| `--exclude` | repeatable selector | none | Exclude a built-in selector from the selected set |
| `--select-name` | repeatable regex | none | Include weights whose checkpoint names match |
| `--exclude-name` | repeatable regex | none | Exclude matching checkpoint weight names |
| `--dry-run` | flag | false | Build and print the plan without writing output |

At least one selection source is needed. Built-in selectors and name patterns are combined as a union, then exclusions are applied. See the [selector reference](selectors.md).

Per-selector mode is an alternate selection form. It uses clauses such as:

```text
--select moe=int4 activation-bits=16 strategy=group group-size=32
--select attention=int8 activation-bits=8 strategy=channel scale-dtype=fp32
```

Each clause starts with `SELECTOR=WEIGHT_FORMAT`, where the weight format is `int4` or `int8`. Selector-local options reuse CLI names without `--`: `activation-bits`, `strategy`, `group-size`, `scale-dtype`, and `chunk-size`. Local `activation-bits` and `strategy` are required. Built-in and name rules are ordered, with the last matching rule winning; global exclusions are applied afterward. Do not mix selector-local clauses with the legacy homogeneous selection form.

### Quantization options

| Option | Type | Default after policy resolution | Description |
|--------|------|-------------------------------|-------------|
| `--bits` | `4` or `8` | `4` | Weight bit width |
| `--activation-bits` | `8` or `16` | `16` | Activation bit width. `8` is the dynamic W8A8 path |
| `--strategy` | `group` or `channel` | `group` | Group quantization or output-channel quantization |
| `--group-size` | positive integer | `32` for MSE INT4, `128` for MSE INT8 and calibrated methods | Group size. Omit for channel strategy |
| `--method` | `mse`, `gptq`, `awq`, `autoround` | `mse` | Quantization algorithm |
| `--format` | `compressed-tensors`, `gptq`, `awq` | inferred from method | Output checkpoint ABI |
| `--scale-dtype` | `fp32` or `bf16` | `fp32` | W8A8 scale dtype |
| `--n-candidates` | positive integer | `200` | MSE candidate count |
| `--chunk-size` | positive integer | `4096` for INT4, `1024` for INT8 | Processing chunk size |

The method determines the default format and restricts compatible formats:

| Method | Format |
|--------|--------|
| `mse` | `compressed-tensors` |
| `gptq` | `gptq` |
| `awq` | `awq` |
| `autoround` | `gptq` |

`--activation-bits 8` requires `--bits 8 --strategy channel`. Channel strategy currently supports INT8 only, and `--group-size` must be omitted. W8A16 channel is valid for ordinary Linear, attention, and shared-expert weights, but routed-expert weights require W8A16 group or the complete W8A8 policy. GPTQ, AWQ, and AutoRound currently support weight-only A16 and require group strategy. Native AutoAWQ GEMM currently supports only 4-bit weights and requires zero points.

### Calibration options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--calibration-data` | path or dataset name | none | Local text, JSON, JSONL, or Hugging Face dataset |
| `--calibration-samples` | positive integer | `128` | Maximum calibration samples |
| `--calibration-seq-length` | positive integer | `512` | Calibration sequence length |
| `--calibration-seed` | integer | `42` | Sampling seed |
| `--calibration-split` | string | `train` | Dataset split |
| `--calibration-text-column` | string | `text` | Text field or dataset column |
| `--trust-remote-code` / `--no-trust-remote-code` | flag | false | Allow remote Transformers model or tokenizer code when required; this is not passed to dataset loading |

The `datasets` optional dependency is needed for Hugging Face dataset names. Local files do not require it. See [calibrate models](../how-to/calibrate-models.md).

### GPTQ options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gptq-block-size` | positive integer | `128` | GPTQ block size |
| `--damp-percent` | float between 0 and 1 | `0.01` | Hessian damping percentage |
| `--desc-act` / `--no-desc-act` | flag | true | Activation ordering |
| `--static-groups` / `--no-static-groups` | flag | false | Use static groups |
| `--true-sequential` / `--no-true-sequential` | flag | true | Quantize projection groups sequentially |
| `--symmetric` / `--no-symmetric` | flag | true | Use symmetric weights |

### AWQ options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--awq-version` | `gemm` | `gemm` | Native AutoAWQ checkpoint variant |
| `--awq-zero-point` / `--no-awq-zero-point` | flag | true | Use zero points. Native GEMM requires true |
| `--awq-duo-scaling` / `--no-awq-duo-scaling` | flag | true | Enable duo scaling |
| `--awq-apply-clip` / `--no-awq-apply-clip` | flag | true | Enable output-MSE clipping |
| `--awq-n-grid` | positive integer | `20` | Scaling search grid size |
| `--awq-max-chunk-memory` | positive integer | `1073741824` | Maximum AWQ chunk memory in bytes |

AWQ currently requires group strategy, A16 activations, 4-bit weights, and zero points.

### AutoRound options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--autoround-iters` | positive integer | `200` | Optimization iterations |
| `--autoround-config` | path | none | Official AutoRound JSON configuration to import |
| `--autoround-lr` | positive float | method default | Rounding learning rate |
| `--autoround-minmax-lr` | positive float | method default | Min/max tuning learning rate |
| `--autoround-batch-size` | positive integer | `8` | Optimization batch size |
| `--autoround-gradient-accumulate-steps` | positive integer | `1` | Gradient accumulation steps |
| `--autoround-momentum` | non-negative float | `0.0` | Optimizer momentum |
| `--autoround-minmax-tuning` / `--no-autoround-minmax-tuning` | flag | true | Enable min/max tuning |
| `--autoround-quantized-input` / `--no-autoround-quantized-input` | flag | true | Enable quantized-input cascading |

AutoRound uses the GPTQ tensor ABI while retaining `autoround` algorithm provenance. The optional `official-autoround` extra provides the official package integration.
When importing an official AutoRound JSON configuration, the bridge recognizes
`scheme`, `bits`, `group_size`, `iters`, `nsamples`, `seqlen`, and both
`enable_quantized_input` and the official historical spelling
`enable_quanted_input`. Explicit CLI and recipe values take precedence over
imported values.

### Backend options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--backend` | string | `cpu` | Backend name exposed by the local PyTorch environment |
| `--device` | string | backend default | Device identifier |

The built-in examples are `cpu`, `cuda`, `npu`, `mlu`, and `musa`. Other registered PyTorch device extensions may be used with their actual backend name. An unavailable accelerator can trigger CPU fallback in the non-calibrated execution path. Treat fallback as functional evidence, not target-device performance evidence.

### Quantize examples

```bash
flagos-compressor quantize \
  --input /data/model \
  --output /data/output \
  --select linear \
  --bits 8 \
  --activation-bits 16 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --device cuda:0 \
  --dry-run
```

```bash
flagos-compressor quantize \
  --input /data/model \
  --output /data/output \
  --select linear \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --scale-dtype bf16 \
  --backend cuda
```

## `inspect`

```bash
flagos-compressor inspect --input /data/model
flagos-compressor inspect --input /data/model --json > inspect-result.json
```

`inspect` reports recognized source formats, tensor groups, shapes, and selection information. Use its JSON output when archiving an audit or investigating an unfamiliar model.

## `validate`

```bash
flagos-compressor validate --input /data/output
flagos-compressor validate --input /data/output --json > validate-result.json
```

`validate` checks artifact structure and metadata. It does not prove inference loading, accuracy, memory, throughput, or production readiness. See [output validation](../how-to/validate-outputs.md).
