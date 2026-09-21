# Processing pipeline

FlagOS-Compressor processes local checkpoint directories through explicit inspection, planning, transformation, export, and validation phases.

## Source checkpoint assumptions

The input is a local [Hugging Face](https://huggingface.co/docs) `safetensors` model directory. A typical input contains:

```text
config.json
model.safetensors.index.json
model-00001-of-000xx.safetensors
...
```

The tool does not download models. The source checkpoint must already exist on local storage.

## Inspection phase

Inspection reads structural metadata before weights are converted or quantized. It identifies:

- model configuration.
- shard index entries.
- checkpoint weight names.
- tensor shapes.
- recognized source formats.
- selectable linear, attention, MLP, routed expert, and shared expert groups.

Inspection is the safest way to decide selectors and detect unfamiliar layouts.

## Planning phase

Planning combines command-line options and optional recipe values into one processing plan. The plan resolves:

- selected and excluded weights.
- quantization method.
- weight and activation bit widths.
- group or channel strategy.
- group size.
- output format.
- calibration settings when required.

Dry-run mode stops after plan validation. This makes it useful for checking selectors and constraints before writing large outputs.

## Transformation phase

The transformation phase operates on selected weights:

- conversion turns recognized MXFP4 or block FP8 tensors into BF16.
- MSE quantization creates INT4 or INT8 weight representations.
- calibrated methods use calibration data to produce GPTQ, AWQ, or AutoRound outputs.
- unselected source-quantized weights are preserved as BF16 when required by the output policy.

Unsupported or unknown low-precision layouts are refused rather than guessed.

## Export phase

Export writes a complete output model directory. Depending on the operation, this may include:

- updated `config.json`.
- `model.safetensors.index.json`.
- generated `model-*.safetensors` shards.
- `quantization_manifest.json`.
- `quantization_report.json`.
- `conversion_report.json`.
- tokenizer and supporting files copied from the source.

The complete directory is the artifact. Individual shard files are not enough for runtime loading.

## Validation phase

Validation checks consistency after export. It is designed to catch structural and metadata errors before runtime testing.

Validation is not a substitute for inference loading, accuracy evaluation, memory profiling, or throughput measurement.
