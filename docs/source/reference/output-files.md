# Output files

Generated file documentation.

## Output directory structure

The exact file set depends on whether the command converts to BF16, writes compressed-tensors output, or writes a native calibrated format. Copy the complete directory for every workflow.

## Compressed-tensors output

A sharded compressed-tensors output commonly contains:

```text
output-directory/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-000xx.safetensors
├── model-00002-of-000xx.safetensors
├── ...
├── quantization_manifest.json
├── quantization_report.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── ...
```

A single-file source may produce `model.safetensors` instead of the index and shard files. The compressed-tensors path emits:

- `config.json` with the `quantization_config` declaration
- the standard safetensors index and shards, or a single safetensors file
- `quantization_manifest.json`
- `quantization_report.json`
- tokenizer and other source support files when present

When per-selector policies produce mixed W4A16/W8A8 output, `config.json` uses the `mixed-precision` top-level format and declares the storage format separately for each configuration group.

`quantization_manifest.json` records quantized weights, scales, bit widths, logical shapes, storage shapes, and storage formats. `quantization_report.json` records processing counts, execution time, requested and actual backends, CPU fallback occurrences, operation events, and fallback reasons.

## Native GPTQ and AutoRound output

Native GPTQ and native AutoRound use the GPTQ tensor ABI rather than the compressed-tensors manifest contract. Their primary files are:

```text
output-directory/
├── config.json
├── quantize_config.json
├── model.safetensors.index.json
├── gptq_model-<bits>bit-<group_size>g<suffix>.safetensors
└── ...
```

Depending on shard layout and exporter behavior, the native output can instead contain standard native shard names and an index such as:

```text
gptq_model-<bits>bit-<group_size>g.index.json
```

The native tensors include `qweight`, `qzeros`, `scales`, and `g_idx`. AutoRound keeps `algorithm: autoround` provenance while using the GPTQ ABI. Native GPTQ and AutoRound should not be documented as requiring `quantization_manifest.json` or `quantization_report.json` unless a specific exporter independently emits those files.

## Native AWQ output

Native AWQ GEMM output primarily contains:

```text
output-directory/
├── config.json
├── quant_config.json
├── model.safetensors.index.json
├── native AWQ weight shards
└── ...
```

The native tensors include `qweight`, `qzeros`, and `scales`. Native AWQ uses `quant_config.json`, not `quantize_config.json`. Native AWQ should not be documented as requiring the compressed-tensors manifest or execution report.

## Core files

### config.json

Model configuration and, for supported formats, the runtime quantization declaration. A compressed-tensors example is:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized"
  }
}
```

Native GPTQ, AutoRound, and AWQ configurations use the method-specific metadata expected by the target loader.

### model.safetensors.index.json

The weight shard index. Every tensor referenced by the index must exist in one of the referenced shard files.

### model-*.safetensors

Standard model weight shards used by the indexed compressed-tensors and conversion paths when emitted with that naming convention.

### model.safetensors

A single-file safetensors weight file used instead of an index and shard set when the output is not sharded.

## Conversion-specific files

### conversion_report.json

BF16 conversion output can include `conversion_report.json`, which records conversion statistics, recognized source formats, processing time, requested and actual backends, and fallback events when reported by the conversion path.

## Publishing requirements

```{warning}
Copy the entire output directory rather than only `.safetensors` files. Preserve configuration, index or single-file weights, method metadata, quantization metadata, and supporting tokenizer files when they are present.
```

For compressed-tensors output, preserve:

```text
config.json
model.safetensors.index.json or model.safetensors
model-*.safetensors when using indexed shards
quantization_manifest.json
quantization_report.json
tokenizer files
```

For native GPTQ or AutoRound, preserve `config.json`, `quantize_config.json`, the native index or single-file weights, native shards, and tokenizer files when present. For native AWQ, preserve `config.json`, `quant_config.json`, native weights, and tokenizer files when present. For BF16 conversion, preserve `conversion_report.json` when emitted.

## Size comparison

```bash
# Compare directory sizes
du -sh \
  "$MODEL_QWEN_DENSE" \
  "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128" \
  "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w4a16-g32" \
  "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a8"
```

## File descriptions

| File | Format | Purpose |
|------|--------|---------|
| `config.json` | JSON | Model and runtime configuration |
| `model.safetensors.index.json` | JSON | Indexed shard map |
| `model.safetensors` | Safetensors | Single-file weight data |
| `model-*.safetensors` | Safetensors | Standard weight shards when emitted |
| `quantization_manifest.json` | JSON | Compressed-tensors quantization metadata |
| `quantization_report.json` | JSON | Compressed-tensors processing report |
| `conversion_report.json` | JSON | BF16 conversion report when emitted |
| `quantize_config.json` | JSON | Native GPTQ or AutoRound configuration |
| `quant_config.json` | JSON | Native AWQ configuration |
| `tokenizer.json` | JSON | Tokenizer data when present |
| `tokenizer_config.json` | JSON | Tokenizer settings when present |
| `special_tokens_map.json` | JSON | Special token settings when present |
