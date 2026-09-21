# Format reference

Use this page to identify checkpoint layouts, quantization schemes, tensor storage, and shape constraints. The [compatibility reference](compatibility.md) adds model-family and runtime requirements.

## Input formats

| Format | Processing | Notes |
|--------|------------|-------|
| BF16 / FP16 / FP32 | Direct selection and quantization | Standard float checkpoint weights |
| MXFP4 E2M1 + E8M0 | Convert to BF16 or quantize through the recognized layout | Scale groups use the supported MXFP4 layout |
| Block FP8 + E8M0 | Convert to BF16 or quantize through the recognized layout | Only strict 128 × 128 block layout is recognized |
| Unknown low-precision layout | Refused | The converter does not guess an ABI |
| Existing compressed-tensors artifact | Not recommended for re-quantization | Start from the original source checkpoint |

The loader accepts a single `model.safetensors` file or an indexed sharded checkpoint with `model.safetensors.index.json` and its referenced shards. PyTorch `.bin` checkpoints are not an accepted input format. The documented validation workflow primarily covers indexed sharded checkpoints; single-file input is implemented but should still be checked in the target code and runtime environment.

A source directory normally contains:

```text
model-directory/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-000xx.safetensors
└── ...
```

For a single-file checkpoint, use `model.safetensors` instead of the index and shard set. Keep input and output directories separate. For a formal run, use a new empty output directory so stale shards, indexes, manifests, or configuration files cannot be mixed with the new result.

### MXFP4 recognition

An MXFP4 weight must be a two-dimensional byte-sized tensor with a matching scale row count. The layout must satisfy:

```text
weight_cols * 2 = scale_cols * 32
```

The logical input dimension is twice the packed byte-column count. A byte tensor with a scale tensor that does not satisfy these relationships is not accepted as MXFP4 merely because its dtype looks plausible.

### Block FP8 recognition

For a two-dimensional weight with `rows` and `cols`, the E8M0 scale tensor must contain exactly:

```text
ceil(rows / 128) * ceil(cols / 128)
```

This count includes padding tiles. Dtype alone is not sufficient to classify a tensor as block FP8.

## Output schemes

| Scheme | Weights | Activations | Strategy | Default group size |
|--------|---------|-------------|----------|--------------------|
| BF16 | BF16 | BF16 | None | Not applicable |
| W4A16 | INT4 | 16-bit | Group | 32 for MSE |
| W8A16 group | INT8 | 16-bit | Group | 128 for MSE and calibrated methods |
| W8A16 channel | INT8 | 16-bit | Output channel | Ordinary Linear, attention, and shared-expert weights |
| W8A8 | INT8 | Dynamic per-token INT8 | Output channel | Not applicable |

Quantization reduces weight-related storage only. KV cache, runtime workspace, communication buffers, graph capture, and framework overhead do not shrink proportionally with weight bit width.

## compressed-tensors storage

MSE W4A16 and W8A16 use the `compressed-tensors` `pack-quantized` scheme. Packed weights use `torch.int32` storage. The logical weight shape is recorded separately as an `int64[2]` `<weight>_shape` tensor.

For INT4, signed values are encoded as unsigned nibble codes by adding `8`. Each `int32` word stores consecutive input values from low to high order. The packed input dimension is:

```text
ceil(in_features * bits / 32)
```

The INT4 packed-input constraint also requires `in_features` to be divisible by `8`.

W8A16 uses `int32` uint8b128 packing. Each word contains four byte codes. The physical packed storage may contain more columns than the logical input dimension when padding is required. Padding uses zero code `128`. The `<weight>_shape` tensor preserves the logical two-dimensional shape so the runtime can distinguish physical storage from logical dimensions. MSE W8A16 scales are stored as BF16.

For group quantization, the scale shape is:

```text
[out_features, ceil(in_features / group_size)]
```

For channel quantization, the scale shape is:

```text
[out_features, 1]
```

MSE W8A8 uses the `compressed-tensors` `int-quantized` scheme. The weight tensor keeps its logical name and uses raw signed `torch.int8` storage. Its scale is per output channel with shape `[out_features, 1]` and uses the selected FP32 or BF16 scale dtype. The activation configuration is dynamic, symmetric, per-token INT8.

The compressed-tensors configuration in `config.json` identifies the scheme. A generated compressed-tensors artifact also contains the quantization manifest and execution report described in [output files](output-files.md). Per-selector mixed W4A16/W8A8 output uses the `mixed-precision` top-level format with `pack-quantized` and `int-quantized` groups.

## Native GPTQ storage

Native GPTQ uses the following tensor ABI:

| Tensor | Shape or dtype |
|--------|---------------|
| `qweight` | `[in_features / pack_factor, out_features]` |
| `qzeros` | `[groups, out_features / pack_factor]` |
| `scales` | `[groups, out_features]` |
| `g_idx` | `int32[in_features]` |

The pack factor is:

```text
pack_factor = 32 // bits
```

The input and output dimensions must satisfy the corresponding pack-factor divisibility constraints. AutoGPTQ zero codes use:

```text
(zero_point - 1) mod 2**bits
```

Native GPTQ and AutoRound remove the original `.weight` tensor and export the native tensors. AutoRound uses this GPTQ ABI and retains `algorithm: autoround` provenance in metadata. It does not define a separate AutoRound storage ABI.

## Native AWQ GEMM storage

Native AWQ GEMM uses:

| Tensor | Shape or dtype |
|--------|---------------|
| `qweight` | `[in_features, out_features / 8]` |
| `qzeros` | `[groups, out_features / 8]` |
| `scales` | `[groups, out_features]`, normally float16 |

The selected layer's `in_features` must be divisible by the group size, and `out_features` must be divisible by `8`. AWQ GEMM nibble order is:

```text
0, 2, 4, 6, 1, 3, 5, 7
```

This is not simple consecutive nibble order. Native AWQ exports use `quant_config.json`. Native GPTQ and AutoRound exports use `quantize_config.json`.

## Fused expert naming

Fused expert banks may be expanded into loader-facing per-expert names such as:

```text
experts.<id>.<projection>.weight
experts.<id>.<projection>.weight_scale
```

Do not infer a fused layout from names alone. Use layout recognition, shape inference, and dry-run output. A fused `gate_up_proj` must be split along the model-defined output dimension, and the complete gate, up, and down bank must remain selected as one inference unit.

## Constraints

### INT4 packing

- `in_features` must be divisible by `8` for packed INT4 storage.
- Group quantization also requires every selected layer's `in_features` to be divisible by `group_size`.

### Strategy constraints

- Channel strategy must omit `group_size`.
- Channel strategy is currently valid for INT8 policies only.
- W8A16 channel quantization is supported for ordinary Linear, attention, and shared-expert weights, but cannot be used for routed experts.
- Routed experts can use group strategy for W4A16 or W8A16.
- W8A8 channel routed-MoE quantization is allowed when the full W8A8 parameter combination is used.

### W8A8 fixed parameters

W8A8 requires exactly:

```bash
--bits 8 --activation-bits 8 --strategy channel
```

Do not pass `--group-size`. W8A8 weights are raw signed INT8 and activations are dynamic, symmetric, per-token INT8.

```{warning}
A structurally valid storage format does not guarantee a runtime kernel. Check the target loader and hardware independently.
```
