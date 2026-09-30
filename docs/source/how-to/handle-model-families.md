# Handle model families

Use model-family checks to choose safe selectors, quantization schemes, and validation scope for Dense, MoE, DeepSeek-class, and special checkpoint layouts.

## Prerequisites

Before selecting a family workflow:

- run `flagos-compressor inspect --input "$MODEL"`
- save JSON inspection output for large or unfamiliar checkpoints
- identify whether the model is Dense, standard MoE, fused MoE, DeepSeek-class, or contains auxiliary modules such as `self_attn.indexer`
- run every command as a dry-run before writing weights

## Handle Dense models

Use Dense checks for standard attention plus gated MLP architectures.

Start with the conservative W8A16 group baseline:

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

Then test additional schemes only after the baseline is valid:

| Case | Selection | Quantization | Purpose |
|------|-----------|--------------|---------|
| D01 | `linear` | W8A16 group-128 | Conservative baseline |
| D02 | `linear` | W4A16 group-32 | High-compression candidate |
| D03 | `linear` | W8A8 channel/token | INT8 activation kernel validation |
| D04 | `attention` | W8A16 channel | Attention-only channel-weight test |

W8A16 channel is valid for ordinary Dense Linear and attention weights. It is not valid for routed MoE weights. Dense models should not report routed expert counts during inspection.

For D04, omit `--group-size`:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-attn-w8a16-channel" \
  --select attention \
  --bits 8 \
  --activation-bits 16 \
  --strategy channel \
  --backend cuda \
  --dry-run
```

## Handle MoE models

Use MoE selectors to keep expert structures intact.

For routed expert compression:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_MOE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-moe-routed-w4a16-g32" \
  --select moe.routed \
  --bits 4 \
  --strategy group \
  --group-size 32 \
  --backend cuda \
  --dry-run
```

For routed experts plus attention:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_MOE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-moe-routed-attn-w8a16-g128" \
  --select moe.routed \
  --select attention \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

```{important}
Do not use a manual regex to select only part of a fused expert bank. Gate, up, and down projections in a fused expert module must be selected together. Do not select only part of a routed expert bank.
```

## Handle Qwen3.5 MoE models

The implementation recognizes the Qwen3.5 MoE model types `qwen3_5_moe` and `qwen3_5_moe_text`, with model classes such as `Qwen3_5MoeForConditionalGeneration` and `Qwen3_5MoeForCausalLM`.

A supported fused expert bank has logical shape:

```text
[num_experts, out_features, in_features]
```

`gate_up_proj` is split along the output dimension. Fused W8A8 banks may be expanded to loader-facing names such as `experts.<id>.<projection>.weight` and `experts.<id>.<projection>.weight_scale`. Do not infer this layout from the model name alone. Confirm layout recognition, shape inference, and complete bank counts in `inspect` and dry-run output. Unknown or inconsistent fused layouts must be rejected rather than guessed.

## Handle DeepSeek-class models

DeepSeek-class models may include MLA-style attention names and standard or fused expert layouts. Recognized projection leaves can include:

- `wq`, `wk`, `wv`, `wo`
- `q_a_proj`, `q_b_proj`
- `kv_a_proj_with_mqa`, `kv_b_proj`
- `wq_a`, `wq_b`, `wkv_a`, `wkv_b`
- `kv_proj`, `o_b_proj`
- `in_proj_qkv`, `in_proj_qkvz`, `in_proj_ba`, `in_proj_z`, `in_proj_b`, `in_proj_a`

These projection leaves are recognized by the classifier, but model-specific
support is determined by `inspect` and the dry-run plan. Treat those outputs
as authoritative for an unfamiliar DeepSeek variant.

Use built-in selectors instead of selecting half of an MLA pair:

```bash
flagos-compressor quantize \
  --input "$MODEL_DEEPSEEK" \
  --output "$MODEL_OUTPUT_ROOT/deepseek-routed-attn-w8a16-g128" \
  --select moe.routed \
  --select attention \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

Two-dimensional routed expert banks may use names such as `experts.<id>.gate_proj`, `up_proj`, and `down_proj`, or `w1`, `w2`, and `w3`. The complete gate/up/down set and every selected expert must remain closed. Fused attention units and fused expert banks must be selected as complete inference units.

Some DeepSeek variants include `self_attn.indexer`. A broad attention selection can be narrowed with:

```bash
flagos-compressor quantize \
  --input "$INPUT_MODEL" \
  --output "$OUTPUT_MODEL" \
  --select attention \
  --select moe \
  --exclude-name '.*\\.self_attn\\.indexer\\..*\\.weight$' \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  --dry-run
```

Some DeepSeek-V4 compressor or indexer modules are classified specially and do not become ordinary linear selections. Always use dry-run output to verify the actual set.

## Validate each family case

After generation, validate each output:

```bash
flagos-compressor validate --input "$MODEL_OUTPUT_ROOT/<output-directory>"
```

Then load the output in the target runtime and run task-specific accuracy, memory, and throughput checks.

## Fallbacks

If a model family cannot be recognized safely:

- narrow the selection and rerun dry-run
- prefer built-in selectors over regex for fused structures
- do not force one model's fused MoE layout onto another model
- add and test explicit layout support before processing an unknown fused expert ABI
