# Select layers

Select layers deliberately when you need to quantize only part of a checkpoint or exclude weights that should remain unquantized.

## Prerequisites

Before selecting layers, run inspection:

```bash
flagos-compressor inspect --input "$INPUT_MODEL"
```

Use the reported module groups and checkpoint weight names as the source of truth. Do not infer names from the model name alone.

## Choose built-in selectors first

Built-in selectors are safer than regex for recognized fused layouts and expert banks.

| Selector | Coverage |
|----------|----------|
| `linear` | all recognized attention, MLP, shared expert, and routed expert linear layers |
| `attention` | Q/K/V/O, MLA, and other recognized attention projections |
| `mlp` | non-routed-expert MLP linear layers |
| `moe` | routed experts and shared experts |
| `moe.routed` | routed experts only |
| `moe.shared` | shared experts only |

```{note}
`linear` does not include embeddings or `lm_head`.
```

## Combine selectors

Multiple `--select` and `--select-name` options form a union:

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

Expected selection: routed experts plus attention projections.

## Exclude by selector or name

Use exclusions to subtract from the selected set:

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-attn-skip-layer0" \
  --select attention \
  --exclude-name '.*\.layers\.0\..*' \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

You can also exclude a built-in group:

```bash
--select moe \
--exclude moe.shared
```

## Select by checkpoint weight name

Use `--select-name` only when built-in selectors are too broad:

```bash
flagos-compressor quantize \
  --input "$MODEL" \
  --output "$OUTPUT" \
  --select-name '.*\.self_attn\.o_proj\.weight$' \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

Regex patterns match checkpoint weight names. Use shell single quotes on POSIX shells to avoid accidental escaping.

## Avoid splitting fused units

```{danger}
Do not select only part of a fused inference unit or routed expert bank. Partial selection can produce artifacts that are structurally invalid or unusable by inference kernels.
```

Avoid splitting:

- MLA-style fused pairs such as `q_a_proj` and `kv_a_proj_with_mqa`
- fused expert bank gate/up/down groups
- partial experts inside a routed expert bank.

Use built-in selectors such as `attention`, `moe`, or `moe.routed` for fused structures.

## Handle auxiliary indexer weights

Some models include `self_attn.indexer` auxiliary modules. If these are selected unintentionally, exclude them:

```bash
--exclude-name '.*\.self_attn\.indexer\..*\.weight$'
```

Always rerun the dry-run after adding the exclusion.

## Verify the selection

A selection task is complete only after dry-run verification:

```bash
flagos-compressor quantize \
  --input "$MODEL" \
  --output "$OUTPUT" \
  --select-name '<your-pattern>' \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

Check that:

- selected counts match the model structure
- no `selectors did not match` message appears
- no fused pair or expert bank is split
- selected layers satisfy bit-width, strategy, and group-size constraints
- calibrated selection covers every required expert, and calibration data routes tokens through each selected expert
