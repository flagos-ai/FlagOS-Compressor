# Selector reference

Layer selector documentation.

## Built-in selectors

| Selector | Coverage |
|----------|----------|
| `linear` | All recognized attention, MLP, shared expert, and routed expert linear layers |
| `attention` | Q/K/V/O, MLA, and other recognized attention projections |
| `mlp` | Non-routed-expert MLP linear layers |
| `moe` | Routed experts + shared experts |
| `moe.routed` | Routed experts only |
| `moe.shared` | Shared experts only |

```{note}
The `linear` selector does not include embeddings or `lm_head`.
```

## Selector usage

### Single selector

```bash
--select linear
```

### Multiple selectors

```bash
--select moe.routed \
--select attention
```

### Selector exclusion

```bash
--select moe \
--exclude moe.shared
```

## Name-based selection

### Select by pattern

```bash
--select-name '.*\.self_attn\.q_proj\.weight$'
```

### Exclude by pattern

```bash
--exclude-name '.*\.layers\.0\..*'
```

## Selection semantics

### Union operation

```bash
--select attention
--select mlp
# Result: attention + mlp
```

### Subtraction operation

```bash
--select linear
--exclude-name '.*\.layers\.0\..*'
# Result: all linear layers except layer 0
```

### Priority order

1. Select by `--select` and `--select-name`
1. Remove by `--exclude` and `--exclude-name`

## Pattern examples

### Attention projections

```bash
# All attention projections
--select attention

# Only query projections
--select-name '.*\.self_attn\.q_proj\.weight$'

# All except first layer attention
--select attention
--exclude-name '.*\.layers\.0\.self_attn\..*'
```

### MLP layers

```bash
# All MLP
--select mlp

# Only gate projections
--select-name '.*\.mlp\.gate_proj\.weight$'
```

### MoE experts

```bash
# All routed experts
--select moe.routed

# All experts (routed + shared)
--select moe
```

## Constraints

### Fused structures

- Do not select partial fused pairs.
- Do not select partial expert banks.
- Use built-in selectors for fused structures.

### Dry-run verification

Always verify with dry-run:

```bash
flagos-compressor quantize \
  --input "$MODEL" \
  --output "$OUTPUT" \
  --select <your-selector> \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

Verify the following:

- Match counts are correct.
- No `selectors did not match` warnings appear.