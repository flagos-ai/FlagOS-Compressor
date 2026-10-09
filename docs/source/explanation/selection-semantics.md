# Selection semantics

The selection model defines which checkpoint weights are transformed and which are left unselected. It is designed to make broad architecture-aware selection easy while still allowing name-based control when needed.

## Selector inputs

A selection can come from:

- built-in selectors passed with `--select`.
- regex patterns passed with `--select-name`.
- exclusions passed with `--exclude`.
- regex exclusions passed with `--exclude-name`.
- recipe `select` entries.

Selections from the CLI and recipe are merged. Explicit CLI parameters can override recipe settings for scalar quantization options.

## Union then subtraction

The selected set is built in two steps:

1. `--select` and `--select-name` add weights to the candidate set.
1. `--exclude` and `--exclude-name` remove weights from that candidate set.

This makes broad selection plus narrow exclusion possible.

## Why built-in selectors matter

Built-in selectors encode knowledge about supported model families and fused layouts. They are safer than regex when a model contains:

- MoE routed expert banks.
- shared experts.
- MLA-style attention projections.
- fused gate/up/down expert tensors.

A regex can match a name, but it does not by itself prove that the resulting set is a safe inference unit.

## Fused-unit boundary

Some tensors are consumed together by inference kernels. Splitting them can produce an output that is structurally inconsistent or not loadable.

Examples include:

- MLA-style projection pairs.
- gate/up/down projections inside expert banks.
- partial experts inside routed expert groups.

The selection model therefore treats fused groups as boundaries that should be selected together.

## Dry-run as selection review

Dry-run mode is the selection review step. It lets you check selected counts, unmatched selectors, layout constraints, and fused-unit constraints before writing weights.

A quantization command should not be considered ready until its dry-run output matches the intended selection.
