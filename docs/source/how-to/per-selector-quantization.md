# Use per-selector quantization

Per-selector mode assigns a complete weight and activation scheme to each selected part of one checkpoint. It is useful when, for example, DSV4 attention should use W8A8 while MoE weights use W4A16 in the same execution.

## Run a mixed quantization command

```bash
flagos-compressor quantize \
  --input /path/to/DSV4-Flash \
  --output /path/to/DSV4-Flash-per-selector \
  --select moe=int4 activation-bits=16 strategy=group group-size=32 \
  --select attention=int8 activation-bits=8 strategy=channel scale-dtype=fp32 \
  --backend cuda
```

Each selector-local clause starts with `SELECTOR=WEIGHT_FORMAT`. The supported weight formats are `int4` and `int8`; `fp4` and `fp8` are reserved extension points.

Selector-local settings reuse existing CLI option names without the leading `--`:

- `activation-bits`
- `strategy`
- `group-size`
- `scale-dtype`
- `chunk-size`

`activation-bits` and `strategy` are required in every local clause. Groupwise INT4 and INT8 use default group sizes of 32 and 128 when `group-size` is omitted. Channel strategy rejects `group-size`.

## Choose selectors

Selectors may be built-in groups:

- `attention`
- `moe`
- `moe.routed`
- `moe.shared`
- `mlp`
- `linear`

Name rules continue to use `--select-name`:

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/output \
  --select-name 'model\.layers\.0\..*=int8' activation-bits=16 strategy=group group-size=128 \
  --backend cuda
```

Name rules are applied after built-in target rules. The last matching rule wins. Global `--exclude` and `--exclude-name` rules are applied after local selector rules.

## Respect mode boundaries

The legacy homogeneous form remains valid:

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-int4 \
  --select moe \
  --bits 4
```

Do not mix selector-local clauses with legacy homogeneous selectors in one command. This prevents a selector from silently inheriting the wrong format.

Per-selector mode supports MSE W4A16, W8A16, and W8A8. GPTQ, AWQ, and AutoRound support calibrated groupwise W4A16 and W8A16. Do not combine selector-local settings with global `--bits`, `--activation-bits`, `--strategy`, `--group-size`, `--scale-dtype`, or `--chunk-size` settings for the same command. Global `--n-candidates`, `--exclude`, `--exclude-name`, the unselected-weight policy, and backend/device options remain available.

The mapping is an artifact contract, not a runtime compatibility guarantee. For example, a DSV4 source FP8 attention weight matching `attention=int8` is exported as INT8 W8A8 storage, including `attn.wo_a`; it is not silently preserved as FP8 for a DeepGEMM implementation.

## Calibrate mixed INT4 and INT8 weights

Use the same selectors for all three calibration methods. For example, this
command assigns attention W8A16 and MoE W4A16:

```bash
flagos-compressor quantize \
  --input /path/to/DeepSeek-V4-Flash \
  --output /path/to/DeepSeek-V4-Flash-awq-mixed \
  --method awq \
  --select attention=int8 activation-bits=16 strategy=group group-size=128 \
  --select moe=int4 activation-bits=16 strategy=group group-size=128 \
  --calibration-data /data/calibration/prompts.jsonl \
  --calibration-samples 256 --calibration-seq-length 2048 \
  --calibration-seed 42 --calibration-unobserved-policy error \
  --no-desc-act --symmetric --no-awq-zero-point \
  --backend cuda --device cuda:0
```

Replace `--method awq` with `gptq` or `autoround` to use that method for both
bit widths. Append `--select moe.shared=int8 activation-bits=16 strategy=group
group-size=128` to make shared experts W8A16 while routed experts remain W4A16.
Later rules override earlier ones.

The same configurations are available as
[`calibrated-mixed-attention8-moe4.yaml`](../../../examples/recipes/calibrated-mixed-attention8-moe4.yaml)
and [`calibrated-mixed-attention8-routed4-shared8.yaml`](../../../examples/recipes/calibrated-mixed-attention8-routed4-shared8.yaml).
Set `--method gptq`, `--method awq`, or `--method autoround` when using either
recipe, and replace its calibration-data path with your prepared corpus.

GPTQ uses the resolved bit width for each matrix's Hessian-based quantization.
AWQ scale and clipping searches use each participating matrix's resolved
scheme, including mixed W4/W8 matrices in one scaling relationship. AutoRound
optimizes the decoder output with independently configured W4/W8 wrappers.

These calibrated mixed checkpoints use GPTQ-compatible packed tensors and
`quant_method: gptq` for loading. The `algorithm` field records the calibration
method for AWQ and AutoRound; GPTQ uses its canonical `quant_method`. AWQ does
not run GPTQ calibration. Legacy homogeneous AWQ retains its AutoAWQ GEMM
format. In AWQ's GPTQ-compatible mixed format, W8 is symmetric;
W4 uses the requested AWQ zero-point setting
(asymmetric by default). Per-module `sym` metadata records this distinction.
`--symmetric` configures GPTQ and AutoRound; it does not override AWQ's
`--awq-zero-point` setting.

Use `--no-awq-zero-point`, as in the command above, when the target loader
requires symmetric W4 as well as W8. The runtime must support the exported
per-module overrides, fused module names, and packed kernels. Successful
calibration and structural validation do not establish runtime compatibility.

Both halves of a fused attention projection and all projections in a routed
expert bank must agree on bit width and group size. Unsupported combinations
fail before calibration. Shared experts may use a different scheme from the
routed bank. Calibrated workflows materialize source FP4/FP8 as BF16;
unselected modules remain floating point, and `unselected: preserve` is not
supported for these methods.

Use `--calibration-unobserved-policy error` to require activation coverage.
With `rtn`, unobserved experts use RTN at their requested bit width; the
calibration report and checkpoint configuration record each fallback, and
`algorithm` gains a `+rtn` suffix.

Mixed calibration writes `calibration_plan.json` with resolved per-module
schemes and retained modules. `calibration_report.json` records activation
coverage and explicit fallbacks, including failures. AWQ also retains its
separate `awq_scaling_report.json` for checking floating-point scaling before
rounding.

## Understand the output

MSE writes one compressed-tensors checkpoint with a configuration group for every requested scheme. A mixed W4A16/W8A8 result uses the `mixed-precision` top-level format and declares `pack-quantized` or `int-quantized` on each group. Calibrated methods write native GPTQ-compatible tensors, dynamic bit-width overrides, and the exact `flagos_module_quantization` mapping.

DSV4 fused-attention and shared-expert runtime aliases are included in the generated mapping. Source-quantized weights that match no rule follow the existing `unselected` policy and are converted to BF16 by default.

Use `--dry-run` to inspect the resolved plan without writing the output checkpoint.

## Fused MoE layout inference

When a fused MoE model has no registered layout adapter, the CLI can infer the 3D bank order from a consistent `gate_up_proj` and `down_proj` pair:

```text
[E, 2I, H] plus [E, H, I] -> [E, out, in]
[E, H, 2I] plus [E, I, H] -> [E, in, out]
```

Every discovered pair must be complete, valid, and agree on one axis order. If the pairs disagree or are incomplete, quantization stops instead of guessing.

## Validate the result

After dry-run review, generate the checkpoint and run:

```bash
flagos-compressor validate --input /path/to/DSV4-Flash-per-selector
```

Then test the exact artifact in the target runtime. Artifact mapping does not establish that the runtime has a compatible kernel.
