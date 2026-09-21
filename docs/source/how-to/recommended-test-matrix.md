# Run the recommended test matrix

This guide preserves the complete model and deployment workflow. Use a fresh output directory for every case and start every quantization run from the original source checkpoint.

## Common setup

Record the source model name, revision, directory integrity information, FlagOS-Compressor commit, backend, device, command or recipe, inspection JSON, dry-run output, and validation JSON.

```bash
MODEL_QWEN_DENSE=/data/models/qwen3.5-dense
MODEL_QWEN_MOE=/data/models/qwen3.5-moe
MODEL_DEEPSEEK=/data/models/deepseek
MODEL_OUTPUT_ROOT=/data/models/flagos-compressor-tests

git rev-parse HEAD
flagos-compressor inspect --input "$MODEL_QWEN_DENSE" --json > qwen3.5-dense-inspect.json
```

The source directory must be separate from the output directory. Prefer a new empty output directory for each case so old shards, indexes, manifests, or configuration files cannot contaminate a result.

## Case overview

| Case | Model | Selection | Policy | Purpose | Priority |
|------|-------|-----------|--------|---------|----------|
| B00 | Source FP4 or FP8 | All recognized low-precision weights | BF16 conversion | Float compatibility and quality reference | Required when applicable |
| D01 | Qwen3.5 Dense | `linear` | W8A16 group-128 | Conservative baseline | Required |
| D02 | Qwen3.5 Dense | `linear` | W4A16 group-32 | High compression candidate | Required |
| D03 | Qwen3.5 Dense | `linear` | W8A8 channel/token | INT8 weight and activation kernels | Required |
| D04 | Qwen3.5 Dense | `attention` | W8A16 channel | Ordinary attention channel-weight test | Optional |
| M01 | Qwen3.5 MoE | `moe.routed` | W4A16 group-32 | Routed expert compression | Required |
| M02 | Qwen3.5 MoE | `moe.routed` and `attention` | W8A16 group-128 | Joint expert and attention quantization | Required |
| M03 | Qwen3.5 MoE | `linear` | W8A8 channel/token | Fused expert expansion and full-linear W8A8 | Required |
| M04 | Qwen3.5 MoE | `moe.routed` | W8A8 channel/token | Routed expert W8A8 | Optional |
| S01 | DeepSeek class | `linear` | W8A16 group-128 | MLA, MLP, and MoE smoke test | Required |
| S02 | DeepSeek MoE | `moe.routed` | W4A16 group-32 | Two-dimensional expert-bank closure | Required |
| S03 | DeepSeek MoE | `moe.routed` and `attention` | W8A16 group-128 | MLA and expert joint quantization | Required |
| S04 | DeepSeek class | `linear` | W8A8 channel/token | W8A8 loading exploration | Optional |
| I01 | Model with `self_attn.indexer` | `attention` and `moe`, excluding indexer | W8A8 channel/token | Keep auxiliary indexer weights out of the generic path | Required when applicable |

For every case, run `inspect`, dry-run, generation, `validate`, runtime loading, and the applicable accuracy, memory, performance, and stability tests.

## B00: convert FP4 or FP8 to BF16

Run this case only when inspection identifies supported MXFP4 or block FP8 tensors:

```bash
flagos-compressor convert \
  --input /data/models/source-low-precision \
  --output "$MODEL_OUTPUT_ROOT/source-bf16" \
  --backend cuda \
  --dry-run

flagos-compressor convert \
  --input /data/models/source-low-precision \
  --output "$MODEL_OUTPUT_ROOT/source-bf16" \
  --backend cuda

flagos-compressor validate \
  --input "$MODEL_OUTPUT_ROOT/source-bf16"
```

If the source is BF16, FP16, or FP32, `No supported FP8/FP4 tensors were found` is expected and the source can be used directly as the float reference.

## D01-D04: Qwen3.5 Dense

Run a dry-run first, then repeat the same command without `--dry-run`.

### D01: W8A16 group-128

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128" \
  --select linear \
  --bits 8 \
  --activation-bits 16 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

### D02: W4A16 group-32

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w4a16-g32" \
  --select linear \
  --bits 4 \
  --activation-bits 16 \
  --strategy group \
  --group-size 32 \
  --backend cuda \
  --dry-run
```

### D03: W8A8 channel/token

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_DENSE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a8" \
  --select linear \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  --dry-run
```

Do not pass `--group-size` for W8A8.

### D04: attention-only W8A16 channel

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

W8A16 channel is valid for ordinary Linear and attention weights. It is not valid for routed MoE weights.

## M01-M04: Qwen3.5 MoE

Qwen3.5-MoE layouts are recognized for `qwen3_5_moe`, `qwen3_5_moe_text`, `Qwen3_5MoeForConditionalGeneration`, and `Qwen3_5MoeForCausalLM`. A supported fused bank has logical shape `[num_experts, out_features, in_features]`, with `gate_up_proj` split along the output dimension.

### M01: routed experts W4A16

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_MOE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-moe-routed-w4a16-g32" \
  --select moe.routed \
  --bits 4 \
  --activation-bits 16 \
  --strategy group \
  --group-size 32 \
  --backend cuda \
  --dry-run
```

### M02: routed experts and attention W8A16

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_MOE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-moe-routed-attn-w8a16-g128" \
  --select moe.routed \
  --select attention \
  --bits 8 \
  --activation-bits 16 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

### M03: all recognized linear weights W8A8

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_MOE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-moe-w8a8" \
  --select linear \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  --dry-run
```

### M04: routed experts W8A8

```bash
flagos-compressor quantize \
  --input "$MODEL_QWEN_MOE" \
  --output "$MODEL_OUTPUT_ROOT/qwen3.5-moe-routed-w8a8" \
  --select moe.routed \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  --dry-run
```

Fused expert gate, up, and down projections must be selected as a complete unit. Unselected ordinary float weights remain unchanged; unselected source low-precision weights are converted to BF16 under the default policy.

## S01-S04: DeepSeek-class models

DeepSeek-class models may contain MLA projections such as `wq`, `wk`, `wv`, `wo`, `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `wq_a`, `wq_b`, `wkv_a`, and `wkv_b`. Use built-in selectors to preserve fused projection closure.

### S01: all recognized linear weights W8A16

```bash
flagos-compressor quantize \
  --input "$MODEL_DEEPSEEK" \
  --output "$MODEL_OUTPUT_ROOT/deepseek-w8a16-g128" \
  --select linear \
  --bits 8 \
  --activation-bits 16 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

### S02: routed experts W4A16

```bash
flagos-compressor quantize \
  --input "$MODEL_DEEPSEEK" \
  --output "$MODEL_OUTPUT_ROOT/deepseek-routed-w4a16-g32" \
  --select moe.routed \
  --bits 4 \
  --activation-bits 16 \
  --strategy group \
  --group-size 32 \
  --backend cuda \
  --dry-run
```

### S03: routed experts and attention W8A16

```bash
flagos-compressor quantize \
  --input "$MODEL_DEEPSEEK" \
  --output "$MODEL_OUTPUT_ROOT/deepseek-routed-attn-w8a16-g128" \
  --select moe.routed \
  --select attention \
  --bits 8 \
  --activation-bits 16 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

### S04: all recognized linear weights W8A8

```bash
flagos-compressor quantize \
  --input "$MODEL_DEEPSEEK" \
  --output "$MODEL_OUTPUT_ROOT/deepseek-w8a8" \
  --select linear \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  --dry-run
```

Only mark S04 as supported for a specific model variant after inspection, dry-run, validation, runtime loading, and accuracy testing all pass.

## I01: exclude auxiliary indexer weights

Check for indexer weights:

```bash
rg 'self_attn\.indexer' \
  "${INPUT_MODEL}/model.safetensors.index.json"
```

For models containing these weights, exclude them from broad attention and MoE selection:

```bash
flagos-compressor quantize \
  --input "${INPUT_MODEL}" \
  --output "${OUTPUT_MODEL}" \
  --select attention \
  --select moe \
  --exclude-name '.*\.self_attn\.indexer\..*\.weight$' \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  --dry-run
```

After the plan is verified, the same command can be run in the background:

```bash
nohup flagos-compressor quantize \
  --input "${INPUT_MODEL}" \
  --output "${OUTPUT_MODEL}" \
  --select attention \
  --select moe \
  --exclude-name '.*\.self_attn\.indexer\..*\.weight$' \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  > quantize-indexer-w8a8.log 2>&1 &

tail -f quantize-indexer-w8a8.log
```

The exclusion affects quantization selection only. Unselected float indexer weights remain float; unselected source low-precision indexer weights follow the unselected-weight policy.

## YAML recipes

The MSE recipes remain supported:

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

W8A8 recipes must omit `group_size`. CLI scalar values override recipe values; CLI and recipe selectors are merged. Archive the recipe with the source revision and command.

## Runtime and platform testing

BF16 conversion outputs are ordinary Hugging Face safetensors directories. Quantized outputs require the target runtime to support the declared `compressed-tensors` schema, model mapping, Linear and MoE kernels, and hardware execution path.

Test each claimed target independently:

| Target | Guidance |
|--------|----------|
| NVIDIA GPU | Use the matching CUDA, PyTorch, driver, and runtime stack, commonly `--backend cuda --device cuda:0` |
| PPU | Use the backend and device identifiers registered by the PPU PyTorch extension |
| Haiguang DCU | Use the backend and device identifiers registered by the DCU PyTorch extension |

An artifact can be generated on CPU or CUDA and copied to another platform, but every target still needs independent loading, correctness, memory, and performance tests. Generation success and CPU fallback do not prove deployment support.

## Release acceptance

- [ ] Source model name, revision, and directory integrity recorded.
- [ ] Actual `git rev-parse HEAD` recorded.
- [ ] `inspect --json` saved.
- [ ] Complete command or recipe saved.
- [ ] Dry-run reports `unmatched quantized: 0`.
- [ ] CPU fallback is explained in the report.
- [ ] `validate` returns `Valid`.
- [ ] Manifest and report are preserved when emitted.
- [ ] BF16 or float reference A/B accuracy test completed.
- [ ] Loading tested on every claimed target platform.
- [ ] Inference framework, PyTorch, driver, and device runtime versions recorded.
- [ ] Kernel and fallback path recorded.
- [ ] Memory, TTFT, TPOT, throughput, and stability measured.
- [ ] Qwen3.5-MoE expert and multi-device communication tests completed.
- [ ] DeepSeek MLA and expert selection counts confirmed.
- [ ] Original source and previous stable artifact retained for rollback.

## Reproducible support conclusion

```text
On FlagOS-Compressor <commit>, model <name@revision> was processed with case <ID>
and produced a <W4A16/W8A16/W8A8> compressed-tensors artifact. Inspect, dry-run,
and validate passed. The artifact loaded on <NVIDIA GPU/PPU/Haiguang DCU model>
with <card count>, <inference framework version>, and <PyTorch and device runtime
versions>. On <dataset>, the accuracy change from the BF16 reference was <result>.
Peak memory changed by <result>. TTFT, TPOT, and throughput were <results>.
Kernel fallback was <none or description>. This conclusion applies only to the
specified model revision, recipe, workload, and software and hardware versions.
```

This wording distinguishes artifact generation, structural validity, runtime loading, kernel support, and measured quality or performance.
