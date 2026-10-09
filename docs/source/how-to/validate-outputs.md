# Validate outputs

Validate generated artifacts before runtime loading so file structure, metadata, shard indices, and quantization declarations are checked consistently. Structural validation is one gate in a larger deployment workflow.

## Prerequisites

Before validation, make sure you have:

- a completed conversion or quantization output directory
- the full output directory, not only `.safetensors` files
- the exact command, recipe, model revision, and backend used for generation
- access to the target inference framework for runtime checks after structural validation

## Run structural validation

Validate the output directory:

```bash
flagos-compressor validate --input "$OUTPUT_MODEL"
```

Expected result:

```text
Valid
```

Save JSON output when reporting or archiving a result:

```bash
flagos-compressor validate \
  --input "$OUTPUT_MODEL" \
  --json > validate-result.json
```

## Check the output contract

Keep the entire generated directory together. The required method-specific files differ by output ABI.

Compressed-tensors output normally contains:

```text
config.json
model.safetensors.index.json or model.safetensors
model-*.safetensors when indexed
quantization_manifest.json
quantization_report.json
tokenizer files when present
```

Native GPTQ or AutoRound output normally contains `config.json`, `quantize_config.json`, native GPTQ tensors and shards, and a native index or single-file weight file when emitted. Native AWQ output normally contains `config.json`, `quant_config.json`, native AWQ tensors and shards, and a native index or single-file weight file when emitted. BF16 conversion can contain `conversion_report.json`.

```{warning}
Do not publish or copy only the `.safetensors` files. Keep configuration, index or single-file weights, method metadata, quantization metadata, and supporting tokenizer files when they are present.
```

See the [output files reference](../reference/output-files.md) for the directory-level contract.

## Understand what validation checks

Validation checks generated artifact structure and metadata, including:

- shard and index consistency
- actual shard tensors corresponding to index entries
- declared total size when present
- compressed-tensors manifest-declared weights and scales
- INT4 and INT8 storage shapes and metadata
- native GPTQ tensors such as `qweight`, `qzeros`, `scales`, and `g_idx`
- native AWQ tensors such as `qweight`, `qzeros`, and `scales`
- `config.json` quantization configuration and method metadata
- supported dynamic-token INT8 activation declarations for W8A8 output
- tokenizer and supporting files when they are preserved by the source model

Validation does not decode packed words to prove nibble or byte-code semantics. It also does not independently prove every mathematical relationship between physical storage and logical shape, and it cannot prove that an orphan manifest entry has a corresponding tensor in every possible malformed artifact.

## Understand what validation does not check

Validation does not prove:

- the target inference framework can load the artifact
- the target accelerator has W4A16, W8A16, W8A8, GPTQ, AWQ, or AutoRound kernels
- the runtime selected the intended kernel rather than a fallback
- accuracy is acceptable
- memory usage, latency, TTFT, TPOT, or throughput meets production goals
- tensor parallel settings are correct
- the model is ready for production

## Run runtime loading checks

After structural validation, load the model in the target runtime. The runtime must support:

1. Reading `config.json` and its `quantization_config` or native method metadata.
1. Loading the declared compressed-tensors, GPTQ, AWQ, or AutoRound ABI.
1. Mapping the exact model architecture and layer names.
1. Executing the required Linear or MoE kernel path, or a verified correctness fallback.
1. Fitting tensor parallelism, KV cache, workspace, communication buffers, graph capture, and framework overhead into available memory.

Example online serving command for a compatible vLLM environment:

```bash
vllm serve /data/models/flagos-compressor-tests/<MODEL_CASE> \
  --served-model-name <SERVED_MODEL_NAME> \
  --tensor-parallel-size <TP_SIZE> \
  --port <PORT>
```

Example offline smoke test:

```python
from vllm import LLM, SamplingParams

model = LLM(
    model="/data/models/flagos-compressor-tests/<MODEL_CASE>",
    tensor_parallel_size=1,
)
outputs = model.generate(
    ["Briefly introduce FlagOS."],
    SamplingParams(temperature=0.0, max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

vLLM can support compressed-tensors for particular combinations of version, model, hardware, and quantization scheme. See the [vLLM compressed-tensors integration guide](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/) and [vLLM quantization support matrix](https://docs.vllm.ai/en/stable/features/quantization/), then verify the exact target combination independently. The `compressed-tensors` package describes and stores quantized weights but does not provide a kernel for every framework or accelerator.

Check runtime logs for:

- recognition of the declared quantization method
- the expected W4A16, W8A16, W8A8, GPTQ, AWQ, or AutoRound path
- a compatible Dense or MoE kernel path
- absence of unexpected fallback or dequantization
- the actual tensor-parallel and memory settings

If vLLM or another runtime fails to load, do not conclude from that failure alone that the model files are invalid. Check the runtime version, model architecture mapping, hardware support, driver and device runtime, and kernel availability.

## Test target platforms

Use the actual backend name registered by the local PyTorch extension:

| Target | Generation or loading guidance |
|--------|--------------------------------|
| NVIDIA GPU | Use the compatible CUDA, PyTorch, driver, and runtime stack, commonly `--backend cuda --device cuda:0` |
| PPU | Use the backend and device identifiers registered by the PPU PyTorch extension |
| Haiguang DCU | Use the backend and device identifiers registered by the DCU PyTorch extension |
| Other accelerator | Use the actual registered PyTorch backend name rather than inferring it from the vendor name |

If the target extension or quantization operator is not ready, generate on CPU or CUDA and copy the artifact to the target environment for independent loading, correctness, memory, and performance testing. Conversion and non-calibrated MSE may use operation-level CPU fallback. Calibrated GPTQ, AWQ, and AutoRound execution requires an available backend and does not silently fall back to CPU.

## Run accuracy and performance tests

Compare every candidate with an unquantized or BF16 reference using the same model revision, tokenizer, prompts, runtime version, hardware, tensor-parallel configuration, and workload.

Record at least:

- task or evaluation dataset and prompt format
- exact output comparison or score
- peak memory and memory allocation settings
- time to first token, or TTFT
- time per output token, or TPOT
- prompt and generation throughput
- batch size, sequence lengths, concurrency, and tensor parallel size
- kernel and fallback messages
- failures, hangs, and long-run stability

A successful generation command is functional evidence only. It is not by itself accuracy, memory, throughput, or production evidence.

## Execute the recommended test matrix

Use a fresh output directory for every case. Run `inspect`, dry-run, generation, `validate`, runtime loading, and the applicable quality tests for each case.

| Case | Recommended model | Selection | Quantization | Purpose |
|------|-------------------|-----------|--------------|---------|
| B00 | Source FP4 or FP8 model | All recognized low-precision weights | BF16 conversion | Float compatibility and quality reference |
| D01 | Qwen3.5 Dense | `linear` | W8A16 group-128 | Conservative full-linear baseline |
| D02 | Qwen3.5 Dense | `linear` | W4A16 group-32 | High-compression candidate |
| D03 | Qwen3.5 Dense | `linear` | W8A8 channel and token | INT8 weight and activation kernel validation |
| D04 | Qwen3.5 Dense | `attention` | W8A16 channel | Attention-only channel-weight test |
| M01 | Qwen3.5 MoE | `moe.routed` | W4A16 group-32 | Routed expert compression |
| M02 | Qwen3.5 MoE | `moe.routed` and `attention` | W8A16 group-128 | Joint expert and attention quantization |
| M03 | Qwen3.5 MoE | `linear` | W8A8 channel and token | Fused expert expansion and full-linear W8A8 |
| M04 | Qwen3.5 MoE | `moe.routed` | W8A8 channel and token | Routed expert W8A8 with other weights preserved |
| S01 | DeepSeek class | `linear` | W8A16 group-128 | MLA plus MLP and MoE smoke test |
| S02 | DeepSeek MoE | `moe.routed` | W4A16 group-32 | Two-dimensional expert-bank closure |
| S03 | DeepSeek MoE | `moe.routed` and `attention` | W8A16 group-128 | MLA and expert joint quantization |
| S04 | DeepSeek class | `linear` | W8A8 channel and token | W8A8 loading exploration |
| I01 | Model with `self_attn.indexer` | `attention` and `moe`, excluding indexer | W8A8 channel and token | Keep auxiliary indexer weights out of the generic path |

The D04 policy uses W8A16 channel for ordinary attention weights and omits `--group-size`. W8A16 channel is still invalid for routed MoE; use W8A16 group there. W8A8 routed MoE is allowed when the complete W8A8 combination is used. Do not run quantization again on an already generated compressed-tensors artifact to combine cases. Start from the original source checkpoint for each case.

## Release acceptance checklist

For every candidate released to users:

- [ ] Save source model name, revision, and directory integrity information.
- [ ] Save the actual `git rev-parse HEAD` output.
- [ ] Save `inspect --json` output.
- [ ] Save the complete command line or recipe.
- [ ] Confirm `unmatched quantized: 0` in dry-run output.
- [ ] Explain any CPU fallback in the generation report.
- [ ] Confirm `validate` returns `Valid`.
- [ ] Preserve the manifest and report when the compressed-tensors path emits them.
- [ ] Complete BF16 or floating-point A/B accuracy testing in the same runtime stack.
- [ ] Complete actual loading tests on each target platform, including NVIDIA GPU, PPU, and Haiguang DCU when those targets are claimed.
- [ ] Record inference framework, PyTorch, driver, and device runtime versions.
- [ ] Record the selected kernels and any fallback path.
- [ ] Complete memory, TTFT, TPOT, throughput, and stability tests.
- [ ] Complete expert and multi-device communication tests for Qwen3.5 MoE.
- [ ] Confirm actual MLA and expert selection counts for DeepSeek-class models.
- [ ] Keep the original source and previous stable artifact for rollback.

## Write a reproducible support conclusion

Do not report only that a model or chip is supported. Record the exact evidence boundary:

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

This format distinguishes five different claims: the tool generated an artifact, the artifact passed structural validation, the runtime loaded it, the target hardware used a supported kernel path, and the measured quality and performance met the stated goal.

## Fallbacks

If structural validation passes but runtime loading fails:

1. Verify inference framework support for the declared `quantization_config` or native method metadata.
1. Check PyTorch, driver, device runtime, and framework versions.
1. Confirm the target hardware has matching kernels.
1. Check model architecture and tensor-name mapping.
1. Test smaller tensor-parallel settings and memory limits.
1. Compare runtime logs with `quantization_report.json` when that report exists.
1. Treat CPU fallback as functional evidence only. It does not establish accelerator performance or deployment compatibility.
