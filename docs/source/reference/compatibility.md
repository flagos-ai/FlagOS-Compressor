# Compatibility reference

Compatibility depends on source checkpoint format, recognized model structure, quantization policy, output ABI, backend availability, and runtime loader support. A model name alone does not establish compatibility.

## Source checkpoint compatibility

The loader supports safetensors checkpoints in either single-file or indexed-sharded form. It does not accept arbitrary low-precision layouts or PyTorch `.bin` files. BF16, FP16, and FP32 weights can be selected directly. Recognized MXFP4 E2M1 with E8M0 scales and strict 128 × 128 block FP8 with E8M0 scales can be converted to BF16 or processed through the recognized path. Unknown low-precision layouts are refused. The documented validation workflow emphasizes indexed sharded checkpoints; verify single-file behavior in the target code and runtime environment before relying on it for a release workflow.

Do not use a generated compressed-tensors artifact as the recommended source for another quantization run. Start from the original checkpoint and apply one quantization policy. Keep input and output directories separate.

## Model-family compatibility

| Family | Supported structures | Selection guidance |
|--------|----------------------|--------------------|
| Dense | Standard attention, MLP, and linear layers | Start with `linear` |
| Standard MoE | Two-dimensional routed expert banks and shared experts | Use `moe.routed`, `moe.shared`, or `moe` |
| Fused MoE | Recognized three-dimensional expert layouts with complete, consistent shape pairs | Use built-in MoE selectors and keep each bank complete |
| Qwen3.5 MoE | `qwen3_5_moe` and `qwen3_5_moe_text` definitions, including the supported fused expert layout | Confirm layout and counts with inspect and dry-run |
| DeepSeek-class | MLA-style attention plus MLP or MoE variants, including DeepSeek-V2/V3/V4 calibration structures | Use attention and MoE selectors and verify projection closure |
| DSV4 fused attention | Fused attention and shared-expert runtime aliases, including `attn.wo_a` handling in per-selector mode | Verify the resolved mixed-precision plan and runtime loader mapping |
| GLM-4 MoE | Architecture-aware calibrated attention and MoE execution | Verify Transformers model loading and calibration coverage |
| Indexer variants | Auxiliary `self_attn.indexer` weights and, for some variants, compressor modules | Exclude unintended indexer weights and inspect special modules |

Fused Qwen3.5-style expert banks use the logical shape `[num_experts, out_features, in_features]`. `gate_up_proj` is split along the output dimension. Other fused three-dimensional banks must not be treated as Qwen3.5 layouts solely because of a model name. Incomplete or inconsistent shape pairs are rejected.

## Quantization compatibility

| Scheme | Constraints | Runtime requirement |
|--------|-------------|---------------------|
| BF16 | Target loader accepts BF16 or a supported float fallback | Safetensors and model-definition support |
| W4A16 | INT4 `in_features` divisible by 8 and by the group size | INT4 weight-only kernel or verified fallback |
| W8A16 group | Selected `in_features` divisible by the group size | INT8 weight-only kernel or verified fallback |
| W8A16 channel | Channel strategy is INT8 only; ordinary Linear, attention, and shared experts are supported, routed experts require group strategy | Channel-wise INT8 weight support |
| W8A8 | `bits=8`, `activation_bits=8`, `strategy=channel`, no group size | Dynamic per-token INT8 activation and weight support |
| GPTQ | Calibrated A16 group quantization and GPTQ ABI | GPTQ loader and matching packed kernels |
| AWQ | Calibrated A16, 4-bit, group quantization with zero points | AWQ GEMM loader and matching kernels |
| AutoRound | Calibrated A16 group quantization with symmetric weights and GPTQ ABI | GPTQ-compatible loader plus AutoRound provenance handling |

Method and format are coupled: MSE uses `compressed-tensors`, GPTQ uses native `gptq`, AWQ uses native `awq`, and AutoRound uses the GPTQ ABI with AutoRound provenance.

## Backend compatibility

The CLI accepts backend names exposed by the local PyTorch environment.

| Backend | Typical use |
|---------|-------------|
| `cpu` | Default execution and functional smoke tests |
| `cuda` | NVIDIA GPU execution |
| `npu` | PyTorch NPU extension environments |
| `mlu` | PyTorch MLU extension environments |
| `musa` | PyTorch MUSA extension environments |

CUDA normally resolves the default device to `cuda:0`. Other registered backend names normally resolve to `<backend>:0` unless `--device` supplies an explicit identifier. For PPU, DCU, or another vendor extension, use the actual backend name registered with PyTorch rather than guessing from the vendor name.

Conversion and non-calibrated MSE quantization may warn and perform operation-level CPU fallback when the requested accelerator is unavailable. Calibrated GPTQ, AWQ, and AutoRound execution requires an available backend and fails rather than silently falling back. Any fallback is functional evidence only, not evidence of target-device performance.

## Runtime compatibility

A target runtime must support all of the following:

1. reading the model's `quantization_config` and method-specific metadata from `config.json`
1. loading the declared compressed-tensors, GPTQ, AWQ, or AutoRound ABI
1. executing the required Linear or MoE kernel path, or a verified fallback
1. mapping the exact model architecture and layer names
1. fitting tensor parallelism, KV cache, workspace, communication buffers, graph capture, and framework overhead into available memory

`compressed-tensors` describes and stores quantized weights. It does not provide a kernel for every framework, model, or accelerator. vLLM can support compressed-tensors for particular combinations, but the exact vLLM version, model, hardware, and kernel path must be tested.

## Portability and evidence boundaries

Generated artifacts can be copied between platforms, but each target platform needs independent loading, accuracy, memory, and performance validation. `validate` proves structural and metadata consistency only. It does not prove inference loading, accuracy, memory capacity, throughput, production readiness, or kernel availability. CUDA generation success and CPU fallback do not establish deployment support on NVIDIA, PPU, DCU, or another accelerator.
