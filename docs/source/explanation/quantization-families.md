# Quantization families

FlagOS-Compressor separates the quantization method, weight and activation bit widths, strategy, and output ABI. Select these as one policy and validate the resulting artifact in the target runtime.

## Output schemes

| Scheme | Weights | Activations | Valid strategy | Typical use |
|--------|---------|-------------|----------------|-------------|
| BF16 | BF16 | BF16 | None | Float baseline after conversion |
| W4A16 | INT4 | 16-bit | Group | High weight compression candidate |
| W8A16 group | INT8 | 16-bit | Group | Conservative weight-only baseline |
| W8A16 channel | INT8 | 16-bit | Output channel | Ordinary Linear and attention-only tests |
| W8A8 | INT8 | Dynamic per-token INT8 | Channel | Weight and activation INT8 testing |

Weight compression does not reduce KV cache, workspace, communication buffers, graph capture, or framework overhead proportionally.

## MSE quantization

MSE is the native non-calibrated INT4 and INT8 path. It can produce W4A16 and W8A16 `compressed-tensors` `pack-quantized` output, or W8A8 `compressed-tensors` `int-quantized` output. Group quantization requires each selected layer's `in_features` to be divisible by the group size. Channel strategy omits `group_size` and is currently valid only for INT8.

MSE uses candidate search to minimize reconstruction error without representative calibration text. The default group size is 32 for MSE INT4 and 128 for MSE INT8. The default chunk size is 4096 for INT4 and 1024 for INT8. W8A16 channel is valid for ordinary Linear, attention, and shared-expert weights, but routed MoE weights must use group strategy. W8A8 is the channel-strategy path for dynamic INT8 activations.
MSE W8A16 scales are stored as BF16. W8A8 scale dtype is configurable as
FP32 or BF16.

## GPTQ

GPTQ is a calibrated weight-only method. It collects a running Hessian from representative activations, applies Cholesky-based error feedback, and can use activation ordering. True-sequential mode quantizes projection groups in sequence so later groups see the effect of earlier quantization.

GPTQ uses the tensor ABI containing `qweight`, `qzeros`, `scales`, and `g_idx`. Its defaults are:

- block size `128`
- damp percent `0.01`
- `desc_act=true`
- `static_groups=false`
- `true_sequential=true`
- `symmetric=true`

GPTQ currently supports weight-only A16 group quantization and maps to the native GPTQ format. Whether a runtime can load a particular GPTQ artifact depends on the loader, projection shapes, and kernel implementation.

## AWQ

AWQ is a calibrated method that searches activation-aware scaling candidates and can apply output-MSE clipping. Its native GEMM path uses asymmetric zero points and GEMM-oriented packing.

The current policy requires:

- 4-bit weights
- A16 activations
- group strategy
- zero points enabled

The defaults include duo scaling, output clipping, a 20-point search grid, and a 1 GiB maximum chunk-memory setting. AWQ maps to the native AWQ format. The target runtime must support the corresponding AWQ GEMM ABI.

## AutoRound

AutoRound optimizes learnable rounding offsets and can tune quantization ranges with min/max updates. Quantized-input cascading lets later optimization steps consume quantized inputs. The current integration runs on a single device and uses settings for iterations, learning rates, batch size, gradient accumulation, momentum, min/max tuning, and quantized-input cascading.

AutoRound currently requires symmetric weights, A16 activations, and group strategy. It uses the GPTQ tensor ABI for storage but retains `algorithm: autoround` provenance in the output metadata. Algorithm choice and packing format are separate concerns. An official AutoRound JSON configuration can supply method settings, while explicit CLI and recipe values take precedence.
The official AutoRound package is optional and loaded lazily only for explicit
reference or parity workflows. Native execution does not import it. Exported
metadata uses `provider: flagos-compressor` while retaining official-compatible
field names.

## Method and format mapping

The implementation enforces this mapping:

| Method | Output format |
|--------|---------------|
| MSE | `compressed-tensors` |
| GPTQ | `gptq` |
| AWQ | `awq` |
| AutoRound | `gptq` with AutoRound provenance |

The format is inferred from the method when omitted. An incompatible explicit format is rejected.

## Calibration and deployment boundary

GPTQ, AWQ, and AutoRound require representative calibration data and an available execution backend. Calibration quality depends on sample diversity and, for MoE models, whether selected experts receive representative routed tokens.

A structurally valid artifact may still be unusable when the runtime lacks the declared ABI, architecture mapping, or hardware kernel. Run structural validation first, then runtime loading, accuracy, memory, and throughput tests.
