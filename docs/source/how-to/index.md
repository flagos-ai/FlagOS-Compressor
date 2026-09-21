# How-to guides

Use these guides when you already know the outcome you want and need the commands, checks, and fallback guidance to complete one task.

## Guide map

| Task | Start here when you need to |
|------|-----------------------------|
| [Installation guide](install.md) | install the CLI from a local repository checkout |
| [FP4 and FP8 conversion](convert-fp4-fp8-to-bf16.md) | convert source-quantized FP4 or FP8 checkpoint weights to BF16 |
| [Weight quantization](quantize-weights.md) | produce W4A16, W8A16, W8A8, GPTQ, AWQ, or AutoRound artifacts |
| [Per-selector quantization](per-selector-quantization.md) | assign different MSE W4A16, W8A16, and W8A8 schemes in one command |
| [Layer selection](select-layers.md) | control which weights are quantized or excluded |
| [Model calibration](calibrate-models.md) | supply calibration data for calibrated methods |
| [Model family handling](handle-model-families.md) | choose checks for Dense, MoE, DeepSeek, and special layouts |
| [Recommended test matrix](recommended-test-matrix.md) | run the complete B00, D, M, S, and indexer case set |
| [Output validation](validate-outputs.md) | validate generated files and prepare runtime loading tests |
| [Troubleshooting](troubleshoot.md) | recover from input, selection, quantization, backend, and runtime failures |

## Working rule

Run `inspect` and a quantization `--dry-run` before writing large output directories. Most avoidable failures are easier to diagnose before shards are written.

If a task fails, switch to the [troubleshooting guide](troubleshoot.md) rather than guessing at the next step.
