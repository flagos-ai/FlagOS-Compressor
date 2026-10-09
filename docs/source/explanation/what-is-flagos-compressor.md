# What is FlagOS-Compressor?

FlagOS-Compressor is a model-agnostic checkpoint conversion and quantization tool for local [Hugging Face](https://huggingface.co/docs) `safetensors` model directories.

It sits between source checkpoint preparation and runtime inference testing. It does not download models, train models, deploy services, or decide whether a model is accurate enough for a business task.

## Problem it solves

Large model checkpoints often need preparation before inference experiments can begin. Source weights may be BF16, FP16, FP32, MXFP4, block FP8, or a mixture of float and source-quantized tensors. Runtime tests may require a BF16 baseline, a compressed-tensors artifact, or a native GPTQ/AWQ layout.

FlagOS-Compressor gives operators explicit control over that preparation step:

- inspect the source checkpoint.
- convert recognized low-precision formats to BF16.
- quantize selected linear weights.
- preserve unselected source-quantized weights as BF16.
- write runtime-oriented output directories.
- validate generated structure and metadata.

## Supported scope

FlagOS-Compressor focuses on:

- MXFP4 and block FP8 to BF16 conversion.
- MSE INT4 and INT8 quantization.
- W4A16, W8A16, and dynamic per-token W8A8 export.
- GPTQ, AWQ, and AutoRound workflows.
- Per-selector MSE W4A16, W8A16, and W8A8 policies in one execution.
- Dense, standard MoE, supported fused MoE layouts, GLM-4 MoE, and DeepSeek-V2/V3/V4 calibration structures.
- compressed-tensors and native GPTQ/AWQ output validation.

## Outside the scope

The tool does not handle:

- model downloading.
- model training or fine-tuning.
- automatic accuracy evaluation.
- automatic inference kernel selection.
- online service deployment.

## Where it fits

```text
Local source model
  -> inspect / dry-run
  -> convert or quantize
  -> validate
  -> inference framework loading
  -> accuracy, memory, and performance testing
```

The output of FlagOS-Compressor is the starting point for runtime testing, not the end of deployment qualification.
