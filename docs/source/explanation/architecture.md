# Architecture

FlagOS-Compressor is organized around local checkpoint inspection, planning, transformation, export, and validation. The architecture is intentionally bounded. It prepares artifact directories, then leaves runtime qualification to inference frameworks and hardware-specific tests.

## Main components

| Component | Responsibility |
|-----------|----------------|
| CLI | parses `convert`, `quantize`, `inspect`, and `validate` commands |
| inspection | reads model config, shard index, tensor names, shapes, and source formats |
| planning | resolves selectors, exclusions, methods, strategies, and output schemes |
| conversion | converts recognized MXFP4 and block FP8 weights to BF16 |
| quantization | applies MSE, GPTQ, AWQ, or AutoRound workflows to selected weights |
| export | writes shards, indices, quantization config, manifests, reports, and tokenizer files |
| validation | checks generated directory structure and metadata consistency |

## Data flow

```text
source checkpoint directory
  -> inspect config, index, tensor metadata
  -> build conversion or quantization plan
  -> transform selected weights
  -> write output checkpoint directory
  -> validate output structure and metadata
```

The source directory is treated as read-only during normal operation. Output is written to a separate directory.

## Command responsibilities

`inspect`
: Reports supported source formats and selectable weight groups.

`convert`
: Converts recognized FP4 or FP8 checkpoint weights to BF16.

`quantize`
: Quantizes selected weights and exports the requested ABI.

`validate`
: Validates a converted or quantized output directory.

## Boundary with runtime frameworks

The architecture does not include inference serving. Runtime frameworks such as [vLLM](https://docs.vllm.ai/) read the generated artifact after FlagOS-Compressor has finished.

That boundary is important because a structurally valid artifact can still fail runtime loading when:

- the framework does not support the declared quantization scheme.
- the accelerator lacks matching kernels.
- tensor parallel settings are invalid.
- runtime memory overhead exceeds available capacity.
