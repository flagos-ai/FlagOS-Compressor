# Explanation

Use these pages to understand FlagOS-Compressor concepts, boundaries, and processing models. These pages explain why the tool behaves as it does. Use the [how-to guides](../how-to/index.md) when you need task steps.

## Concept map

| Page | Concept |
|------|---------|
| [Project scope](what-is-flagos-compressor.md) | project scope and non-goals |
| [Architecture](architecture.md) | major tool responsibilities and data flow |
| [Processing pipeline](processing-pipeline.md) | how local checkpoint directories are inspected, converted, quantized, and written |
| [Quantization families](quantization-families.md) | differences between MSE, GPTQ, AWQ, AutoRound, and output schemes |
| [Selection semantics](selection-semantics.md) | how selectors form a safe set of weights |
| [Validation boundaries](validation-boundaries.md) | what artifact validation can and cannot prove |
