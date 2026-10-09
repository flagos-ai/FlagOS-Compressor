# FlagOS-Compressor Documentation

FlagOS-Compressor is a checkpoint conversion and quantization tool for local [Hugging Face](https://huggingface.co/docs) `safetensors` model directories.

Use this documentation to install the tool, quantize model weights, convert checkpoints, calibrate outputs, and validate generated artifacts.

## Start here

::::{grid} 1 1 2 2
:gutter: 2
:class-container: flagos-home-grid flagos-home-grid-primary

:::{grid-item-card} Run your first quantization
:link: tutorial/index
:link-type: doc

Follow one complete inspect -> quantize -> validate path with a local model directory.
:::

:::{grid-item-card} Install and prepare the CLI
:link: how-to/install
:link-type: doc

Set up dependencies, optional extras, and the command-line entrypoint before running a job.
:::

::::

## Common workflows

::::{grid} 1 2 2 3
:gutter: 2
:class-container: flagos-home-grid

:::{grid-item-card} Quantize weights
:link: how-to/quantize-weights
:link-type: doc

Run weight-only quantization with a recipe or explicit CLI options.
:::

:::{grid-item-card} Calibrate models
:link: how-to/calibrate-models
:link-type: doc

Collect calibration data and use it for quantization workflows.
:::

:::{grid-item-card} Select layers
:link: how-to/select-layers
:link-type: doc

Target modules with selectors, regex rules, and policy constraints.
:::

:::{grid-item-card} Validate outputs
:link: how-to/validate-outputs
:link-type: doc

Check generated artifacts before loading or publishing a compressed model.
:::

:::{grid-item-card} Convert FP4 and FP8
:link: how-to/convert-fp4-fp8-to-bf16
:link-type: doc

Convert low-precision checkpoint formats back to BF16 when needed.
:::

:::{grid-item-card} Troubleshoot failures
:link: how-to/troubleshoot
:link-type: doc

Recover from common configuration, validation, and model-family issues.
:::

::::

```{toctree}
:caption: Tutorial
:maxdepth: 2
:hidden:

tutorial/index
```

```{toctree}
:caption: Explanation
:maxdepth: 2
:hidden:

Overview <explanation/index>
explanation/what-is-flagos-compressor
explanation/architecture
explanation/processing-pipeline
explanation/quantization-families
explanation/selection-semantics
explanation/validation-boundaries
```

```{toctree}
:caption: How-to guides
:maxdepth: 2
:hidden:

Overview <how-to/index>
how-to/install
how-to/convert-fp4-fp8-to-bf16
how-to/quantize-weights
how-to/per-selector-quantization
how-to/select-layers
how-to/calibrate-models
how-to/handle-model-families
how-to/recommended-test-matrix
how-to/validate-outputs
how-to/troubleshoot
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

Overview <reference/index>
reference/cli
reference/formats
reference/selectors
reference/recipes
reference/output-files
reference/compatibility
```
