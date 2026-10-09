# FlagOS-Compressor

FlagOS-Compressor converts and quantizes local Hugging Face `safetensors` checkpoints. It supports BF16 conversion, selected-layer quantization, calibrated quantization, and output validation.

## Why use FlagOS-Compressor

- Convert supported low-precision checkpoint weights to BF16.
- Quantize selected model modules with MSE, GPTQ, AWQ, or native AutoRound workflows.
- Inspect source checkpoints and validate generated artifacts from one CLI.

## Install

Install the project from a source checkout in editable mode:

```bash
pip install -e .
```

See the [installation guide](docs/source/how-to/install.md) for prerequisites and optional extras.

## Quick start

Inspect a checkpoint, quantize selected weights, and validate the output:

```bash
flagos-compressor inspect --input /path/to/model
```

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/output \
  --select linear \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda
```

```bash
flagos-compressor validate --input /path/to/output
```

Run quantization with `--dry-run` before writing a large output directory. Structural validation does not replace runtime or performance testing. See the [output validation guide](docs/source/how-to/validate-outputs.md) for the complete verification workflow.

## Documentation and help

Start with the [documentation portal](docs/source/index.md).

- [Tutorial](docs/source/tutorial/index.md) for a complete first run
- [How-to guides](docs/source/how-to/index.md) for installation, quantization, calibration, model families, and troubleshooting
- [Reference](docs/source/reference/index.md) for CLI, formats, selectors, recipes, outputs, and compatibility
- [Troubleshooting guide](docs/source/how-to/troubleshoot.md) when a command fails
