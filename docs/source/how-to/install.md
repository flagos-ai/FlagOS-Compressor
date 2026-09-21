# Install FlagOS-Compressor

Install FlagOS-Compressor before running inspection, conversion, quantization, or validation commands.

## Prerequisites

Prepare the following before installing:

- Python 3.10 or later
- enough local storage for the repository, dependencies, and generated model outputs
- a PyTorch installation appropriate for your target backend
- Git, if you install from a repository checkout

## Install from a source checkout

Use a source checkout when you need editable code, documentation updates, or a pinned commit.

```bash
git clone https://github.com/flagos-ai/FlagOS-Compressor.git
cd FlagOS-Compressor
python -m pip install -e .
```

Install source extras with the same optional dependency names:

```bash
python -m pip install -e ".[datasets]"
python -m pip install -e ".[official-autoround]"
```

Use `datasets` for Hugging Face dataset-backed calibration inputs. Use `official-autoround` only when you need the official AutoRound package path. Native AutoRound execution does not require that package.

The official AutoRound Python entry point is loaded lazily only when an
explicit reference or parity workflow requests it. Normal installation and
native FlagOS-Compressor execution do not import the optional package.

Use the main branch unless your workflow requires a pinned tag or commit:

```bash
git switch main
git pull --ff-only origin main
```

## Verify the installation

Check the package entry point:

```bash
flagos-compressor --help
flagos-compressor quantize --help
```

Expected result:

- `flagos-compressor --help` lists `convert`, `quantize`, `inspect`, and `validate`
- `flagos-compressor quantize --help` lists quantization, selection, calibration, GPTQ, AWQ, and AutoRound options.

If you installed from source, also record the commit:

```bash
git rev-parse HEAD
```

## Choose a backend

Pass the backend explicitly in conversion and quantization commands:

| Backend | Use when | Example |
|---------|----------|---------|
| `cuda` | CUDA-enabled PyTorch can see an NVIDIA GPU | `--backend cuda --device cuda:0` |
| `cpu` | you need a portable fallback or functional smoke test | `--backend cpu` |
| `npu`, `mlu`, `musa` | your PyTorch build exposes one of these extension backends | `--backend npu` |

```{note}
The CLI default backend is `cpu`. CUDA is usually the preferred generation environment when available, but generated artifacts still need independent validation on their runtime target.
```

## Record version information

Record the exact tool revision and model inputs when producing artifacts:

```bash
cat > version_info.txt << EOF
Tool: FlagOS-Compressor
Package: $(python -m pip show flagos-compressor | grep '^Version:')
Date: $(date)
Model: qwen3.5-dense
Model revision: main
Output: qwen3.5-dense-w8a16-g128
EOF
```

For a source checkout, add the commit:

```bash
git rev-parse HEAD >> version_info.txt
```

Copy the record next to generated outputs when you need reproducibility:

```bash
cp version_info.txt "$MODEL_OUTPUT_ROOT/qwen3.5-dense-w8a16-g128/"
```

## Fallbacks

If installation fails:

- verify the Python version is at least 3.10
- install a PyTorch build that matches your device backend
- use `python -m pip install -e .` from the repository root for source installs
- rerun `flagos-compressor --help` before attempting model processing
