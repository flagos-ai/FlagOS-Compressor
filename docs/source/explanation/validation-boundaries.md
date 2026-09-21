# Validation boundaries

Validation checks whether a converted or quantized output directory is internally consistent. It is a structural gate before runtime testing, not a guarantee of deployment readiness.

## What validation proves

Validation can confirm that the generated artifact has the expected file and metadata structure. It checks areas such as:

- shard index consistency.
- referenced weight and scale tensors.
- expected INT4 or INT8 storage shapes.
- quantization manifest entries.
- output configuration declarations.
- required report and support files.

A passing validation result means the artifact is well-formed according to FlagOS-Compressor's checks.

## What validation does not prove

Validation does not prove:

- a runtime framework can load the artifact.
- target hardware has compatible kernels.
- graph compilation will succeed.
- accuracy is acceptable.
- memory and performance targets are met.
- tensor parallel settings are valid.

These are runtime and evaluation responsibilities.

## Why runtime checks remain necessary

Inference frameworks interpret `config.json`, quantization metadata, and weight storage formats through their own loaders and kernels. Support varies by framework version, backend, device, and quantization scheme.

For example, a W8A8 artifact can pass structural validation while still failing on a target platform that lacks dynamic per-token INT8 kernels.

## Practical validation chain

A complete validation chain has multiple gates:

```text
inspect source
  -> dry-run plan
  -> generate output
  -> validate output structure
  -> load in runtime framework
  -> run accuracy, memory, and performance tests
```

FlagOS-Compressor owns the first four gates. The final runtime gates must be executed in the deployment environment.
