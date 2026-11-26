"""Incremental, failure-preserving calibration coverage reports."""

from contextlib import contextmanager
import json
from pathlib import Path
import time


class CalibrationCoverage(dict):
    """Reference rows plus separately measured AutoRound optimizer rows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimization_input_rows = None
        self.fallbacks = {}


class CalibrationReport:
    def __init__(self, path, policy, sample_blocks):
        self.path = Path(path)
        self.policy = policy
        self.data = {
            "schema": "flagos-compressor.calibration.v1",
            "method": policy.method,
            "weight_bits": policy.num_bits,
            "group_size": policy.group_size,
            "seed": policy.calibration.seed,
            "sample_blocks": sample_blocks,
            "state": "running",
            "layers": [],
            "unobserved_policy": policy.calibration.unobserved_policy,
            "coverage_unit": "input rows per selected projection",
            "coverage_scope": (
                "one full FP reference pass before optimization"
                if policy.method == "autoround"
                else "statistics collection pass"
            ),
        }
        if policy.target_scheme_rules:
            from dataclasses import asdict

            self.data.update(
                weight_bits="per_module",
                group_size="per_module",
                target_scheme_rules=[
                    asdict(rule) for rule in policy.target_scheme_rules
                ],
            )

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    @contextmanager
    def layer(self, name, module_names):
        rows = CalibrationCoverage({name: 0 for name in module_names})
        record = {"name": name, "state": "running", "input_rows": rows}
        if self.policy.target_scheme_rules:
            from flagos_compressor.inspect.tensor_classifier import classify_weight

            record["module_quantization"] = {}
            for module_name in module_names:
                weight_name = f"{name}.{module_name}.weight"
                _, tags = classify_weight(weight_name)
                settings = self.policy.settings_for_name(weight_name, tags)
                record["module_quantization"][module_name] = {
                    "bits": settings.num_bits,
                    "group_size": settings.group_size,
                }
                if self.policy.method == "awq":
                    record["module_quantization"][module_name]["sym"] = not (
                        settings.awq.zero_point and settings.num_bits == 4
                    )
        self.data["layers"].append(record)
        self.save()
        started = time.monotonic()
        try:
            yield rows
            record["state"] = "completed"
        except Exception as exc:
            record.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            self.data["state"] = "failed"
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["unobserved_modules"] = sorted(
                name for name, count in rows.items() if count == 0
            )
            record["fallbacks"] = rows.fallbacks
            if rows.optimization_input_rows is not None:
                record["optimization_input_rows"] = rows.optimization_input_rows
                record["optimization_unobserved_modules"] = sorted(
                    name
                    for name, count in rows.optimization_input_rows.items()
                    if count == 0
                )
            self.save()

    def complete(self, quantized_modules):
        fallbacks = sum(
            len(layer.get("fallbacks", {})) for layer in self.data["layers"]
        )
        self.data.update(
            state="completed",
            quantized_modules=quantized_modules,
            calibrated_modules=quantized_modules - fallbacks,
            fallback_modules=fallbacks,
            effective_method=self.data["method"] + ("+rtn" if fallbacks else ""),
        )
        self.save()
