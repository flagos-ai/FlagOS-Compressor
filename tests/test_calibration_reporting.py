import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from flagos_compressor.calibration.moe import LinearExperts2D
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.core.policy import (
    AWQPolicy,
    AutoRoundPolicy,
    CalibrationPolicy,
    GPTQPolicy,
    QuantizationPolicy,
    TargetSchemeRule,
)


class NativeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 8, bias=False)
        self.up_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)

    def forward(self, inputs):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


class NativeExperts(nn.ModuleList):
    def __init__(self, *, call_empty=False):
        super().__init__([NativeExpert(), NativeExpert()])
        self.call_empty = call_empty

    def forward(self, inputs, indices, weights):
        result = torch.zeros_like(inputs)
        for expert_index, expert in enumerate(self):
            tokens, positions = torch.where(indices == expert_index)
            if tokens.numel() or self.call_empty:
                output = expert(inputs[tokens]) * weights[tokens, positions, None]
                result.index_add_(0, tokens, output)
        return result


class RoutedBlock(nn.Module):
    def __init__(self, only_first=False, expert_layout="fused"):
        super().__init__()
        source = nn.Module()
        source.hidden_dim = 8
        source.gate_up_proj = nn.Parameter(torch.randn(2, 16, 8) * 0.1)
        source.down_proj = nn.Parameter(torch.randn(2, 8, 8) * 0.1)
        self.mlp = nn.Module()
        self.mlp.experts = LinearExperts2D(source)
        if expert_layout != "fused":
            self.mlp.experts = NativeExperts(call_empty=expert_layout == "native_empty")
        self.mlp.shared_experts = nn.Sequential()
        self.mlp.shared_experts.add_module("gate_proj", nn.Linear(8, 8, bias=False))
        self.only_first = only_first

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, 8)
        indices = torch.zeros((flat.shape[0], 1), dtype=torch.long, device=flat.device)
        if not self.only_first:
            indices[2:] = 1
        weights = torch.ones_like(indices, dtype=flat.dtype)
        result = self.mlp.experts(flat, indices, weights) + self.mlp.shared_experts(
            flat
        )
        return result.reshape_as(hidden_states)


def run_case(tmp_path, method, only_first=False, expert_layout="fused"):
    torch.manual_seed(42)
    layer = RoutedBlock(only_first, expert_layout)
    policy = QuantizationPolicy(
        selections=("linear",),
        method=method,
        num_bits=4,
        group_size=8,
        gptq=GPTQPolicy(desc_act=False),
        awq=AWQPolicy(n_grid=2),
        autoround=AutoRoundPolicy(iters=3, batch_size=1),
    )
    path = tmp_path / "coverage.json"
    result = quantize_model_sequential(
        [("model.layers.0", layer)],
        [((torch.randn(1, 6, 8),), {})],
        policy,
        device=torch.device("cpu"),
        report_path=path,
    )
    return result, json.loads(path.read_text())


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
@pytest.mark.parametrize("expert_layout", ["fused", "native", "native_empty"])
def test_report_counts_routed_tokens_without_optimizer_repetitions(
    tmp_path, method, expert_layout
):
    result, report = run_case(tmp_path, method, expert_layout=expert_layout)
    assert report["state"] == "completed"
    assert report["quantized_modules"] == len(result) == 7
    rows = report["layers"][0]["input_rows"]
    for projection in ("gate_proj", "up_proj", "down_proj"):
        assert rows[f"mlp.experts.0.{projection}"] == 2
        assert rows[f"mlp.experts.1.{projection}"] == 4
    assert rows["mlp.shared_experts.gate_proj"] == 6
    assert report["layers"][0]["unobserved_modules"] == []
    if method == "autoround":
        training = report["layers"][0]["optimization_input_rows"]
        assert training["mlp.experts.0.gate_proj"] == 2 * 3
        assert training["mlp.experts.1.gate_proj"] == 4 * 3
        assert training["mlp.shared_experts.gate_proj"] == 6 * 3
        assert report["layers"][0]["optimization_unobserved_modules"] == []


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
@pytest.mark.parametrize("expert_layout", ["fused", "native", "native_empty"])
def test_failed_calibration_preserves_missing_expert_report(
    tmp_path, method, expert_layout
):
    with pytest.raises(RuntimeError, match="did not route any tokens"):
        run_case(tmp_path, method, only_first=True, expert_layout=expert_layout)
    report = json.loads((tmp_path / "coverage.json").read_text())
    assert report["state"] == "failed"
    layer = report["layers"][0]
    assert layer["state"] == "failed"
    assert "mlp.experts.1.gate_proj" in layer["unobserved_modules"]
    assert layer["input_rows"]["mlp.experts.0.gate_proj"] == 6
    assert layer["elapsed_seconds"] >= 0


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
@pytest.mark.parametrize("expert_layout", ["fused", "native", "native_empty"])
@pytest.mark.parametrize("routed_scheme", [None, (4, 4), (8, 4)])
def test_explicit_rtn_fallback_preserves_unobserved_weights_and_export_provenance(
    tmp_path, method, expert_layout, routed_scheme
):
    from safetensors.torch import save_file
    from flagos_compressor.formats.native_quantized import save_native_quantized_model
    from flagos_compressor.core.validation import validate_artifact

    torch.manual_seed(42)
    layer = RoutedBlock(only_first=True, expert_layout=expert_layout)
    originals = {
        name: module.weight.detach().clone()
        for name, module in layer.named_modules()
        if isinstance(module, nn.Linear) and ".experts.1." in name
    }
    policy = QuantizationPolicy(
        selections=("linear",),
        method=method,
        num_bits=4,
        group_size=8,
        calibration=CalibrationPolicy(unobserved_policy="rtn"),
        gptq=GPTQPolicy(desc_act=False),
        awq=AWQPolicy(n_grid=2),
        autoround=AutoRoundPolicy(iters=3, batch_size=1),
    )
    if routed_scheme:
        policy = replace(
            policy,
            selections=(),
            format=None,
            target_scheme_rules=(
                TargetSchemeRule(
                    selection="moe.routed", weight_format=f"int{routed_scheme[0]}",
                    activation_num_bits=16, strategy="group", group_size=routed_scheme[1],
                ),
                TargetSchemeRule(
                    selection="moe.shared", weight_format="int4",
                    activation_num_bits=16, strategy="group", group_size=8,
                ),
            ),
        )
    packing = "awq" if method == "awq" and routed_scheme is None else "gptq"
    result = quantize_model_sequential(
        [("model.layers.0", layer)],
        [((torch.randn(1, 6, 8),), {})],
        policy,
        device=torch.device("cpu"),
        report_path=tmp_path / "coverage.json",
    )
    report = json.loads((tmp_path / "coverage.json").read_text())
    assert report["state"] == "completed"
    assert report["effective_method"] == method + "+rtn"
    assert report["fallback_modules"] == 3 and report["calibrated_modules"] == 4
    assert len(result) == 7
    for name, original in originals.items():
        entry = result["model.layers.0." + name]
        assert entry.algorithm == "rtn" and entry.fallback_from == method
        assert entry.packing == packing
        assert (entry.num_bits, entry.group_size) == (routed_scheme or (4, 8))
        current = layer.get_submodule(name).weight
        assert current.norm() > 0
        assert torch.mean((current - original).square()) < 0.1 * torch.mean(
            original.square()
        )
        assert report["layers"][0]["input_rows"][name] == 0
        assert report["layers"][0]["fallbacks"][name]["method"] == "rtn"
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    source = tmp_path / "source"
    source.mkdir()
    save_file({"dummy": torch.ones(1)}, source / "model.safetensors")
    (source / "config.json").write_text(
        json.dumps({"model_type": "toy", "torch_dtype": "float32"})
    )
    output = tmp_path / "export"
    save_native_quantized_model(
        source, output, model, result, method=method, bits=4, group_size=8
    )
    qc = json.loads((output / "config.json").read_text())["quantization_config"]
    assert qc["algorithm"] == method + "+rtn"
    assert qc["fallback_quantization"]["module_count"] == 3
    assert qc["quant_method"] == packing
    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
def test_unselected_native_experts_do_not_require_coverage(tmp_path, method):
    layer = RoutedBlock(only_first=True, expert_layout="native")
    policy = QuantizationPolicy(
        selections=("moe.shared",), method=method, num_bits=4, group_size=8,
        awq=AWQPolicy(n_grid=2), autoround=AutoRoundPolicy(iters=2, batch_size=1),
    )
    path = tmp_path / "coverage.json"
    result = quantize_model_sequential(
        [("model.layers.0", layer)], [((torch.randn(1, 6, 8),), {})],
        policy, device=torch.device("cpu"), report_path=path,
    )
    assert list(result) == ["model.layers.0.mlp.shared_experts.gate_proj"]
    report = json.loads(path.read_text())
    assert report["state"] == "completed"
    assert report["fallback_modules"] == 0
    assert report["layers"][0]["unobserved_modules"] == []


def test_empty_gptq_hook_does_not_create_calibration_observations():
    from flagos_compressor.quantizers.gptq import GPTQQuantizer

    quantizer = GPTQQuantizer(torch.randn(8, 8))
    quantizer.add_batch(torch.empty(0, 8))
    assert quantizer.num_samples == 0 and quantizer.hessian.count_nonzero() == 0
    with pytest.raises(RuntimeError, match="at least one calibration batch"):
        quantizer.quantize(group_size=8)


def test_unobserved_policy_rejects_silent_skip():
    with pytest.raises(ValueError, match="unobserved_policy"):
        CalibrationPolicy(unobserved_policy="skip")


@pytest.mark.parametrize("expert_layout", ["fused", "native", "native_empty"])
def test_autoround_reports_optimizer_gap_separately_from_reference_coverage(
    tmp_path, monkeypatch, expert_layout
):
    from flagos_compressor.calibration import runner

    class InputRoutedBlock(RoutedBlock):
        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, 8)
            indices = (flat[:, :1] > 0).long()
            result = self.mlp.experts(
                flat, indices, torch.ones_like(indices, dtype=flat.dtype)
            )
            return (result + self.mlp.shared_experts(flat)).reshape_as(hidden_states)

    monkeypatch.setattr(runner, "_sample_indices", lambda *args: [0])
    layer = InputRoutedBlock(expert_layout=expert_layout)
    samples = [((torch.full((1, 6, 8), sign),), {}) for sign in (-1.0, 1.0)]
    policy = QuantizationPolicy(
        selections=("linear",),
        method="autoround",
        num_bits=4,
        group_size=8,
        calibration=CalibrationPolicy(unobserved_policy="rtn"),
        autoround=AutoRoundPolicy(iters=2, batch_size=1),
    )
    result = quantize_model_sequential(
        [("model.layers.0", layer)],
        samples,
        policy,
        device=torch.device("cpu"),
        report_path=tmp_path / "coverage.json",
    )
    report = json.loads((tmp_path / "coverage.json").read_text())["layers"][0]
    assert not report["unobserved_modules"]
    name = "mlp.experts.1.gate_proj"
    assert report["input_rows"][name] == 6
    assert report["optimization_input_rows"][name] == 0
    assert result["model.layers.0." + name].fallback_reason == "no_optimization_input"
    assert len(report["fallbacks"]) == 3
