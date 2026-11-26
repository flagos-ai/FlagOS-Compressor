import json
from dataclasses import replace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from flagos_compressor.calibration.plan import build_calibration_plan
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.cli.helpers import build_quantization_policy
from flagos_compressor.cli.main import build_parser
from flagos_compressor.core.policy import (
    CalibrationPolicy,
    GPTQPolicy,
    QuantizationPolicy,
    TargetSchemeRule,
)
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.native_quantized import save_native_quantized_model
from test_calibration_reporting import RoutedBlock
from test_gptq_quantization import _unpack_rows, _unpack_columns


def rule(target, bits, group=8):
    return TargetSchemeRule(
        selection=target,
        weight_format=f"int{bits}",
        activation_num_bits=16,
        strategy="group",
        group_size=group,
    )


def policy(shared8=False):
    rules = (rule("attention", 8), rule("moe", 4))
    if shared8:
        rules += (rule("moe.shared", 8),)
    return QuantizationPolicy(
        method="gptq",
        target_scheme_rules=rules,
        calibration=CalibrationPolicy(unobserved_policy="rtn"),
        gptq=GPTQPolicy(desc_act=False),
    )


class MixedBlock(RoutedBlock):
    def __init__(self, only_first=False):
        super().__init__(only_first)
        self.self_attn = nn.Module()
        self.self_attn.q_a_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.kv_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.o_b_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.self_attn.o_b_proj(
            self.self_attn.q_a_proj(hidden_states)
            + self.self_attn.kv_proj(hidden_states)
        )
        return super().forward(hidden_states)


@pytest.mark.parametrize("shared8", [False, True])
def test_actual_mixed_calibration_export_roundtrip(tmp_path, shared8):
    torch.manual_seed(42)
    block = MixedBlock(only_first=True)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([block])
    layers = [("model.layers.0", block)]
    settings = policy(shared8)
    plan = build_calibration_plan(layers, settings)
    report_path = tmp_path / "report.json"
    results = quantize_model_sequential(
        layers,
        [((torch.randn(1, 16, 8),), {})],
        settings,
        device=torch.device("cpu"),
        report_path=report_path,
    )
    assert len(results) == 10
    for name, entry in results.items():
        expected_bits = (
            8
            if ".self_attn." in name or (shared8 and ".shared_experts." in name)
            else 4
        )
        assert entry.num_bits == plan["selected_modules"][name]["bits"] == expected_bits
        assert entry.group_size == 8
        packed = entry.packed
        assert tuple(packed.qweight.shape) == (8 // (32 // expected_bits), 8)
        codes = _unpack_rows(packed.qweight, expected_bits)
        zeros = (_unpack_columns(packed.qzeros, expected_bits) + 1) & (
            (1 << expected_bits) - 1
        )
        weight = (
            packed.scales[packed.g_idx.long()] * (codes - zeros[packed.g_idx.long()])
        ).t()
        torch.testing.assert_close(
            weight, model.get_submodule(name).weight, rtol=1e-5, atol=1e-6
        )
        assert (entry.algorithm == "rtn") == (".experts.1." in name)
    report = json.loads(report_path.read_text())
    assert report["weight_bits"] == "per_module" and report["fallback_modules"] == 3
    assert report["layers"][0]["module_quantization"]["self_attn.q_a_proj"]["bits"] == 8
    source = tmp_path / "source"
    source.mkdir()
    save_file({"dummy": torch.ones(1)}, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({"model_type": "toy"}))
    output = tmp_path / "export"
    save_native_quantized_model(
        source,
        output,
        model,
        results,
        method="gptq",
        bits=4,
        group_size=8,
        max_shard_size=512,
    )
    qc = json.loads((output / "config.json").read_text())["quantization_config"]
    assert (
        qc["flagos_module_quantization"]["model.layers.0.self_attn.q_a_proj"]["bits"]
        == 8
    )
    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    assert validation["int8_tensors"] == (4 if shared8 else 3)
    assert validation["int4_tensors"] == (6 if shared8 else 7)
    # Removing a real W8 override must invalidate the artifact, even when all
    # tensor names remain present and the top-level GPTQ config still says W4.
    qc["dynamic"] = {
        key: value for key, value in qc["dynamic"].items() if not key.startswith("+:")
    }
    cfg = json.loads((output / "config.json").read_text())
    cfg["quantization_config"] = qc
    (output / "config.json").write_text(json.dumps(cfg))
    assert not validate_artifact(output)["valid"]


def test_gptq_rejects_mixed_bits_within_fused_attention_before_calibration():
    settings = replace(
        policy(),
        target_scheme_rules=policy().target_scheme_rules
        + (
            TargetSchemeRule(
                name_pattern=r"\.kv_proj\.weight$",
                weight_format="int4",
                activation_num_bits=16,
                strategy="group",
                group_size=8,
            ),
        ),
    )
    with pytest.raises(ValueError, match="fused"):
        build_calibration_plan([("model.layers.0", MixedBlock())], settings)


def test_gptq_rejects_different_bits_within_routed_expert_bank():
    settings = replace(
        policy(),
        target_scheme_rules=policy().target_scheme_rules
        + (
            TargetSchemeRule(
                name_pattern=r"\.experts\.1\.",
                weight_format="int8",
                activation_num_bits=16,
                strategy="group",
                group_size=8,
            ),
        ),
    )
    with pytest.raises(ValueError, match="routed MoE bank"):
        build_calibration_plan([("model.layers.0", MixedBlock())], settings)


def test_cli_resolves_gptq_local_schemes_and_last_shared_override():
    args = build_parser().parse_args(
        [
            "quantize",
            "--input",
            "in",
            "--output",
            "out",
            "--method",
            "gptq",
            "--select",
            "attention=int8",
            "activation-bits=16",
            "strategy=group",
            "group-size=128",
            "--select",
            "moe=int4",
            "activation-bits=16",
            "strategy=group",
            "group-size=128",
            "--select",
            "moe.shared=int8",
            "activation-bits=16",
            "strategy=group",
            "group-size=128",
        ]
    )
    result = build_quantization_policy(args)
    assert result.format == "gptq"
    chosen = result.settings_for_name(
        "model.layers.0.mlp.shared_experts.gate_proj.weight", ("moe", "moe.shared")
    )
    assert chosen.num_bits == 8 and chosen.group_size == 128


@pytest.mark.parametrize("method", ["awq", "autoround"])
def test_other_calibration_algorithms_accept_explicit_mixed_rules(method):
    settings = replace(policy(), method=method, format=None)
    assert settings.format == "gptq"
    assert (
        settings.settings_for_name(
            "model.layers.0.self_attn.q_a_proj.weight", ("attention",)
        ).num_bits
        == 8
    )


def test_gptq_rejects_channelwise_activation_quantization():
    with pytest.raises(ValueError, match="groupwise W4A16/W8A16"):
        QuantizationPolicy(
            method="gptq",
            target_scheme_rules=(
                TargetSchemeRule(
                    selection="attention",
                    weight_format="int8",
                    activation_num_bits=8,
                    strategy="channel",
                ),
            ),
        )
