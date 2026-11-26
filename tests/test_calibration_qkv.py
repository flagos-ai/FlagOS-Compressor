"""Fused QKV schemes must agree before calibration and native export."""

from dataclasses import replace

import pytest
from torch import nn

from flagos_compressor.calibration.plan import build_calibration_plan
from flagos_compressor.calibration.runner import NativeQuantizedLayer
from flagos_compressor.core.compressed_tensors import validate_fusion_closure
from flagos_compressor.core.policy import QuantizationPolicy, TargetSchemeRule
from flagos_compressor.formats.native_quantized import save_native_quantized_model
from flagos_compressor.packing.autogptq import pack_autogptq
from flagos_compressor.quantizers.rtn import quantize_rtn
from test_native_export import _source_checkpoint


class QKVBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        # GQA can have different output widths within the same fused QKV unit.
        self.self_attn.q_proj = nn.Linear(8, 16, bias=False)
        self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.o_proj = nn.Linear(8, 8, bias=False)


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
@pytest.mark.parametrize("conflict", ["bits", "group_size", "excluded", None])
def test_qkv_preflight_checks_effective_schemes(method, conflict):
    base = TargetSchemeRule(
        selection="attention",
        weight_format="int4",
        activation_num_bits=16,
        strategy="group",
        group_size=8,
    )
    rules = [base]
    if conflict in ("bits", "group_size"):
        rules.append(
            replace(
                base,
                selection=None,
                name_pattern=r"\.q_proj\.weight$",
                weight_format="int8" if conflict == "bits" else "int4",
                group_size=4 if conflict == "group_size" else 8,
            )
        )
    settings = QuantizationPolicy(
        method=method,
        target_scheme_rules=tuple(rules),
        exclude_names=(r"\.q_proj\.weight$",) if conflict == "excluded" else (),
    )
    layers = [("model.layers.0", QKVBlock())]
    if conflict:
        with pytest.raises(ValueError, match="fused QKV") as exc:
            build_calibration_plan(layers, settings)
        for projection in ("q_proj", "k_proj", "v_proj"):
            assert "model.layers.0.self_attn." + projection in str(exc.value)
    else:
        assert build_calibration_plan(layers, settings)["selected_module_count"] == 4


@pytest.mark.parametrize(
    "selected_projections",
    [("q_proj",), ("k_proj",), ("v_proj",), ("k_proj", "v_proj")],
)
def test_qkv_closure_rejects_partial_selection(selected_projections):
    prefix = "model.layers.0.self_attn."
    weights = {prefix + name + ".weight" for name in ("q_proj", "k_proj", "v_proj")}
    with pytest.raises(ValueError, match="fused QKV"):
        validate_fusion_closure(
            weights, {prefix + name + ".weight" for name in selected_projections}
        )


def test_qkv_closure_keeps_attention_units_independent():
    units = [
        {
            f"model.layers.{i}.self_attn.{name}.weight"
            for name in ("q_proj", "k_proj", "v_proj")
        }
        for i in range(2)
    ]
    weights = units[0] | units[1] | {"model.layers.0.self_attn.o_proj.weight"}
    validate_fusion_closure(weights, units[0])
    validate_fusion_closure(weights, units[1])
    validate_fusion_closure(weights, weights)
    validate_fusion_closure(weights, set())
    # An isolated projection is not evidence of a fused QKV unit.
    isolated = {"model.layers.2.self_attn.q_proj.weight"}
    validate_fusion_closure(weights | isolated, isolated)


@pytest.mark.parametrize("conflict", ["bits", "group_size", "excluded", None])
def test_native_export_validates_qkv_even_without_preflight(tmp_path, conflict):
    model = QKVBlock()
    results = {}
    for name, linear in model.named_modules():
        if not isinstance(linear, nn.Linear):
            continue
        if name.endswith("q_proj") and conflict == "excluded":
            continue
        bits = 8 if name.endswith("q_proj") and conflict == "bits" else 4
        group = 4 if name.endswith("q_proj") and conflict == "group_size" else 8
        quant = quantize_rtn(linear.weight, bits=bits, group_size=group, symmetric=True)
        packed = pack_autogptq(
            quant.weight, quant.scales, quant.zeros, quant.g_idx, bits=bits
        )
        results[name] = NativeQuantizedLayer(
            "gptq", packed, num_bits=bits, group_size=group
        )
    source = _source_checkpoint(tmp_path, model)
    output = tmp_path / "export"

    def export():
        save_native_quantized_model(
            source, output, model, results, method="gptq", bits=4, group_size=8
        )

    if conflict:
        with pytest.raises(ValueError, match="fused QKV"):
            export()
        assert not output.exists()
    else:
        export()
        assert (output / "config.json").exists()
