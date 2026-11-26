"""Malformed dynamic schemes should produce module-specific diagnostics."""

import json

import pytest

from flagos_compressor.calibration.runner import NativeQuantizedLayer
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.native_quantized import save_native_quantized_model
from flagos_compressor.packing.autogptq import pack_autogptq
from flagos_compressor.quantizers.rtn import quantize_rtn
from test_native_export import _ToyModel, _source_checkpoint


@pytest.fixture
def native_artifact(tmp_path):
    model = _ToyModel()
    results = {}
    for name in ("proj", "untouched"):
        quant = quantize_rtn(
            model.get_submodule(name).weight, bits=4, group_size=8, symmetric=True
        )
        packed = pack_autogptq(
            quant.weight, quant.scales, quant.zeros, quant.g_idx, bits=4
        )
        results[name] = NativeQuantizedLayer("gptq", packed, num_bits=4, group_size=8)
    output = tmp_path / "output"
    save_native_quantized_model(
        _source_checkpoint(tmp_path, model),
        output,
        model,
        results,
        method="gptq",
        bits=4,
        group_size=8,
    )
    return output


def set_dynamic(output, dynamic):
    path = output / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"]["dynamic"] = dynamic
    path.write_text(json.dumps(config))


@pytest.mark.parametrize(
    "field,value",
    [("group_size", value) for value in (None, "8", True, 8.0, 0, -1, [], {})]
    + [("bits", value) for value in (None, "4", True, 4.0, 2, [], {})],
)
def test_invalid_dynamic_scheme_returns_module_error(native_artifact, field, value):
    set_dynamic(native_artifact, {r"+:^proj$": {field: value}})
    result = validate_artifact(native_artifact)
    assert not result["valid"]
    assert any(
        "proj" in error and field in error and repr(value) in error
        for error in result["errors"]
    )


@pytest.mark.parametrize(
    "override", [{}, {"bits": 4}, {"group_size": 8}, {"bits": 4, "group_size": 8}]
)
def test_valid_dynamic_scheme_inherits_omitted_fields(native_artifact, override):
    set_dynamic(native_artifact, {r"+:^proj$": override})
    result = validate_artifact(native_artifact)
    assert result["valid"], result["errors"]
    assert result["int4_tensors"] == 2


def test_invalid_dynamic_scheme_does_not_stop_other_module_validation(native_artifact):
    set_dynamic(
        native_artifact,
        {
            r"+:^proj$": {"group_size": None},
            r"+:^untouched$": {"bits": "4"},
        },
    )
    result = validate_artifact(native_artifact)
    assert not result["valid"]
    assert any("proj" in error and "group_size" in error for error in result["errors"])
    assert any("untouched" in error and "bits" in error for error in result["errors"])
