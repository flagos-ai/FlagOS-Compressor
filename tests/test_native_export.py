import json

import pytest
import torch
from safetensors.torch import save_file

from flagos_compressor.calibration.runner import NativeQuantizedLayer
from flagos_compressor.calibration.moe import LinearExperts2D
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.native_quantized import (
    _runtime_linear_names,
    save_native_quantized_model,
)
from flagos_compressor.integrations.autoround import (
    official_autoround_export_config,
)
from flagos_compressor.packing.autoawq import pack_autoawq_gemm
from flagos_compressor.packing.autogptq import pack_autogptq
from flagos_compressor.quantizers.awq import pseudo_quantize_awq
from flagos_compressor.core.policy import AutoRoundPolicy, QuantizationPolicy


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)
        self.untouched = torch.nn.Linear(8, 8, bias=False)


def test_runtime_linear_names_include_custom_grouped_linear():
    class GroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 4))

    model = _ToyModel()
    model.grouped = GroupedLinear()

    assert _runtime_linear_names(model) == ["grouped", "proj", "untouched"]


def _source_checkpoint(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {name: tensor.detach() for name, tensor in model.state_dict().items()},
        source / "model.safetensors",
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "toy", "torch_dtype": "float32"}),
        encoding="utf-8",
    )
    return source


def test_native_export_rejects_rtn_without_explicit_fallback_provenance(tmp_path):
    from flagos_compressor.quantizers.rtn import quantize_rtn

    model = _ToyModel()
    result = quantize_rtn(model.proj.weight, bits=4, group_size=8, symmetric=True)
    packed = pack_autogptq(
        result.weight, result.scales, result.zeros, result.g_idx, bits=4
    )
    with pytest.raises(ValueError, match="another method"):
        save_native_quantized_model(
            "unused",
            tmp_path / "export",
            model,
            {"proj": NativeQuantizedLayer("rtn", packed, packing="gptq")},
            method="gptq",
            bits=4,
            group_size=8,
        )


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_native_export_is_sharded_configured_and_validated(tmp_path, method):
    torch.manual_seed(3)
    model = _ToyModel().eval()
    source = _source_checkpoint(tmp_path, model)
    group_size = 8
    if method == "awq":
        params = pseudo_quantize_awq(
            model.proj.weight,
            bits=4,
            group_size=group_size,
            zero_point=True,
        )
        model.proj.weight.data.copy_(params.weight)
        packed = pack_autoawq_gemm(
            params.weight,
            params.scales,
            params.zeros,
            group_size=group_size,
        )
    else:
        scales = torch.full((8, 1), 0.05)
        zeros = torch.full((8, 1), 8.0)
        codes = torch.clamp(
            torch.round(model.proj.weight / scales) + zeros,
            0,
            15,
        )
        fake = scales * (codes - zeros)
        model.proj.weight.data.copy_(fake)
        packed = pack_autogptq(
            fake,
            scales,
            zeros,
            torch.zeros(8, dtype=torch.int32),
            bits=4,
        )
    quantized = {"proj": NativeQuantizedLayer(method, packed)}
    output = tmp_path / method

    save_native_quantized_model(
        source,
        output,
        model,
        quantized,
        method=method,
        bits=4,
        group_size=group_size,
        max_shard_size=64,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quant_config = config["quantization_config"]
    assert quant_config["quant_method"] == method
    if method == "awq":
        assert "untouched" in quant_config["modules_to_not_convert"]
    else:
        assert quant_config["modules_in_block_to_quantize"] == [["proj"]]
        assert (output / "gptq_model-4bit-8g.safetensors.index.json").exists()

    result = validate_artifact(output)
    assert result["valid"], result["errors"]
    assert result["native_method"] == method
    assert result["native_quantized_tensors"] == 1


def test_native_export_declares_actual_runtime_model_architecture(tmp_path):
    model = _ToyModel().eval()
    source = _source_checkpoint(tmp_path, model)
    source_config = source / "config.json"
    source_config.write_text(
        json.dumps(
            {
                "model_type": "outer_multimodal",
                "architectures": ["StaleOuterModel"],
            }
        ),
        encoding="utf-8",
    )
    scales = torch.full((8, 1), 0.05)
    zeros = torch.full((8, 1), 8.0)
    codes = torch.clamp(torch.round(model.proj.weight / scales) + zeros, 0, 15)
    fake = scales * (codes - zeros)
    packed = pack_autogptq(
        fake,
        scales,
        zeros,
        torch.zeros(8, dtype=torch.int32),
        bits=4,
    )
    output = tmp_path / "runtime-config"

    save_native_quantized_model(
        source,
        output,
        model,
        {"proj": NativeQuantizedLayer("gptq", packed)},
        method="gptq",
        bits=4,
        group_size=8,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "outer_multimodal"
    assert config["architectures"] == ["_ToyModel"]


def test_native_awq_export_skips_unselected_fused_moe_unit(tmp_path):
    class ModelWithExperts(_ToyModel):
        def __init__(self):
            super().__init__()
            fused = torch.nn.Module()
            fused.hidden_dim = 8
            fused.gate_up_proj = torch.nn.Parameter(torch.randn(2, 8, 8))
            fused.down_proj = torch.nn.Parameter(torch.randn(2, 8, 4))
            self.experts = LinearExperts2D(fused)

    model = ModelWithExperts().eval()
    source = tmp_path / "source-moe"
    source.mkdir()
    save_file({"dummy": torch.ones(1)}, source / "model.safetensors")
    (source / "config.json").write_text(
        json.dumps({"model_type": "toy"}), encoding="utf-8"
    )
    params = pseudo_quantize_awq(
        model.proj.weight,
        bits=4,
        group_size=8,
        zero_point=True,
    )
    packed = pack_autoawq_gemm(
        params.weight,
        params.scales,
        params.zeros,
        group_size=8,
    )
    output = tmp_path / "awq-moe-skip"

    save_native_quantized_model(
        source,
        output,
        model,
        {"proj": NativeQuantizedLayer("awq", packed)},
        method="awq",
        bits=4,
        group_size=8,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert "experts" in config["quantization_config"]["modules_to_not_convert"]
    assert validate_artifact(output)["valid"]


def test_autoround_exports_loader_compatible_gptq_with_provenance(tmp_path):
    model = _ToyModel().eval()
    source = _source_checkpoint(tmp_path, model)
    scales = torch.full((8, 1), 0.05)
    zeros = torch.full((8, 1), 8.0)
    codes = torch.clamp(torch.round(model.proj.weight / scales) + zeros, 0, 15)
    fake = scales * (codes - zeros)
    model.proj.weight.data.copy_(fake)
    packed = pack_autogptq(
        fake,
        scales,
        zeros,
        torch.zeros(8, dtype=torch.int32),
        bits=4,
    )
    output = tmp_path / "autoround"
    policy = QuantizationPolicy(
        selections=("linear",),
        method="autoround",
        group_size=8,
        autoround=AutoRoundPolicy(iters=17, batch_size=2),
    )

    save_native_quantized_model(
        source,
        output,
        model,
        {"proj": NativeQuantizedLayer("autoround", packed, packing="gptq")},
        method="autoround",
        bits=4,
        group_size=8,
        autoround_config=official_autoround_export_config(policy),
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quant_config = config["quantization_config"]
    assert quant_config["quant_method"] == "gptq"
    assert quant_config["checkpoint_format"] == "gptq"
    assert quant_config["algorithm"] == "autoround"
    assert quant_config["provider"] == "flagos-compressor"
    assert quant_config["iters"] == 17
    assert quant_config["batch_size"] == 2
    assert quant_config["enable_quanted_input"] is True
    assert (output / "quantize_config.json").exists()
    result = validate_artifact(output)
    assert result["valid"], result["errors"]
    assert result["native_method"] == "gptq"
    assert result["native_algorithm"] == "autoround"
