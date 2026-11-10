import json

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from flagos_compressor.calibration.inputs import prepare_calibration_inputs
from flagos_compressor.calibration.modeling import decoder_layers, load_transformers_model
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.core.policy import AWQPolicy, AutoRoundPolicy, QuantizationPolicy
from flagos_compressor.formats.native_quantized import save_native_quantized_model
from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


class _MixedDtypeLlama(LlamaForCausalLM):
    """Model-wide BF16 loading also casts these checkpoint-native FP32 values."""

    def __init__(self, config):
        super().__init__(config)
        layer = self.model.layers[0]
        layer.self_attn.attention_sink_bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        layer.mlp.e_score_correction_bias = torch.nn.Parameter(
            torch.zeros(2), requires_grad=False
        )
        self.register_buffer("native_float_buffer", torch.zeros(2))
        self.register_buffer("native_int_buffer", torch.tensor([1, 2]))


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
def test_native_fp32_survives_loading_calibration_and_export(
    tmp_path, monkeypatch, sharded, method
):
    torch.manual_seed(17)
    config = LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
        use_cache=False,
    )
    source_model = _MixedDtypeLlama(config).to(torch.bfloat16).eval()
    state = source_model.state_dict(keep_vars=True)
    originals = {}
    for name in (
        "model.layers.0.self_attn.attention_sink_bias",
        "model.layers.0.mlp.e_score_correction_bias",
        "native_float_buffer",
    ):
        original = torch.tensor([1.0003, -0.01003], dtype=torch.float32)
        state[name].data = original.clone()
        originals[name] = original
        assert not torch.equal(original, original.bfloat16().float())
    source = tmp_path / "source"
    source_model.save_pretrained(source, max_shard_size="2KB" if sharded else "1GB")
    # Use an actual Transformers save/load, without a network or tokenizer fixture.
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _MixedDtypeLlama.from_pretrained)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    model, _tokenizer = load_transformers_model(source)
    loaded = model.state_dict()
    for name, original in originals.items():
        assert loaded[name].dtype == torch.float32
        torch.testing.assert_close(loaded[name], original, rtol=0, atol=0)
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.native_int_buffer.dtype == torch.int64

    layers = decoder_layers(model)
    samples, layer_kwargs = prepare_calibration_inputs(
        model, layers,
        [{"input_ids": torch.randint(0, 32, (1, 8)),
          "attention_mask": torch.ones(1, 8, dtype=torch.long)}],
        device=torch.device("cpu"),
    )
    quantized = quantize_model_sequential(
        layers, samples,
        QuantizationPolicy(
            selections=("linear",), method=method, num_bits=4, group_size=8,
            awq=AWQPolicy(n_grid=1), autoround=AutoRoundPolicy(iters=1, batch_size=1),
        ),
        device=torch.device("cpu"), layer_kwargs=layer_kwargs,
    )
    output = tmp_path / "output"
    save_native_quantized_model(
        source, output, model, quantized, method=method, bits=4, group_size=8
    )
    checkpoint = HfSafetensorsCheckpoint(output)
    for name, original in originals.items():
        exported = checkpoint.load_tensor(name)
        assert exported.dtype == torch.float32
        torch.testing.assert_close(exported, original, rtol=0, atol=0)
    assert checkpoint.load_tensor("model.embed_tokens.weight").dtype == torch.bfloat16
    assert checkpoint.load_tensor("native_int_buffer").dtype == torch.int64


@pytest.mark.parametrize("target", ["bfloat16", "float16", "float32"])
def test_uniform_fp32_checkpoint_respects_requested_load_dtype(tmp_path, monkeypatch, target):
    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
    source = LlamaForCausalLM(config).float().eval()
    source.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    saved = json.loads(config_path.read_text())
    saved.update(dtype=target, torch_dtype=target)
    config_path.write_text(json.dumps(saved))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    model, _ = load_transformers_model(tmp_path)
    assert all(parameter.dtype == getattr(torch, target) for parameter in model.parameters())


def test_declared_fp32_exception_survives_uniform_source_cast(tmp_path):
    from flagos_compressor.calibration.modeling import _restore_source_fp32
    from safetensors.torch import save_file

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(8, 8, dtype=torch.bfloat16))
    model.special = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    model._keep_in_fp32_modules = ["special"]
    original = torch.tensor([1.0003, -0.01003])
    save_file({"weight": torch.ones(8, 8), "special": original}, tmp_path / "model.safetensors")
    _restore_source_fp32(model, tmp_path)
    assert model.weight.dtype == torch.bfloat16
    assert model.special.dtype == torch.float32
    torch.testing.assert_close(model.special, original, rtol=0, atol=0)
