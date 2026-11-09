"""Exercise MiMo's layer-dependent input contract without downloading weights."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

from flagos_compressor.calibration.inputs import prepare_calibration_inputs
from flagos_compressor.calibration.modeling import decoder_layers, forward_layer_samples
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.core.policy import AWQPolicy, AutoRoundPolicy, QuantizationPolicy


class _HybridBackbone(nn.Module):
    """Real attention blocks with MiMo's global/SWA mask and RoPE selection."""

    # MiMo remote code does not supply the usual decoder discovery metadata.
    _no_split_modules = []

    def __init__(self, pattern, swa_name="sliding_window_attention"):
        super().__init__()
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=len(pattern),
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
            max_position_embeddings=32,
            use_cache=False,
            sliding_window=3,
            layer_types=["sliding_attention" if p else "full_attention" for p in pattern],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
        config._attn_implementation = "eager"
        self.config = config
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config, index) for index in range(len(pattern))
        )
        for layer, is_swa in zip(self.layers, pattern):
            layer.attention_type = (
                swa_name if is_swa else "full_attention"
            )
        self.rotary_emb = LlamaRotaryEmbedding(config)
        swa_config = copy.deepcopy(config)
        swa_config.rope_parameters = {"rope_type": "default", "rope_theta": 100.0}
        self.swa_rotary_emb = LlamaRotaryEmbedding(swa_config)
        self.embedding_scale = 1.5
        self.eval()

    def reference_kwargs(self, batch, layer):
        hidden = self.embed_tokens(batch["input_ids"])
        length = hidden.shape[1]
        positions = batch.get("position_ids", torch.arange(length).unsqueeze(0))
        indices = torch.arange(length)
        visible = indices[None, :] <= indices[:, None]
        rotary = self.rotary_emb
        if layer.attention_type in ("sliding_window_attention", "sliding_attention"):
            visible &= indices[None, :] > indices[:, None] - 3
            rotary = self.swa_rotary_emb
        visible = visible[None, None] & batch["attention_mask"][:, None, None, :].bool()
        mask = torch.zeros(visible.shape, dtype=hidden.dtype)
        mask.masked_fill_(~visible, torch.finfo(hidden.dtype).min)
        return {
            "attention_mask": mask,
            "position_embeddings": rotary(hidden, positions),
            "position_ids": positions,
            "cache_position": indices,
            "past_key_values": None,
            "use_cache": False,
        }


    def forward(self, input_ids, attention_mask, position_ids=None, use_cache=False):
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        hidden = self.embed_tokens(input_ids) * self.embedding_scale
        for layer in self.layers:
            options = self.reference_kwargs(batch, layer)
            options["capture_marker"] = torch.tensor([7])
            hidden = layer(hidden, **options)
        return hidden


class _HybridModel(nn.Module):
    _no_split_modules = []

    def __init__(self, pattern, swa_name="sliding_window_attention"):
        super().__init__()
        self.config = SimpleNamespace(model_type="mimo_v2", num_hidden_layers=len(pattern))
        self.model = _HybridBackbone(pattern, swa_name)
        self.eval()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def reference_kwargs(self, batch, layer):
        return self.model.reference_kwargs(batch, layer)


def _batches():
    return [
        {"input_ids": torch.randint(0, 32, (1, 8)),
         "attention_mask": torch.ones(1, 8, dtype=torch.long)},
        {"input_ids": torch.randint(0, 32, (1, 8)),
         "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]]),
         "position_ids": torch.arange(4, 12).unsqueeze(0)},
    ]


@pytest.mark.parametrize("pattern", [(0, 1, 0), (1, 0, 1)])
@pytest.mark.parametrize("swa_name", ["sliding_window_attention", "sliding_attention"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_hybrid_layer_replay_matches_full_forward(pattern, swa_name, batch_size):
    torch.manual_seed(42)
    model = _HybridModel(pattern, swa_name)
    layers = decoder_layers(model)
    batches = [{key: value.expand(batch_size, -1) for key, value in batch.items()}
               for batch in _batches()]
    samples, layer_kwargs = prepare_calibration_inputs(
        model, layers, batches, device=torch.device("cpu")
    )
    with torch.no_grad():
        for sample_index, batch in enumerate(batches):
            expected = model.model.embed_tokens(batch["input_ids"]) * model.model.embedding_scale
            actual = samples[sample_index][0][0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.device.type == "cpu"
            wrong = actual
            for name, layer in layers:
                assert layer_kwargs[name][sample_index]["capture_marker"].item() == 7
                reference = model.reference_kwargs(batch, layer)
                options = layer_kwargs[name][sample_index]
                torch.testing.assert_close(options["attention_mask"], reference["attention_mask"])
                torch.testing.assert_close(
                    options["position_embeddings"], reference["position_embeddings"]
                )
                expected = layer(expected, **reference)
                actual = layer(actual, **options)
                wrong = layer(wrong, **samples[sample_index][1])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, model(**batch), rtol=0, atol=0)
            assert not torch.allclose(wrong, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
def test_all_calibrators_use_each_layers_mask_and_rope(method):
    torch.manual_seed(7)
    model = _HybridModel((0, 1, 0))
    layers = decoder_layers(model)
    # Two equal-length samples exercise both AutoRound input streams.
    batches = _batches()[:1] * 2
    samples, layer_kwargs = prepare_calibration_inputs(
        model, layers, batches, device=torch.device("cpu")
    )
    calls = {name: 0 for name, _ in layers}
    handles = []
    for name, layer in layers:
        reference = model.reference_kwargs(batches[0], layer)

        def check(_module, _args, kwargs, name=name, reference=reference):
            calls[name] += 1
            torch.testing.assert_close(kwargs["attention_mask"], reference["attention_mask"])
            torch.testing.assert_close(
                kwargs["position_embeddings"], reference["position_embeddings"]
            )

        handles.append(layer.register_forward_pre_hook(check, with_kwargs=True))
    try:
        results = quantize_model_sequential(
            layers, samples,
            QuantizationPolicy(
                selections=("linear",), method=method, num_bits=4, group_size=8,
                awq=AWQPolicy(n_grid=1),
                autoround=AutoRoundPolicy(iters=2, batch_size=1),
            ),
            device=torch.device("cpu"), layer_kwargs=layer_kwargs,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert len(results) == 21
    assert all(count >= 2 for count in calls.values())


def test_non_mimo_uses_existing_first_layer_capture():
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
    )).eval()
    layers = decoder_layers(model)
    samples, layer_kwargs = prepare_calibration_inputs(
        model, layers, _batches()[:1], device=torch.device("cpu")
    )
    assert layer_kwargs is None
    assert len(forward_layer_samples(layers[0][1], samples, device=torch.device("cpu"))) == 1


def test_mimo_discovery_prefers_standard_contract_and_registered_names():
    model = _HybridModel((0, 1))
    # Move the real modules to a nested registered path.
    backbone = model.model
    del model.model
    model.language_model = backbone
    model._no_split_modules = ["LlamaDecoderLayer"]
    assert [name for name, _ in decoder_layers(model)] == [
        "language_model.layers.0", "language_model.layers.1"]
    # A second decoder-like list must not override a valid standard contract.
    class OtherDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_type = "full_attention"
            self.self_attn = nn.Identity()
            self.mlp = nn.Identity()
    model.other = nn.ModuleList([OtherDecoder(), OtherDecoder()])
    assert len(decoder_layers(model)) == 2
    model._no_split_modules = []
    with pytest.raises(RuntimeError, match="uniquely"):
        decoder_layers(model)
    del model.other
    assert [name for name, _ in decoder_layers(model)] == [
        "language_model.layers.0", "language_model.layers.1"]


def test_mask_api_compatibility_is_scoped_to_the_backbone(monkeypatch):
    from flagos_compressor.calibration.inputs import _mimo_mask_compat

    def modern_mask(*, inputs_embeds):
        return inputs_embeds + 1

    # Mimic remote forward code which imports a newer HF helper but still
    # calls it with the older argument names.
    monkeypatch.setitem(globals(), "create_causal_mask", modern_mask)
    class Backbone(nn.Module):
        def forward(self, value):
            return create_causal_mask(input_embeds=value, cache_position=None)

    backbone, other = Backbone(), Backbone()
    with pytest.raises(TypeError):
        backbone(torch.ones(1))
    with _mimo_mask_compat(backbone):
        torch.testing.assert_close(backbone(torch.ones(1)), torch.full((1,), 2.))
        assert globals()["create_causal_mask"] is modern_mask
        with pytest.raises(TypeError):
            other(torch.ones(1))
    assert "forward" not in backbone.__dict__
    with pytest.raises(TypeError):
        backbone(torch.ones(1))
