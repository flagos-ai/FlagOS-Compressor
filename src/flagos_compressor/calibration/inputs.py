"""Prepare model-specific inputs for sequential, cache-free calibration."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from functools import wraps
from types import FunctionType, MethodType
from typing import Any

import torch
from torch import nn

from flagos_compressor.calibration.modeling import (
    capture_first_layer_inputs,
    move_to_device,
)

LayerSamples = list[tuple[tuple[Any, ...], dict[str, Any]]]
LayerKwargs = dict[str, list[dict[str, Any]]]


@torch.no_grad()
def prepare_calibration_inputs(
    model: nn.Module,
    layers: list[tuple[str, nn.Module]],
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[LayerSamples, LayerKwargs | None]:
    """Return initial hidden states and optional kwargs for each decoder layer.

    Most models reuse the first block's kwargs. MiMo instead selects distinct
    masks and rotary embeddings for global and sliding-window attention.
    """
    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    if getattr(config, "model_type", None) == "mimo_v2":
        return _mimo_inputs(model, layers, batches, device=device)
    return capture_first_layer_inputs(
        model, layers[0][1], batches, device=device
    ), None


def _attention_type(value: str) -> str:
    # MiMo's layer attribute and HF's config.layer_types use different names.
    if value in ("sliding_attention", "sliding_window_attention"):
        return "sliding_attention"
    if value == "full_attention":
        return value
    raise ValueError(f"Unsupported MiMo attention type: {value}")


def _compatible_mask(function):
    parameters = inspect.signature(function).parameters

    @wraps(function)
    def call(*args, **kwargs):
        if "inputs_embeds" in parameters and "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        elif "input_embeds" in parameters and "inputs_embeds" in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        if "cache_position" not in parameters and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            kwargs.pop("cache_position", None)
        return function(*args, **kwargs)

    return call


@contextmanager
def _mimo_mask_compat(backbone):
    """Adapt old mask calls on this instance without changing remote-code globals."""
    forward = getattr(backbone.forward, "__func__", None)
    if forward is None:
        yield
        return
    namespace = dict(forward.__globals__)
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        if callable(namespace.get(name)):
            namespace[name] = _compatible_mask(namespace[name])
    replacement = FunctionType(forward.__code__, namespace, forward.__name__,
                               forward.__defaults__, forward.__closure__)
    replacement.__kwdefaults__ = forward.__kwdefaults__
    previous = backbone.__dict__.get("forward")
    backbone.forward = MethodType(replacement, backbone)
    try:
        yield
    finally:
        if previous is None:
            del backbone.forward
        else:
            backbone.forward = previous


def _mimo_inputs(
    model: nn.Module,
    layers: list[tuple[str, nn.Module]],
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[LayerSamples, LayerKwargs]:
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    modules = dict(model.named_modules())
    first_name = layers[0][0]
    container_name = first_name.rsplit(".", 1)[0]
    backbone_name = container_name.rsplit(".", 1)[0] if "." in container_name else ""
    backbone = modules[backbone_name]
    attention_types = {_attention_type(layer.attention_type) for _, layer in layers}
    factories = {
        "full_attention": (create_causal_mask, "rotary_emb"),
        "sliding_attention": (create_sliding_window_causal_mask, "swa_rotary_emb"),
    }
    if (
        "sliding_attention" in attention_types
        and getattr(backbone.config, "sliding_window", None) is None
    ):
        raise ValueError("MiMo sliding attention requires config.sliding_window")

    with _mimo_mask_compat(backbone):
        captured = capture_first_layer_inputs(model, layers[0][1], batches, device=device)
    if len(captured) != len(batches):
        raise RuntimeError("MiMo forward did not deliver every sample to the first decoder")
    samples: LayerSamples = []
    layer_kwargs: LayerKwargs = {name: [] for name, _ in layers}
    for batch, (args, captured_kwargs) in zip(batches, captured):
        batch = move_to_device(batch, torch.device("cpu"))
        options = dict(captured_kwargs)
        if not args:
            args = (options.pop("hidden_states"),)
        hidden = args[0].detach().cpu()
        args = (hidden, *args[1:])
        cache_position = options.get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(hidden.shape[1])
        position_ids = options.get("position_ids", batch.get("position_ids"))
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(hidden.shape[0], -1)
        by_type: dict[str, dict[str, Any]] = {}
        for attention_type in attention_types:
            mask_factory, rotary_name = factories[attention_type]
            rotary = getattr(backbone, rotary_name)
            mask_kwargs = {
                "config": backbone.config,
                "attention_mask": batch.get("attention_mask"),
                "past_key_values": None,
                "position_ids": position_ids,
            }
            mask_kwargs["inputs_embeds"] = hidden
            mask_kwargs["cache_position"] = cache_position
            raw_mask = batch.get("attention_mask")
            if isinstance(raw_mask, dict):
                masks = {_attention_type(key): value for key, value in raw_mask.items()}
                attention_mask = masks[attention_type]
            else:
                attention_mask = _compatible_mask(mask_factory)(**mask_kwargs)
            by_type[attention_type] = {
                **options,
                "attention_mask": attention_mask,
                "position_embeddings": rotary(hidden, position_ids),
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": None,
                "use_cache": False,
            }
        for name, layer in layers:
            layer_kwargs[name].append(dict(by_type[_attention_type(layer.attention_type)]))
        samples.append((args, layer_kwargs[layers[0][0]][-1]))
    return samples, layer_kwargs


def samples_with_layer_kwargs(
    samples: LayerSamples, kwargs: list[dict[str, Any]]
) -> LayerSamples:
    """Keep propagated hidden states, replacing only layer-specific inputs."""
    if len(samples) != len(kwargs):
        raise ValueError("Layer kwargs must match the number of calibration samples")
    return [(args, dict(options)) for (args, _), options in zip(samples, kwargs)]
