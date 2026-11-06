"""Thin Transformers modeling integration for sequential calibration."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_transformers_model(
    model_path: str | Path,
    *,
    trust_remote_code: bool = False,
) -> tuple[nn.Module, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "GPTQ/AWQ/AutoRound model calibration requires transformers>=5,<6"
        ) from exc
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        low_cpu_mem_usage=False,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    from flagos_compressor.calibration.moe import linearize_fused_experts

    linearize_fused_experts(model)
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    return model, tokenizer


def decoder_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Discover ordered decoder blocks using the Transformers no-split contract."""
    try:
        class_names = set(model._get_no_split_modules("auto"))
    except AttributeError:
        class_names = set(getattr(model, "_no_split_modules", []))
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ in class_names
    ]
    # ``_no_split_modules`` may contain both an outer block and nested special
    # modules. Keep only modules that are not descendants of another candidate.
    names = {name for name, _ in candidates}
    result = [
        (name, module)
        for name, module in candidates
        if not any(name.startswith(parent + ".") for parent in names if parent != name)
    ]
    if not result:
        raise RuntimeError(
            "Transformers did not expose decoder layers through _no_split_modules; "
            "pass a supported Transformers model definition"
        )
    return result


def sanitize_kwargs(module: nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(module.forward).parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.values()
    ):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature}


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def prepare_forward_kwargs(
    module: nn.Module, kwargs: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Replay a calibration sample without mutating its captured KV cache.

    Some models (including DeepSeek-V4 compressed attention) create a cache
    even with use_cache=False. Reusing that object across GPTQ groups, AWQ
    candidates or AutoRound iterations appends the same sequence repeatedly.
    Preserve the captured cache type/configuration, but give each call its own
    copy. Setting it to None changes compressed-attention behavior.
    """
    prepared = sanitize_kwargs(module, kwargs)
    for name in ("past_key_values", "past_key_value", "shared"):
        if prepared.get(name) is not None:
            prepared[name] = copy.deepcopy(prepared[name])
    return move_to_device(prepared, device)


class _CapturedLayerInput(RuntimeError):
    pass


@torch.no_grad()
def capture_first_layer_inputs(
    model: nn.Module,
    first_layer: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Capture exact args/kwargs delivered by Transformers to its first block."""
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def hook(_module, args, kwargs):
        copied_args = tuple(
            item.detach().cpu() if isinstance(item, torch.Tensor) else item
            for item in args
        )
        copied_kwargs = {
            key: move_to_device(value, torch.device("cpu"))
            for key, value in kwargs.items()
        }
        captured.append((copied_args, copied_kwargs))
        raise _CapturedLayerInput()

    handle = first_layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for batch in batches:
            try:
                # Keep the model on CPU. The catcher aborts before the first
                # decoder block, so only embeddings/mask preparation execute.
                # Moving the full model to one GPU would defeat layer-wise calibration.
                model(
                    **move_to_device(batch, torch.device("cpu")),
                    use_cache=False,
                )
            except _CapturedLayerInput:
                pass
    finally:
        handle.remove()
    return captured


@torch.no_grad()
def forward_layer_samples(
    layer: nn.Module,
    samples: list[tuple[tuple[Any, ...], dict[str, Any]]],
    *,
    device: torch.device,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    outputs: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    layer.to(device)
    for args, kwargs in samples:
        moved_args = move_to_device(args, device)
        moved_kwargs = prepare_forward_kwargs(layer, kwargs, device)
        output = layer(*moved_args, **moved_kwargs)
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        outputs.append(((hidden.detach().cpu(),), kwargs))
    return outputs


__all__ = [
    "capture_first_layer_inputs",
    "decoder_layers",
    "forward_layer_samples",
    "load_transformers_model",
    "move_to_device",
    "prepare_forward_kwargs",
    "sanitize_kwargs",
]
