"""Thin Transformers modeling integration for sequential calibration."""

from __future__ import annotations

import inspect
from collections import Counter
from math import prod
from pathlib import Path
from typing import Any

import torch
from torch import nn


@torch.no_grad()
def _restore_source_fp32(model: nn.Module, model_path: str | Path) -> None:
    """Recover FP32 exceptions without undoing an intentional model-wide cast.

    Casting the loaded tensor back to FP32 cannot recover its original values.
    Restore from the checkpoint before calibration and expert linearization;
    native export will then retain the dtype through the model state dict.
    """
    from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint

    checkpoint = HfSafetensorsCheckpoint(model_path)
    state = model.state_dict(keep_vars=True)
    metadata_entries = list(checkpoint.iter_tensor_metadata())
    stored_elements = Counter()
    for name, _shard, metadata in metadata_entries:
        if name in state and state[name].is_floating_point():
            stored_elements[metadata["dtype"]] += prod(metadata["shape"])
    # A mixed checkpoint's small FP32 exceptions should survive loading. A
    # wholly FP32 checkpoint requested in BF16/FP16 is an intentional cast.
    reduced_elements = stored_elements["BF16"] + stored_elements["F16"]
    mixed_source = reduced_elements > stored_elements["F32"]
    declared = set(getattr(model, "_keep_in_fp32_modules", None) or ())
    declared.update(getattr(model, "_keep_in_fp32_modules_strict", None) or ())
    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    if getattr(config, "model_type", None) == "mimo_v2":
        declared.update(("attention_sink_bias", "e_score_correction_bias"))
    for name, _shard, metadata in metadata_entries:
        tensor = state.get(name)
        if metadata["dtype"] != "F32" or tensor is None:
            continue
        if tensor.dtype == torch.float32:
            continue
        explicitly_kept = any(f".{part}." in f".{name}." for part in declared)
        if not explicitly_kept and not (
            mixed_source and tensor.dtype in (torch.bfloat16, torch.float16)
        ):
            continue
        if list(tensor.shape) != metadata["shape"]:
            raise ValueError(f"Source FP32 tensor {name} changed shape during loading")
        tensor.data = checkpoint.load_tensor(name).to(device=tensor.device)


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
    _restore_source_fp32(model, model_path)
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
    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    is_mimo = getattr(config, "model_type", None) == "mimo_v2"
    try:
        class_names = set(model._get_no_split_modules("auto"))
    except AttributeError:
        class_names = set(getattr(model, "_no_split_modules", None) or ())
    except ValueError:
        if not is_mimo:
            raise
        class_names = set()
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
    if not result and is_mimo:
        # Older remote MiMo models omit no-split metadata. Locate the decoder
        # container, retaining registered module names for selectors/export.
        expected_count = getattr(config, "num_hidden_layers", None)
        containers = [
            module for _, module in model.named_modules()
            if isinstance(module, nn.ModuleList) and len(module)
            and (expected_count is None or len(module) == expected_count)
            and all(hasattr(layer, "attention_type") and hasattr(layer, "self_attn")
                    and hasattr(layer, "mlp") for layer in module)
        ]
        if len(containers) != 1:
            raise RuntimeError("Cannot uniquely identify MiMo decoder layers")
        names_by_id = {id(module): name for name, module in model.named_modules()}
        result = [(names_by_id[id(layer)], layer) for layer in containers[0]]
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
        moved_kwargs = sanitize_kwargs(layer, move_to_device(kwargs, device))
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
    "sanitize_kwargs",
]
