"""AutoAWQ-compatible scale search, clipping, and pseudo quantization."""

from __future__ import annotations

import copy

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class AWQTensorResult:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor | None


@torch.no_grad()
def pseudo_quantize_awq(
    weight: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
) -> AWQTensorResult:
    """Match AutoAWQ ``pseudo_quantize_tensor`` for a 2D weight."""
    if weight.dim() != 2:
        raise ValueError("AWQ requires a 2D weight")
    if bits != 4:
        raise ValueError("AutoAWQ GEMM currently supports 4-bit weights")
    original_shape = weight.shape
    if group_size <= 0:
        group_size = int(original_shape[-1])
    if original_shape[-1] % group_size:
        raise ValueError(
            f"in_features={original_shape[-1]} must be divisible by group_size={group_size}"
        )
    groups = weight.reshape(-1, group_size)
    if not torch.isfinite(groups).all().item():
        raise ValueError("Cannot AWQ-quantize NaN or infinity")

    if zero_point:
        maximum = groups.amax(dim=1, keepdim=True)
        minimum = groups.amin(dim=1, keepdim=True)
        max_int = (1 << bits) - 1
        scales = (maximum - minimum).clamp(min=1e-5) / max_int
        zeros = (-torch.round(minimum / scales)).clamp_(0, max_int)
        fake = (
            torch.clamp(torch.round(groups / scales) + zeros, 0, max_int) - zeros
        ) * scales
        zeros_out = zeros.view(original_shape[0], -1)
    else:
        maximum = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = (1 << (bits - 1)) - 1
        min_int = -(1 << (bits - 1))
        scales = maximum / max_int
        fake = torch.clamp(torch.round(groups / scales), min_int, max_int) * scales
        zeros_out = None

    return AWQTensorResult(
        weight=fake.reshape(original_shape),
        scales=scales.view(original_shape[0], -1),
        zeros=zeros_out,
    )


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Cannot extract a tensor from {type(output).__name__}")


def _mse_chunked(left: torch.Tensor, right: torch.Tensor, max_bytes: int) -> float:
    left = left.reshape(-1)
    right = right.reshape(-1)
    chunk_size = max(1, min(left.numel(), max_bytes // (left.element_size() * 2)))
    loss = 0.0
    for lchunk, rchunk in zip(
        torch.split(left, chunk_size), torch.split(right, chunk_size)
    ):
        loss += (lchunk.float() - rchunk.float()).pow(2).sum().item()
    return loss / max(1, left.numel())


@torch.no_grad()
def search_awq_scale(
    module: nn.Module,
    linears: Iterable[nn.Linear],
    inputs: torch.Tensor,
    *,
    kwargs: dict[str, Any] | None = None,
    group_size: int = 128,
    zero_point: bool = True,
    duo_scaling: bool = True,
    n_grid: int = 20,
    max_chunk_memory: int = 1024 * 1024 * 1024,
) -> torch.Tensor:
    """Run AutoAWQ's output-MSE grid search for one scaling relationship."""
    linears = list(linears)
    if not linears:
        raise ValueError("AWQ scale search requires at least one Linear")
    kwargs = dict(kwargs or {})
    kwargs.pop("use_cache", None)
    device = next(module.parameters()).device
    inputs = inputs.to(device)

    weight = torch.cat([linear.weight for linear in linears], dim=0)
    original_shape = weight.shape
    normalized = weight.view(-1, group_size)
    normalized = normalized.abs() / (normalized.abs().amax(dim=1, keepdim=True) + 1e-6)
    weight_mean = normalized.view(original_shape).mean(dim=0).float()
    input_mean = inputs.detach().abs().reshape(-1, inputs.shape[-1]).float().mean(dim=0)

    def forward():
        # Each candidate starts from the captured state, including caches that
        # the model creates even when use_cache=False.
        prepared = dict(kwargs)
        for name in ("past_key_values", "past_key_value", "shared"):
            if prepared.get(name) is not None:
                prepared[name] = copy.deepcopy(prepared[name])
        return _first_tensor(module(inputs, **prepared))

    reference = forward().detach()
    original_weights = [linear.weight.detach().clone() for linear in linears]
    best_error = float("inf")
    best_scales: torch.Tensor | None = None

    for grid_index in range(n_grid):
        ratio = grid_index / n_grid
        if duo_scaling:
            scales = (
                input_mean.pow(ratio) / (weight_mean.pow(1 - ratio) + 1e-4)
            ).clamp(min=1e-4)
        else:
            scales = input_mean.pow(ratio).clamp(min=1e-4)
        scales = scales / torch.sqrt(scales.max() * scales.min())
        scales[~torch.isfinite(scales)] = 1
        scale_view = scales.to(device).view(1, -1)

        try:
            for linear in linears:
                # Match apply_awq_scale's in-place storage rounding before
                # fake quantization, while retaining FP32 scale statistics.
                scaled = (linear.weight.data * scale_view).to(linear.weight.dtype)
                linear.weight.copy_(
                    pseudo_quantize_awq(
                        scaled,
                        bits=4,
                        group_size=group_size,
                        zero_point=zero_point,
                    ).weight
                    / scale_view
                )
            candidate = forward()
            error = _mse_chunked(reference, candidate, max_chunk_memory)
            if error < best_error:
                best_error = error
                best_scales = scales.detach().clone()
        finally:
            for linear, original in zip(linears, original_weights):
                linear.weight.copy_(original)

    if best_scales is None:
        raise RuntimeError("AWQ scale search found no finite candidate")
    return best_scales.cpu()


@torch.no_grad()
def search_awq_clip(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    *,
    group_size: int = 128,
    zero_point: bool = True,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    sample_tokens: int = 512,
    output_chunk_size: int = 256,
) -> torch.Tensor:
    """Return AutoAWQ per-output/per-group symmetric clipping maxima."""
    if weight.dim() != 2 or inputs.shape[-1] != weight.shape[1]:
        raise ValueError("AWQ clipping inputs do not match the weight")
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError("AWQ clipping group_size must divide in_features")
    features = inputs.reshape(-1, in_features)
    step = max(1, features.shape[0] // sample_tokens)
    features = features[::step].reshape(1, -1, in_features // group_size, group_size)
    grouped_weight = weight.reshape(out_features, 1, -1, group_size)
    results: list[torch.Tensor] = []

    for start in range(0, out_features, output_chunk_size):
        current = grouped_weight[start : start + output_chunk_size]
        maximum = current.abs().amax(dim=-1, keepdim=True)
        best_maximum = maximum.clone()
        minimum_error = torch.full_like(maximum, float("inf"))
        current_features = features.to(current.device)
        reference = (current_features * current).sum(dim=-1)
        for shrink_index in range(int(max_shrink * n_grid)):
            candidate_maximum = maximum * (1 - shrink_index / n_grid)
            clipped = torch.clamp(current, -candidate_maximum, candidate_maximum)
            fake = pseudo_quantize_awq(
                clipped.reshape(-1, group_size),
                bits=4,
                group_size=group_size,
                zero_point=zero_point,
            ).weight.reshape_as(clipped)
            candidate = (current_features * fake).sum(dim=-1)
            error = (candidate - reference).pow(2).mean(dim=1).reshape_as(minimum_error)
            better = error < minimum_error
            minimum_error[better] = error[better]
            best_maximum[better] = candidate_maximum[better]
        results.append(best_maximum)
    return torch.cat(results, dim=0).squeeze(1).detach().cpu()


@torch.no_grad()
def apply_awq_clip(weight: torch.Tensor, maximum: torch.Tensor) -> None:
    original_shape = weight.shape
    grouped = weight.reshape(*maximum.shape[:2], -1)
    grouped.clamp_(-maximum.to(weight.device), maximum.to(weight.device))
    weight.copy_(grouped.reshape(original_shape))


@torch.no_grad()
def apply_awq_scale(
    previous: nn.Module,
    linears: Iterable[nn.Module],
    scales: torch.Tensor,
) -> None:
    """Apply AutoAWQ's algebraically equivalent scale transform."""
    linears = list(linears)
    if not linears:
        raise ValueError("AWQ scaling requires at least one balance Linear")
    scales = scales.to(linears[0].weight.device)
    if isinstance(previous, nn.Linear):
        previous.weight[-scales.numel() :].div_(scales.view(-1, 1))
        if previous.bias is not None:
            previous.bias[-scales.numel() :].div_(scales)
    elif hasattr(previous, "weight") and previous.weight is not None:
        # Some RMSNorm implementations store a zero-centered parameter and use
        # ``1 + weight`` in forward.  Equalizing the raw parameter would break
        # the algebraic identity; transform its effective weight instead.
        class_name = previous.__class__.__name__.lower().replace("_", "")
        zero_centered_families = (
            "gemma",
            "minimaxm3",
            "qwen35",
            "qwen3next",
        )
        if any(token in class_name for token in zero_centered_families):
            previous.weight.add_(1).div_(scales).sub_(1)
        else:
            previous.weight.div_(scales)
        if getattr(previous, "bias", None) is not None:
            previous.bias.div_(scales)
    else:
        raise NotImplementedError(
            f"AWQ scaling does not support previous module {type(previous).__name__}"
        )
    for linear in linears:
        if linear.weight.dim() != 2 or linear.weight.shape[1] != scales.numel():
            raise ValueError(
                "AWQ balance weight must be 2D with input features matching scales"
            )
        linear.weight.mul_(scales.view(1, -1))


def should_skip_awq_clip(module_name: str) -> bool:
    return any(token in module_name for token in ("q_", "k_", "query", "key", "Wqkv"))


__all__ = [
    "AWQTensorResult",
    "apply_awq_clip",
    "apply_awq_scale",
    "pseudo_quantize_awq",
    "search_awq_clip",
    "search_awq_scale",
    "should_skip_awq_clip",
]
