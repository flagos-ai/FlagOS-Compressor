"""Native, device-agnostic AutoRound fake quantization primitives.

The implementation mirrors AutoRound's symmetric group-quantization semantics:
a learnable rounding offset and optional learnable minimum/maximum shrink factors
are optimized with straight-through rounding and sign-based updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AutoRoundResult:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor


def round_ste(value: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass and use an identity backward pass."""
    return (torch.round(value) - value).detach() + value


def _grouped(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("AutoRound currently supports two-dimensional weights")
    if group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError(
            f"in_features={weight.shape[1]} must be divisible by group_size={group_size}"
        )
    return weight.reshape(weight.shape[0], -1, group_size)


def fake_quantize_symmetric(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    value: torch.Tensor,
    min_scale: torch.Tensor,
    max_scale: torch.Tensor,
    weight_min: torch.Tensor | None = None,
    weight_max: torch.Tensor | None = None,
) -> AutoRoundResult:
    """Apply AutoRound-compatible symmetric group-wise QDQ."""
    if bits not in {4, 8}:
        raise ValueError("AutoRound supports 4-bit and 8-bit weights")
    grouped = _grouped(weight.float(), group_size)
    grouped_value = _grouped(value.float(), group_size)
    if min_scale.shape != grouped.shape[:2] or max_scale.shape != grouped.shape[:2]:
        raise ValueError("AutoRound min/max scale shapes must match weight groups")

    minimum = (
        grouped.amin(dim=-1).clamp(max=0)
        if weight_min is None
        else weight_min
    )
    maximum = (
        grouped.amax(dim=-1).clamp(min=0)
        if weight_max is None
        else weight_max
    )
    if minimum.shape != grouped.shape[:2] or maximum.shape != grouped.shape[:2]:
        raise ValueError("AutoRound cached weight ranges must match weight groups")
    minimum_abs = -(minimum * min_scale.clamp(0, 1))
    maximum_abs = maximum * max_scale.clamp(0, 1)
    choose_minimum = maximum_abs < minimum_abs
    signed_maximum = torch.where(choose_minimum, -minimum_abs, maximum_abs)
    maxq = 1 << (bits - 1)
    # AutoRound defaults to FP16 scales and uses this floor to avoid unstable
    # reciprocal values during optimization.
    epsilon = 1e-5
    signed_maximum = torch.where(
        signed_maximum.abs() < epsilon,
        torch.full_like(signed_maximum, epsilon),
        signed_maximum,
    )
    scales = signed_maximum / maxq
    quantized = round_ste(grouped / scales.unsqueeze(-1) + grouped_value)
    quantized = quantized.clamp(-maxq, maxq - 1)
    dequantized = quantized * scales.unsqueeze(-1)
    zeros = torch.full_like(scales, maxq)
    return AutoRoundResult(
        weight=dequantized.reshape_as(weight).to(weight.dtype),
        scales=scales,
        zeros=zeros,
    )


class AutoRoundLinear(nn.Module):
    """Temporary linear wrapper exposing AutoRound's trainable parameters."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        bits: int,
        group_size: int,
        enable_minmax_tuning: bool,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.bits = bits
        self.group_size = group_size
        self.num_forwards = 0
        self.num_optimization_rows = 0
        groups = linear.in_features // group_size
        self.value = nn.Parameter(torch.zeros_like(linear.weight, dtype=torch.float32))
        self.min_scale = nn.Parameter(
            torch.ones(
                linear.out_features,
                groups,
                device=linear.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=enable_minmax_tuning,
        )
        self.max_scale = nn.Parameter(
            torch.ones(
                linear.out_features,
                groups,
                device=linear.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=enable_minmax_tuning,
        )
        grouped = _grouped(linear.weight.detach().float(), group_size)
        self.register_buffer(
            "weight_min",
            grouped.amin(dim=-1).clamp(max=0),
        )
        self.register_buffer(
            "weight_max",
            grouped.amax(dim=-1).clamp(min=0),
        )

    def quantized(self) -> AutoRoundResult:
        return fake_quantize_symmetric(
            self.linear.weight,
            bits=self.bits,
            group_size=self.group_size,
            value=self.value,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            weight_min=self.weight_min,
            weight_max=self.weight_max,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.num_forwards += 1
        if torch.is_grad_enabled():
            self.num_optimization_rows += inputs.numel() // inputs.shape[-1]
        quantized = self.quantized()
        return F.linear(inputs, quantized.weight, self.linear.bias)


class SignSGD(torch.optim.Optimizer):
    """SignSGD update used by the AutoRound reference implementation."""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        super().__init__(params, {"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            momentum = float(group["momentum"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                direction = parameter.grad
                if momentum:
                    state = self.state[parameter]
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = state["momentum_buffer"] = direction.clone()
                    else:
                        buffer.mul_(momentum).add_(direction)
                    direction = buffer
                parameter.add_(direction.sign(), alpha=-float(group["lr"]))
        return loss


__all__ = [
    "AutoRoundLinear",
    "AutoRoundResult",
    "SignSGD",
    "fake_quantize_symmetric",
    "round_ste",
]
