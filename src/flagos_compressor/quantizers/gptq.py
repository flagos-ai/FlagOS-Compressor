"""AutoGPTQ-compatible second-order weight quantization.

The numerical flow intentionally follows AutoGPTQ's ``GPTQ.fasterquant``:
running-normalized Hessian accumulation, optional descending activation order,
static or dynamic groups, Cholesky-based error feedback, and ``g_idx`` recovery.
The implementation is torch-only and does not depend on AutoGPTQ kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GPTQResult:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    g_idx: torch.Tensor


def _find_params(
    weight: torch.Tensor,
    *,
    bits: int,
    symmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match AutoGPTQ ``Quantizer.find_params(..., weight=True)``."""
    if weight.dim() != 2:
        raise ValueError("GPTQ parameter search requires a 2D weight")
    maxq = (1 << bits) - 1
    zeros = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    xmin = torch.minimum(weight.min(dim=1).values, zeros)
    xmax = torch.maximum(weight.max(dim=1).values, zeros)
    if symmetric:
        xmax = torch.maximum(xmin.abs(), xmax)
        negative = xmin < 0
        xmin = torch.where(negative, -xmax, xmin)
    constant = (xmin == 0) & (xmax == 0)
    xmin = torch.where(constant, torch.full_like(xmin, -1), xmin)
    xmax = torch.where(constant, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / maxq
    if symmetric:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-xmin / scale)
    return scale.unsqueeze(1), zero.unsqueeze(1)


def _fake_quantize(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    *,
    bits: int,
) -> torch.Tensor:
    maxq = (1 << bits) - 1
    codes = torch.clamp(torch.round(weight / scale) + zero, 0, maxq)
    return scale * (codes - zero)


def _inverse_hessian_factor(hessian: torch.Tensor) -> torch.Tensor:
    """Return the upper factor U satisfying U.T @ U = inverse(hessian).

    With reversal matrix J, chol(J H J) = L gives H = T T.T for
    T = J L J. Thus U = inverse(T) is the desired upper factor.
    All backends use this factorization: it avoids forming inverse(H) and
    decomposing it again, as well as cholesky_inverse's torch_npu CPU fallback.
    Reversing both axes preserves GPTQ's original column order. Floating-point
    results need not be bitwise identical across devices or to the old method.
    """
    try:
        factor = torch.linalg.cholesky(hessian.flip((-2, -1)))
    except torch.linalg.LinAlgError as exc:
        raise RuntimeError(
            "GPTQ Hessian is not positive definite; increase damp_percent or "
            "use more calibration samples"
        ) from exc
    upper = factor.flip((-2, -1))
    identity = torch.eye(hessian.shape[-1], dtype=hessian.dtype, device=hessian.device)
    # Preserve unsupported-kernel and device errors instead of misdiagnosing
    # them as a non-positive Hessian. Singular solves may return NaN/Inf.
    result = torch.linalg.solve_triangular(upper, identity, upper=True)
    if not torch.isfinite(result).all():
        raise RuntimeError(
            "GPTQ Hessian factor contains NaN or infinity; increase damp_percent "
            "or use more calibration samples"
        )
    return result


class GPTQQuantizer:
    """Collect a Linear input Hessian and quantize one 2D weight matrix."""

    def __init__(self, weight: torch.Tensor, *, bits: int = 4, symmetric: bool = True):
        if weight.dim() != 2:
            raise ValueError(f"GPTQ requires a 2D weight, got {weight.dim()}D")
        if bits not in {4, 8}:
            raise ValueError("GPTQ currently supports 4-bit and 8-bit weights")
        self.weight = weight
        self.bits = bits
        self.symmetric = symmetric
        self.columns = int(weight.shape[1])
        self.hessian = torch.zeros(
            (self.columns, self.columns),
            dtype=torch.float32,
            device=weight.device,
        )
        self.num_samples = 0

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor) -> None:
        """Accumulate the exact running-normalized Hessian used by AutoGPTQ."""
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() < 2 or inputs.shape[-1] != self.columns:
            raise ValueError(
                f"Expected GPTQ inputs ending in {self.columns}, got {tuple(inputs.shape)}"
            )
        batch_samples = int(inputs.shape[0])
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        else:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        inputs = inputs.t()
        new_total = self.num_samples + batch_samples
        self.hessian.mul_(self.num_samples / new_total)
        self.num_samples = new_total
        normalized = math.sqrt(2 / self.num_samples) * inputs.float()
        self.hessian.add_(normalized.matmul(normalized.t()))

    @torch.no_grad()
    def quantize(
        self,
        *,
        block_size: int = 128,
        damp_percent: float = 0.01,
        group_size: int = -1,
        desc_act: bool = False,
        static_groups: bool = False,
    ) -> GPTQResult:
        if self.num_samples == 0:
            raise RuntimeError("GPTQ requires at least one calibration batch")
        if block_size <= 0 or not 0 < damp_percent < 1:
            raise ValueError("Invalid GPTQ block_size or damp_percent")
        if group_size == -1:
            group_size = self.columns
        if group_size <= 0 or self.columns % group_size:
            raise ValueError(
                f"in_features={self.columns} must be divisible by group_size={group_size}"
            )

        weight = self.weight.detach().float().clone()
        hessian = self.hessian.clone()
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        static_params: list[tuple[torch.Tensor, torch.Tensor]] = []
        scale_parts: list[torch.Tensor] = []
        zero_parts: list[torch.Tensor] = []
        if static_groups:
            for start in range(0, self.columns, group_size):
                params = _find_params(
                    weight[:, start : start + group_size],
                    bits=self.bits,
                    symmetric=self.symmetric,
                )
                static_params.append(params)
                scale_parts.append(params[0])
                zero_parts.append(params[1])

        permutation = torch.arange(self.columns, device=weight.device)
        if desc_act:
            permutation = torch.argsort(torch.diag(hessian), descending=True)
            weight = weight[:, permutation]
            hessian = hessian[permutation][:, permutation]
        inverse_permutation = torch.argsort(permutation)

        damp = damp_percent * torch.mean(torch.diag(hessian))
        diagonal = torch.arange(self.columns, device=weight.device)
        hessian[diagonal, diagonal] += damp
        hessian_inverse = _inverse_hessian_factor(hessian)

        quantized = torch.zeros_like(weight)
        current_params: tuple[torch.Tensor, torch.Tensor] | None = None
        for block_start in range(0, self.columns, block_size):
            block_end = min(block_start + block_size, self.columns)
            block_weight = weight[:, block_start:block_end].clone()
            block_error = torch.zeros_like(block_weight)
            block_inverse = hessian_inverse[
                block_start:block_end, block_start:block_end
            ]

            for offset in range(block_end - block_start):
                column_index = block_start + offset
                column = block_weight[:, offset]
                divisor = block_inverse[offset, offset]

                if static_groups:
                    original_index = int(permutation[column_index]) if desc_act else column_index
                    current_params = static_params[original_index // group_size]
                elif column_index % group_size == 0:
                    current_params = _find_params(
                        weight[:, column_index : column_index + group_size],
                        bits=self.bits,
                        symmetric=self.symmetric,
                    )
                    scale_parts.append(current_params[0])
                    zero_parts.append(current_params[1])

                assert current_params is not None
                scale, zero = current_params
                qcolumn = _fake_quantize(
                    column.unsqueeze(1),
                    scale,
                    zero,
                    bits=self.bits,
                ).flatten()
                quantized[:, column_index] = qcolumn
                error = (column - qcolumn) / divisor
                block_weight[:, offset:] -= error.unsqueeze(1).matmul(
                    block_inverse[offset, offset:].unsqueeze(0)
                )
                block_error[:, offset] = error

            weight[:, block_end:] -= block_error.matmul(
                hessian_inverse[block_start:block_end, block_end:]
            )

        if static_groups and desc_act:
            g_idx = permutation.to(torch.int32) // group_size
        else:
            g_idx = torch.arange(
                self.columns, device=quantized.device, dtype=torch.int32
            ) // group_size
        if desc_act:
            quantized = quantized[:, inverse_permutation]
            g_idx = g_idx[inverse_permutation]

        scales = torch.cat(scale_parts, dim=1)
        zeros = torch.cat(zero_parts, dim=1)
        result_weight = quantized.reshape(self.weight.shape).to(self.weight.dtype)
        return GPTQResult(
            weight=result_weight,
            scales=scales,
            zeros=zeros,
            g_idx=g_idx,
        )


__all__ = ["GPTQQuantizer", "GPTQResult"]
