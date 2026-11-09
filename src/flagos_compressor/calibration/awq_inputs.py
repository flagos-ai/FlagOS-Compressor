"""Replay AWQ inputs with the arguments from each actual submodule call."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _copy_arguments(value: Any, device: torch.device, memo=None) -> Any:
    """Snapshot call state, moving tensors inside Python cache objects too."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device, copy=True)
    elif isinstance(value, (dict, list)):
        result = {} if isinstance(value, dict) else []
        memo[id(value)] = result
        if isinstance(value, dict):
            result.update((key, _copy_arguments(item, device, memo)) for key, item in value.items())
        else:
            result.extend(_copy_arguments(item, device, memo) for item in value)
    elif isinstance(value, tuple):
        result = tuple(_copy_arguments(item, device, memo) for item in value)
    elif hasattr(value, "__dict__") and not isinstance(value, (nn.Module, type)):
        result = copy.copy(value)
        if result is value:
            return copy.deepcopy(value, memo)
        memo[id(value)] = result
        for name, item in vars(value).items():
            vars(result)[name] = _copy_arguments(item, device, memo)
    else:
        result = copy.deepcopy(value, memo)
    memo[id(value)] = result
    return result


@dataclass
class AWQCall:
    rows: int
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    input_key: str | None

    @classmethod
    def capture(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> AWQCall:
        options = dict(kwargs)
        if args and isinstance(args[0], torch.Tensor):
            hidden, remaining, key = args[0], args[1:], None
        else:
            key = next((name for name in ("hidden_states", "input", "x")
                        if isinstance(options.get(name), torch.Tensor)), None)
            if key is None:
                raise TypeError("AWQ cannot identify the submodule's tensor input")
            hidden, remaining = options.pop(key), args
        return cls(hidden.shape[0], _copy_arguments(remaining, torch.device("cpu")),
                   _copy_arguments(options, torch.device("cpu")), key)


class AWQReplay(nn.Module):
    """Split merged features back into their original forward calls.

    Position embeddings, masks and token IDs need not share batch shapes. Keep
    their original shapes and calling convention instead of guessing how to
    concatenate them. The input tensor is supplied by scale search, so earlier
    scale adjustments to captured features are respected.
    """

    def __init__(self, module: nn.Module, calls: list[AWQCall]):
        super().__init__()
        self.module = module
        self.calls = calls

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.calls or sum(call.rows for call in self.calls) != inputs.shape[0]:
            raise ValueError("AWQ captured calls do not match the feature rows")
        outputs = []
        offset = 0
        for call in self.calls:
            hidden = inputs[offset:offset + call.rows]
            offset += call.rows
            args = _copy_arguments(call.args, inputs.device)
            kwargs = _copy_arguments(call.kwargs, inputs.device)
            if call.input_key is None:
                output = self.module(hidden, *args, **kwargs)
            else:
                kwargs[call.input_key] = hidden
                output = self.module(*args, **kwargs)
            if isinstance(output, (tuple, list)):
                output = output[0]
            elif hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            if not isinstance(output, torch.Tensor):
                raise TypeError("AWQ inspected module must return a tensor output")
            outputs.append(output)
        return torch.cat(outputs, dim=0)
