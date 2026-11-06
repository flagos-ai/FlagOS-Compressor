"""Discover direct AWQ consumers and check scaling before rounding any weights.

Observation supplements architecture mappings; it is not a symbolic graph proof.
The bounded forward check also catches unsupported functional branches/nonlinearities
on its recorded calibration samples. Neither check covers unseen execution paths.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
import weakref

import torch
from torch import nn

from .modeling import move_to_device, prepare_forward_kwargs


def _input_tensor(args, kwargs):
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    for key in ("input", "hidden_states", "x"):
        if isinstance(kwargs.get(key), torch.Tensor):
            return kwargs[key]
    return None


def _activation_key(value):
    # Only full, contiguous, channel-preserving aliases are recognized. Equal
    # values, copied tensors, permutations and in-place changes are not evidence
    # of a scaling relationship.
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim < 2 or not value.numel() or not value.is_contiguous()):
        return None
    try:
        version = value._version
    except RuntimeError:  # inference tensors have no version counter
        return None
    return (value.device, value.dtype, value.untyped_storage().data_ptr(),
            value.storage_offset(), value.numel(), value.shape[-1], version)


class AWQConsumerDiscovery:
    """Observe actual producer -> Linear connections during the capture pass."""

    def __init__(self, layer, mappings):
        self.layer = layer
        self.modules = dict(layer.named_modules())
        self.proposed = mappings
        self.live = defaultdict(list)
        self.consumers = defaultdict(set)
        self.uses = defaultdict(set)

    def _producer(self, name, output):
        key = _activation_key(output)
        if key is not None:
            # A weak reference prevents allocator address reuse from being
            # mistaken for an edge, without retaining all expert activations.
            self.live[key].append((name, weakref.ref(output)))

    def _consumer(self, name, args, kwargs):
        key = _activation_key(_input_tensor(args, kwargs))
        sources = {source for source, ref in self.live.get(key, ()) if ref() is not None}
        self.uses[name].update(sources or {None})
        for source in sources:
            self.consumers[source].add(name)

    @contextmanager
    def capture(self):
        handles = []
        try:
            handles.append(self.layer.register_forward_pre_hook(
                lambda *_: self.live.clear()))
            for name in {item.previous_name for item in self.proposed}:
                handles.append(self.modules[name].register_forward_hook(
                    lambda _m, _a, out, name=name: self._producer(name, out)))
            for name, module in self.modules.items():
                # Custom 2D weights may use a different orientation or nonlinear
                # forward. Retain explicit mappings for them; do not guess.
                if type(module) is nn.Linear:
                    handles.append(module.register_forward_pre_hook(
                        lambda _m, args, kwargs, name=name: self._consumer(name, args, kwargs),
                        with_kwargs=True))
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.live.clear()

    def resolve(self):
        result, records = [], []
        for mapping in self.proposed:
            width = self.modules[mapping.quantized_names[0]].weight.shape[1]
            direct = sorted(name for name in self.consumers[mapping.previous_name]
                            if self.modules[name].in_features == width)
            for name in direct:
                if self.uses[name] != {mapping.previous_name}:
                    raise RuntimeError(
                        f"AWQ cannot safely balance reused Linear {name}: it consumes "
                        f"both {mapping.previous_name} and other inputs. "
                        "An explicit model scaling adapter is required.")
            added = tuple(name for name in direct if name not in mapping.linear_names)
            result.append(replace(mapping, linear_names=mapping.linear_names + added))
            records.append({"previous": mapping.previous_name,
                            "quantized": list(mapping.quantized_names),
                            "balanced": list(result[-1].linear_names),
                            "discovered": list(added)})
        return result, records


def _tensor_leaves(value, path="output"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _tensor_leaves(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_leaves(item, f"{path}[{index}]")


class AWQScalingCheck:
    """Compare full tensor outputs on up to four evenly spaced sample blocks.

    References come from the existing activation collection forward. Additional
    forwards happen once after *all* scales, not once per expert/mapping. State
    passed to subsequent V4.1 layers in ``shared`` is checked as well.
    """

    def __init__(self, layer, samples, *, max_samples=4):
        if not samples:
            raise ValueError("AWQ scaling verification requires calibration samples")
        count = min(len(samples), max_samples)
        self.indices = sorted({round(i * (len(samples) - 1) / max(count - 1, 1))
                               for i in range(count)})
        self.references = {}
        # Tolerate low-precision multiply/divide rounding, not quantization
        # error: this check runs before clipping or weight quantization.
        dtypes = {p.dtype for p in layer.parameters() if p.is_floating_point()}
        if torch.bfloat16 in dtypes:
            self.rms_tolerance, self.max_tolerance = 0.02, 0.2
        elif torch.float16 in dtypes:
            self.rms_tolerance, self.max_tolerance = 0.002, 0.02
        else:
            self.rms_tolerance, self.max_tolerance = 1e-5, 1e-4
        self.report = {"state": "pending", "sample_indices": self.indices,
                       "total_samples": len(samples),
                       "scope": "full layer outputs and shared-state tensors on recorded samples",
                       "relative_rms_tolerance": self.rms_tolerance,
                       "max_error_over_rms_tolerance": self.max_tolerance,
                       "mappings": [], "applied_mappings": 0}

    def observe(self, index, output, kwargs):
        if index in self.indices:
            values = dict(_tensor_leaves((output, kwargs.get("shared"))))
            if not values:
                raise RuntimeError("AWQ cannot verify a layer without tensor outputs")
            self.references[index] = {name: value.detach().to("cpu", copy=True)
                                      for name, value in values.items()}

    def _compare(self, reference, value):
        if reference.shape != value.shape or reference.dtype != value.dtype:
            return False, {"reason": "shape_or_dtype_changed"}
        value = value.detach().cpu()
        if not reference.is_floating_point():
            return torch.equal(reference, value), {"reason": "discrete_output_changed"}
        if not torch.isfinite(reference).all() or not torch.isfinite(value).all():
            return False, {"reason": "non_finite_output"}
        # Work in chunks so the verification does not introduce another full
        # FP32 copy of a large activation tensor.
        reference, value = reference.reshape(-1), value.reshape(-1)
        squared_error, squared_reference, max_error = 0.0, 0.0, 0.0
        for start in range(0, reference.numel(), 1048576):
            ref = reference[start:start + 1048576].float()
            difference = value[start:start + 1048576].float() - ref
            squared_error += difference.square().sum(dtype=torch.float64).item()
            squared_reference += ref.square().sum(dtype=torch.float64).item()
            if difference.numel():
                max_error = max(max_error, difference.abs().max().item())
        rms = max((squared_reference / max(reference.numel(), 1)) ** 0.5, 1e-8)
        relative_rms = (squared_error / max(reference.numel(), 1)) ** 0.5 / rms
        normalized_max = max_error / rms
        return (relative_rms <= self.rms_tolerance and normalized_max <= self.max_tolerance), {
            "relative_rms_error": relative_rms, "max_error_over_rms": normalized_max}

    @torch.no_grad()
    def verify(self, layer, samples, device):
        self.report["checked_samples"] = 0
        self.report["max_relative_rms_error"] = 0.0
        try:
            for index in self.indices:
                args, kwargs = samples[index]
                prepared = prepare_forward_kwargs(layer, kwargs, device)
                output = layer(*move_to_device(args, device), **prepared)
                actual = dict(_tensor_leaves((output, prepared.get("shared"))))
                expected = self.references[index]
                if actual.keys() != expected.keys():
                    raise RuntimeError(f"sample {index}: tensor output structure changed")
                for name, reference in expected.items():
                    ok, error = self._compare(reference, actual[name])
                    self.report["max_relative_rms_error"] = max(
                        self.report["max_relative_rms_error"], error.get("relative_rms_error", 0.0))
                    if not ok:
                        self.report["failure"] = {"sample_index": index, "output": name, **error}
                        raise RuntimeError(f"sample {index}, {name}: {error}")
                self.report["checked_samples"] += 1
            self.report["state"] = "passed"
        except Exception as exc:
            self.report.update(state="failed", error=str(exc))
            raise RuntimeError(
                f"AWQ scaling changed the layer before quantization ({exc}). "
                "Layer weights will be restored; check missing consumers, nonlinear "
                "operations, shared parameters or stateful forward behavior. "
                "No quantized weights from this layer were produced.") from exc
        finally:
            self.references.clear()


class AWQScaleTransaction:
    """Restore exact pre-scaling parameters on any search/validation failure."""

    def __init__(self, layer):
        storages = defaultdict(list)
        for name, parameter in layer.named_parameters(remove_duplicate=False):
            if not parameter.numel():
                continue
            start = parameter.storage_offset() * parameter.element_size()
            stop = start + (1 + sum((size - 1) * stride for size, stride in
                                   zip(parameter.shape, parameter.stride()))) * parameter.element_size()
            storages[self._storage(parameter)].append((start, stop, name, id(parameter)))
        self.aliases = defaultdict(set)
        for intervals in storages.values():
            active = []
            for start, stop, name, identifier in sorted(intervals):
                active = [entry for entry in active if entry[0] > start]
                for _, other_name, other_id in active:
                    self.aliases[identifier].update((name, other_name))
                    self.aliases[other_id].update((name, other_name))
                active.append((stop, name, identifier))
        self.originals = {}

    @staticmethod
    def _storage(parameter):
        # Non-overlapping slices of fused MoE parameters are allowed.
        return parameter.device, parameter.untyped_storage().data_ptr()

    def remember(self, previous, linears):
        parameters = [previous.weight]
        if getattr(previous, "bias", None) is not None:
            parameters.append(previous.bias)
        parameters.extend(module.weight for module in linears)
        for parameter in parameters:
            if self.aliases[id(parameter)]:
                raise RuntimeError("AWQ scaling does not support aliased parameters: "
                                   + ", ".join(sorted(self.aliases[id(parameter)])[:4]))
            if id(parameter) not in self.originals:
                self.originals[id(parameter)] = (parameter, parameter.detach().to("cpu", copy=True))

    def __enter__(self):
        return self

    @torch.no_grad()
    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            for parameter, original in self.originals.values():
                parameter.copy_(original)
        self.originals.clear()
