"""Discover direct AWQ consumers and check scaling before rounding any weights.

Observation supplements architecture mappings; it is not a symbolic graph proof.
The bounded forward check also catches unsupported functional branches/nonlinearities
on its recorded calibration samples. Neither check covers unseen execution paths.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from itertools import product
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
        self.unobserved = defaultdict(int)
        self.producer_failures = set()

    def _producer(self, name, output):
        key = _activation_key(output)
        if key is not None:
            # A weak reference prevents allocator address reuse from being
            # mistaken for an edge, without retaining all expert activations.
            self.live[key].append((name, weakref.ref(output)))
        else:
            self.producer_failures.add(name)

    def _consumer(self, name, args, kwargs):
        key = _activation_key(_input_tensor(args, kwargs))
        sources = {source for source, ref in self.live.get(key, ()) if ref() is not None}
        self.uses[name].update(sources)
        if not sources:
            self.unobserved[name] += 1
        for source in sources:
            self.consumers[source].add(name)

    @contextmanager
    def capture(self):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("AWQ consumer discovery requires tensor version counters; "
                               "run calibration with torch.no_grad(), outside inference_mode")
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
            balanced = mapping.linear_names + added
            reasons = []
            if mapping.previous_name in self.producer_failures:
                reasons.append("producer activation identity/version was unavailable")
            for name in direct:
                if self.unobserved[name]:
                    reasons.append(f"{name}: {self.unobserved[name]} calls had an unobserved source")
            # A registered Linear that was never called may be used through
            # F.linear(weight). Its output cannot be checked by a module hook.
            for name, module in self.modules.items():
                if (name.startswith(mapping.inspect_name + ".")
                        and type(module) is nn.Linear and module.in_features == width
                        and name not in self.uses):
                    reasons.append(f"{name}: no module call observed (possible functional branch)")
            record = {"previous": mapping.previous_name,
                      "quantized": list(mapping.quantized_names),
                      "balanced": list(balanced), "discovered": list(added),
                      "state": "skipped" if reasons else "eligible",
                      "reasons": reasons}
            records.append(record)
            if not reasons:
                result.append(replace(mapping, linear_names=balanced))
        owners = {}
        for mapping in result:
            for name in mapping.linear_names:
                previous = owners.setdefault(name, mapping.previous_name)
                if previous != mapping.previous_name:
                    raise RuntimeError(f"AWQ balance consumer {name} is claimed by both "
                                       f"{previous} and {mapping.previous_name}")
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


def _state_leaves(value, path, seen=None):
    """Include mutable cache/shared tensor state and scalar counters, cycle-safe."""
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (bool, int, float)):
        yield path, torch.tensor(value)
    elif value is not None and id(value) not in seen:
        seen.add(id(value))
        if isinstance(value, Mapping):
            children = value.items()
        elif isinstance(value, (list, tuple)):
            children = enumerate(value)
        elif hasattr(value, "__dict__") and not isinstance(value, nn.Module):
            children = vars(value).items()
        else:
            return
        for key, item in children:
            yield from _state_leaves(item, f"{path}.{key}", seen)


def _checked_outputs(output, kwargs):
    values = dict(_tensor_leaves(output))
    for name in ("shared", "past_key_value", "past_key_values"):
        values.update(_state_leaves(kwargs.get(name), name))
    return values


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
        self.branches_captured = False
        self.branch_references = {}
        self.branch_exclusions = set()
        self.branch_values = {}
        self.branch_shapes = {}
        self.branch_counts = defaultdict(int)
        self.current_index = None
        self.report = {"state": "pending", "sample_indices": self.indices,
                       "total_samples": len(samples),
                       "scope": "recorded layer outputs, cache/shared state and sampled Linear branch outputs; not a graph proof",
                       "branch_elements_per_call": 2048,
                       "tolerances_by_output_dtype": {
                           "bfloat16": {"relative_rms": 0.02, "max_over_rms": 0.2},
                           "float16": {"relative_rms": 0.002, "max_over_rms": 0.02},
                           "float32": {"relative_rms": 1e-5, "max_over_rms": 1e-4}},
                       "mappings": [], "applied_mappings": 0}

    def begin_sample(self, index):
        self.current_index = index
        self.branch_values = {}
        self.branch_shapes = {}
        self.branch_counts.clear()

    @contextmanager
    def capture_branches(self, layer):
        self.branches_captured = True
        handles = []
        def observe_branch(name, output):
            if self.current_index not in self.indices:
                return
            call = self.branch_counts[name]
            self.branch_counts[name] += 1
            for path, value in _tensor_leaves(output, f"branch.{name}[{call}]"):
                self.branch_shapes[path] = tuple(value.shape)
                # Bound extra reference storage even for large hidden widths.
                flat = value.detach().reshape(-1)
                step = max(1, (flat.numel() + 2047) // 2048)
                self.branch_values[path] = flat[::step].to("cpu", copy=True)
        try:
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    handles.append(module.register_forward_hook(
                        lambda _m, _a, out, name=name: observe_branch(name, out)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def observe(self, index, output, kwargs):
        if index in self.indices:
            values = _checked_outputs(output, kwargs)
            if not values:
                raise RuntimeError("AWQ cannot verify a layer without tensor outputs")
            self.references[index] = {name: value.detach().to("cpu", copy=True)
                                      for name, value in values.items()}
            self.branch_references[index] = (dict(self.branch_shapes), dict(self.branch_values))

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
        # Use each output's dtype, rather than a loose BF16 tolerance merely
        # because some unrelated parameter in the layer happens to be BF16.
        rms_tolerance, max_tolerance = {
            torch.bfloat16: (0.02, 0.2), torch.float16: (0.002, 0.02),
        }.get(reference.dtype, (1e-5, 1e-4))
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
        return (relative_rms <= rms_tolerance and normalized_max <= max_tolerance), {
            "relative_rms_error": relative_rms, "max_error_over_rms": normalized_max}

    @torch.no_grad()
    def verify(self, layer, samples, device):
        self.report["checked_samples"] = 0
        self.report["max_relative_rms_error"] = 0.0
        self.report["checked_branch_calls"] = 0
        try:
            with self.capture_branches(layer) if self.branches_captured else nullcontext():
                self._verify_samples(layer, samples, device)
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
            self.branch_references.clear()
            self.branch_values.clear()

    def _verify_samples(self, layer, samples, device):
        for index in self.indices:
            self.begin_sample(index)
            args, kwargs = samples[index]
            prepared = prepare_forward_kwargs(layer, kwargs, device)
            output = layer(*move_to_device(args, device), **prepared)
            actual = _checked_outputs(output, prepared)
            expected = dict(self.references[index])
            shapes, branches = self.branch_references.get(index, ({}, {}))
            if shapes != self.branch_shapes:
                raise RuntimeError(f"sample {index}: Linear branch call structure changed")
            # Only producers of transforms actually applied may change their
            # output coordinates. A skipped mapping must not hide a branch.
            def invariant(path):
                return path.removeprefix("branch.").split("[", 1)[0] not in self.branch_exclusions
            actual.update((path, value) for path, value in self.branch_values.items() if invariant(path))
            expected.update((path, value) for path, value in branches.items() if invariant(path))
            self.report["checked_branch_calls"] += sum(self.branch_counts.values())
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


def _storage_runs(parameter):
    """Sorted byte runs for ordinary strided views, without element-sized sets.

    Unknown/internal-overlap layouts return None and remain conservative.
    Contiguous dimensions coalesce, so transposed fused slices need one run
    per storage row rather than one entry per weight element.
    """
    size = parameter.element_size()
    block, extent, outer = size, size, []
    for stride, count in sorted((stride * size, count) for count, stride in
                                zip(parameter.shape, parameter.stride()) if count > 1):
        if stride < extent:
            return None
        if stride == block and not outer:
            block *= count
        else:
            outer.append((stride, count))
        extent += stride * (count - 1)
    base = parameter.storage_offset() * size
    return ((base + sum(i * stride for i, (stride, _) in zip(indices, reversed(outer))),
             base + sum(i * stride for i, (stride, _) in zip(indices, reversed(outer))) + block)
            for indices in product(*(range(count) for _, count in reversed(outer))))


def _storage_overlaps(left, right):
    left_runs, right_runs = _storage_runs(left), _storage_runs(right)
    if left_runs is None or right_runs is None:
        return True
    a, b = next(left_runs, None), next(right_runs, None)
    while a is not None and b is not None:
        if a[0] < b[1] and b[0] < a[1]:
            return True
        if a[1] <= b[0]:
            a = next(left_runs, None)
        else:
            b = next(right_runs, None)
    return False


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
            storages[self._storage(parameter)].append((start, stop, name, parameter))
        self.aliases = defaultdict(set)
        for intervals in storages.values():
            active = []
            for start, stop, name, parameter in sorted(intervals, key=lambda item: item[:3]):
                active = [entry for entry in active if entry[0] > start]
                for _, other_name, other in active:
                    if _storage_overlaps(parameter, other):
                        self.aliases[id(parameter)].update((name, other_name))
                        self.aliases[id(other)].update((name, other_name))
                active.append((stop, name, parameter))
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
