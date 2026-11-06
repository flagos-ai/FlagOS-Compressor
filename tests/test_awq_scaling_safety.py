import copy
import json

import pytest
import torch
from torch import nn

from flagos_compressor.calibration.awq_safety import (
    AWQConsumerDiscovery, AWQScalingCheck, AWQScaleTransaction,
)
from flagos_compressor.calibration.mappings import infer_awq_mappings
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.core.policy import AWQPolicy, QuantizationPolicy
from flagos_compressor.quantizers.awq import apply_awq_scale


class AttentionWithExtraBranch(nn.Module):
    def __init__(self, mode="direct"):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.auxiliary = nn.Linear(8, 8, bias=False)
        self.mode = mode

    def forward(self, hidden_states):
        if self.mode == "functional":
            extra = nn.functional.linear(hidden_states, self.auxiliary.weight)
        else:
            extra_input = hidden_states
            if self.mode == "view":
                extra_input = hidden_states.view(-1, 8)
            elif self.mode == "clone":
                extra_input = hidden_states.clone()
            extra = self.auxiliary(input=extra_input).reshape_as(hidden_states)
        return self.q_proj(hidden_states) + extra


class BranchedBlock(nn.Module):
    def __init__(self, mode="direct"):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = AttentionWithExtraBranch(mode)

    def forward(self, hidden_states, shared=None):
        if shared is not None:
            assert shared["calls"] == 0
            shared["calls"] += 1
        return self.self_attn(self.input_layernorm(hidden_states))


def policy():
    return QuantizationPolicy(selections=(), include_names=(r"self_attn\.q_proj\.weight$",),
                              method="awq", group_size=8, awq=AWQPolicy(n_grid=2))


def force_scale(monkeypatch):
    # Isolate mapping correctness from the scale search: every test exercises
    # an actual non-identity transform and the real rounding/packing path.
    monkeypatch.setattr("flagos_compressor.calibration.runner.search_awq_scale",
                        lambda *args, **kwargs: torch.tensor([0.5, 2.0] * 4))


@pytest.mark.parametrize("mode", ["direct", "view"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_new_floating_branch_is_balanced_but_not_quantized(tmp_path, monkeypatch, mode, dtype):
    torch.manual_seed(3)
    block = BranchedBlock(mode).to(dtype).eval()
    original = block.self_attn.auxiliary.weight.detach().clone()
    sample = ((torch.randn(1, 6, 8).to(dtype),), {"shared": {"calls": 0}})
    force_scale(monkeypatch)
    path = tmp_path / "report.json"
    result = quantize_model_sequential([("model.layers.0", block)], [sample], policy(),
                                      device=torch.device("cpu"), awq_report_path=path)
    assert set(result) == {"model.layers.0.self_attn.q_proj"}
    torch.testing.assert_close(block.self_attn.auxiliary.weight,
                               original * torch.tensor([0.5, 2.0] * 4).to(dtype), rtol=0, atol=0)
    report = json.loads(path.read_text())["layers"][0]["awq_scaling"]
    assert report["state"] == "passed"
    assert report["checked_samples"] == 1
    assert report["mappings"][0]["discovered"] == ["self_attn.auxiliary"]
    assert sample[1]["shared"] == {"calls": 0}


@pytest.mark.parametrize("mode", ["functional", "clone"])
def test_unrecognized_branch_fails_before_packing_and_restores_weights(tmp_path, monkeypatch, mode):
    torch.manual_seed(4)
    block = BranchedBlock(mode).eval()
    original = copy.deepcopy(block.state_dict())
    force_scale(monkeypatch)
    def no_packing(*args, **kwargs):
        pytest.fail("unsafe scaling reached weight packing")
    monkeypatch.setattr("flagos_compressor.calibration.runner.pack_autoawq_gemm", no_packing)
    path = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="scaling changed the layer before quantization"):
        quantize_model_sequential([("model.layers.0", block)],
                                 [((torch.randn(1, 6, 8),), {})], policy(),
                                 device=torch.device("cpu"), awq_report_path=path)
    for name, tensor in block.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    report = json.loads(path.read_text())
    assert report["state"] == "failed"
    assert report["layers"][0]["awq_scaling"]["weights_restored"]
    assert "failure" in report["layers"][0]["awq_scaling"]
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in block.modules())


def test_discovery_rejects_linear_reused_on_scaled_and_unscaled_input():
    class ReusedBlock(BranchedBlock):
        def forward(self, hidden_states):
            return super().forward(hidden_states) + self.self_attn.auxiliary(hidden_states)
    block = ReusedBlock().eval()
    discovery = AWQConsumerDiscovery(block, infer_awq_mappings(block, {"self_attn.q_proj"}))
    with discovery.capture():
        block(torch.randn(1, 3, 8))
    with pytest.raises(RuntimeError, match="reused Linear self_attn.auxiliary"):
        discovery.resolve()


def test_discovery_observes_conditional_branch_after_first_sample():
    class ConditionalAttention(AttentionWithExtraBranch):
        def forward(self, hidden_states):
            output = self.q_proj(hidden_states)
            if hidden_states[0, 0, 0] > 0:
                output = output + self.auxiliary(hidden_states)
            return output
    block = BranchedBlock().eval()
    block.self_attn = ConditionalAttention()
    discovery = AWQConsumerDiscovery(block, infer_awq_mappings(block, {"self_attn.q_proj"}))
    values = torch.tensor([[[-7., 1, 1, 1, 1, 1, 1, 1]]])
    with discovery.capture():
        block(values)
        block(-values)
    _, records = discovery.resolve()
    assert records[0]["discovered"] == ["self_attn.auxiliary"]


def test_scaled_norm_shared_by_attention_and_mlp_discovers_both():
    class ParallelBlock(BranchedBlock):
        def __init__(self):
            super().__init__()
            self.parallel_mlp = nn.Linear(8, 8, bias=False)
        def forward(self, hidden_states):
            x = self.input_layernorm(hidden_states)
            return self.self_attn(x) + self.parallel_mlp(x)
    block = ParallelBlock().eval()
    discovery = AWQConsumerDiscovery(block, infer_awq_mappings(block, {"self_attn.q_proj"}))
    with discovery.capture():
        block(torch.randn(1, 4, 8))
    resolved, _ = discovery.resolve()
    assert set(resolved[0].linear_names) == {
        "self_attn.q_proj", "self_attn.auxiliary", "parallel_mlp"}


def test_custom_linear_forward_is_not_assumed_to_be_linear():
    class SquaredWeight(nn.Linear):
        def forward(self, input):
            return nn.functional.linear(input, self.weight.square())
    block = BranchedBlock().eval()
    block.self_attn.auxiliary = SquaredWeight(8, 8, bias=False)
    discovery = AWQConsumerDiscovery(block, infer_awq_mappings(block, {"self_attn.q_proj"}))
    with discovery.capture():
        block(torch.randn(1, 3, 8))
    _, records = discovery.resolve()
    assert records[0]["discovered"] == []


def test_transaction_rejects_tied_parameters_but_allows_disjoint_expert_slices():
    block = BranchedBlock()
    bank = torch.randn(2, 8, 8)
    block.self_attn.q_proj.weight = nn.Parameter(bank[0])
    block.self_attn.auxiliary.weight = nn.Parameter(bank[1])
    with AWQScaleTransaction(block) as transaction:
        transaction.remember(block.input_layernorm, [block.self_attn.q_proj, block.self_attn.auxiliary])
    block.self_attn.auxiliary.weight = nn.Parameter(bank[0])
    with pytest.raises(RuntimeError, match="aliased parameters"):
        with AWQScaleTransaction(block) as transaction:
            transaction.remember(block.input_layernorm, [block.self_attn.q_proj])


def test_transaction_restores_all_earlier_transforms_on_later_failure():
    block = BranchedBlock()
    original = copy.deepcopy(block.state_dict())
    with pytest.raises(RuntimeError, match="later search failed"):
        with AWQScaleTransaction(block) as transaction:
            transaction.remember(block.input_layernorm, [block.self_attn.q_proj])
            apply_awq_scale(block.input_layernorm, [block.self_attn.q_proj], torch.full((8,), 2.))
            raise RuntimeError("later search failed")
    for name, tensor in block.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


def test_check_covers_later_sample_and_shared_state_even_when_output_is_unchanged():
    class StateBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(8))
        def forward(self, x, shared):
            assert not shared
            shared["later_layer_input"] = x * self.scale
            return torch.zeros_like(x)
    from flagos_compressor.calibration.runner import _run_samples
    block = StateBlock().eval()
    samples = [((torch.ones(1, 2, 8) * i,), {"shared": {}}) for i in range(5)]
    check = AWQScalingCheck(block, samples)
    _run_samples(block, samples, torch.device("cpu"), after_forward=check.observe)
    with torch.no_grad():
        block.scale.mul_(2)
    with pytest.raises(RuntimeError, match="scaling changed"):
        check.verify(block, samples, torch.device("cpu"))
    assert check.report["failure"]["sample_index"] > 0
    assert "later_layer_input" in check.report["failure"]["output"]
    assert all(not kwargs["shared"] for _, kwargs in samples)


def test_check_rejects_unknown_activation_clamp():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(8, 8, bias=False)
            self.down_proj = nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.up_proj.weight.copy_(torch.eye(8))
                self.down_proj.weight.copy_(torch.eye(8))
        def forward(self, x):
            # No known limit/swiglu_limit attribute: name matching cannot see it.
            return self.down_proj(self.up_proj(x).clamp(-1, 1))
    from flagos_compressor.calibration.runner import _run_samples
    block = nn.Sequential(MLP()).eval()
    samples = [((torch.full((1, 2, 8), 1.5),), {})]
    check = AWQScalingCheck(block, samples)
    _run_samples(block, samples, torch.device("cpu"), after_forward=check.observe)
    apply_awq_scale(block[0].up_proj, [block[0].down_proj], torch.full((8,), 2.))
    with pytest.raises(RuntimeError, match="scaling changed"):
        check.verify(block, samples, torch.device("cpu"))


@pytest.mark.parametrize("family", ["Llama", "Gemma", "Qwen3"])
def test_partial_attention_selection_keeps_unselected_kv_correct(tmp_path, family):
    import transformers
    from flagos_compressor.calibration.modeling import capture_first_layer_inputs, decoder_layers
    torch.manual_seed(8)
    config_cls = getattr(transformers, family + "Config")
    model_cls = getattr(transformers, family + "ForCausalLM")
    model = model_cls(config_cls(hidden_size=16, intermediate_size=32, head_dim=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        vocab_size=32, max_position_embeddings=32, use_cache=False)).eval()
    layers = decoder_layers(model)
    samples = capture_first_layer_inputs(model, layers[0][1],
        [{"input_ids": torch.randint(0, 32, (1, 8))}], device=torch.device("cpu"))
    path = tmp_path / "llama.json"
    result = quantize_model_sequential(layers, samples, policy(), device=torch.device("cpu"), awq_report_path=path)
    assert len(result) == 1
    report = json.loads(path.read_text())["layers"][0]["awq_scaling"]
    assert report["state"] == "passed"
    assert set(report["mappings"][0]["discovered"]) == {"self_attn.k_proj", "self_attn.v_proj"}


def test_latent_query_branch_is_discovered_without_a_deepseek_class_name():
    class CustomLatentAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_a_proj = nn.Linear(8, 8, bias=False)
            self.q_a_norm = nn.LayerNorm(8)
            self.q_b_proj = nn.Linear(8, 8, bias=False)
            self.extra_query = nn.Linear(8, 8, bias=False)
        def forward(self, x):
            latent = self.q_a_norm(self.q_a_proj(x))
            return self.q_b_proj(latent) + self.extra_query(latent)
    from flagos_compressor.calibration.runner import _run_samples
    block = BranchedBlock().eval()
    block.self_attn = CustomLatentAttention()
    samples = [((torch.randn(1, 4, 8),), {})]
    mappings = infer_awq_mappings(block, {"self_attn.q_a_proj", "self_attn.q_b_proj"})
    discovery = AWQConsumerDiscovery(block, mappings)
    check = AWQScalingCheck(block, samples)
    with discovery.capture():
        _run_samples(block, samples, torch.device("cpu"), after_forward=check.observe)
    mappings, records = discovery.resolve()
    assert records[1]["discovered"] == ["self_attn.extra_query"]
    modules = dict(block.named_modules())
    for mapping in mappings:
        apply_awq_scale(modules[mapping.previous_name], [modules[n] for n in mapping.linear_names],
                        torch.tensor([0.5, 2.] * 4))
    check.verify(block, samples, torch.device("cpu"))
    assert check.report["state"] == "passed"
