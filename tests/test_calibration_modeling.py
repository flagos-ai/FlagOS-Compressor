import torch

from flagos_compressor.calibration.data import build_calibration_batches
from flagos_compressor.calibration.mappings import infer_awq_mappings
from flagos_compressor.calibration.modeling import (
    capture_first_layer_inputs,
    decoder_layers,
)
from flagos_compressor.calibration.moe import LinearExperts2D, linearize_fused_experts
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.core.policy import (
    AWQPolicy,
    AutoRoundPolicy,
    CalibrationPolicy,
    GPTQPolicy,
    QuantizationPolicy,
)


class _Tokenizer:
    def encode(self, text, **_kwargs):
        return [ord(character) % 31 + 1 for character in text]


def test_calibration_data_is_packed_into_fixed_blocks():
    policy = CalibrationPolicy(
        data=("abcdefgh", "ijklmnop", "qrstuvwx"),
        samples=3,
        sequence_length=8,
    )
    batches = build_calibration_batches(_Tokenizer(), policy)

    assert len(batches) == 3
    assert all(batch["input_ids"].shape == (1, 8) for batch in batches)
    assert all(torch.all(batch["attention_mask"] == 1) for batch in batches)


def test_transformers_v5_fused_experts_linearize_exactly():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    torch.manual_seed(7)
    config = Qwen3MoeConfig(
        hidden_size=8,
        moe_intermediate_size=4,
        num_experts=2,
    )
    original = Qwen3MoeExperts(config).eval()
    # Standalone experts allocate torch.empty weights; no model post_init runs.
    for parameter in original.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.1)
    hidden = torch.randn(5, 8)
    expert_indices = torch.tensor([[0], [1], [0], [1], [1]])
    routing_weights = torch.rand(5, 1)

    expected = original(hidden, expert_indices, routing_weights)
    linearized = LinearExperts2D(original)
    actual = linearized(hidden, expert_indices, routing_weights)

    torch.testing.assert_close(actual, expected)
    assert isinstance(linearized[0].gate_proj, torch.nn.Linear)
    assert isinstance(linearized[0].up_proj, torch.nn.Linear)
    assert isinstance(linearized[0].down_proj, torch.nn.Linear)


def test_linearized_experts_only_observe_router_selected_tokens():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    original = Qwen3MoeExperts(
        Qwen3MoeConfig(hidden_size=8, moe_intermediate_size=4, num_experts=2)
    ).eval()
    for parameter in original.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.1)
    linearized = LinearExperts2D(original)
    observed: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    handles = [
        expert.gate_proj.register_forward_pre_hook(
            lambda _module, args, index=index: observed[index].append(args[0])
        )
        for index, expert in enumerate(linearized)
    ]
    hidden = torch.randn(5, 8)
    expert_indices = torch.tensor([[0], [1], [0], [1], [1]])
    routing_weights = torch.ones(5, 1)

    try:
        linearized(hidden, expert_indices, routing_weights)
    finally:
        for handle in handles:
            handle.remove()

    assert [item.shape[0] for item in observed[0]] == [2]
    assert [item.shape[0] for item in observed[1]] == [3]


def test_awq_moe_mapping_balances_router_without_quantizing_it():
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(2, 8))

    class Moe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Router()
            experts = torch.nn.Module()
            experts.down_proj = torch.nn.Parameter(torch.randn(2, 8, 4))
            experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 8, 8))
            experts.hidden_dim = 8
            self.experts = LinearExperts2D(experts)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = torch.nn.LayerNorm(8)
            self.mlp = Moe()

    block = Block()
    selected = {
        name
        for name, module in block.named_modules()
        if name.startswith("mlp.experts.") and isinstance(module, torch.nn.Linear)
    }
    mapping = next(
        item
        for item in infer_awq_mappings(block, selected)
        if item.previous_name == "post_attention_layernorm"
    )

    assert "mlp.gate" in mapping.linear_names
    assert "mlp.gate" not in mapping.quantized_names


def test_awq_mapping_skips_gqa_value_to_output_scale_when_value_is_narrower():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 2, bias=False)
            self.v_proj = torch.nn.Linear(8, 2, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(8)
            self.self_attn = Attention()

    block = Block()
    selected = {
        name
        for name, module in block.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    mappings = infer_awq_mappings(block, selected)

    assert any(mapping.inspect_name == "self_attn" for mapping in mappings)
    assert not any(mapping.inspect_name == "self_attn.o_proj" for mapping in mappings)


def test_tiny_llama_runs_gptq_and_awq_sequentially():
    from transformers import LlamaConfig, LlamaForCausalLM

    for method in ("gptq", "awq"):
        torch.manual_seed(11)
        model = LlamaForCausalLM(
            LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=32,
                max_position_embeddings=32,
                use_cache=False,
            )
        ).eval()
        layers = decoder_layers(model)
        batches = [
            {
                "input_ids": torch.randint(0, 32, (1, 8)),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }
            for _ in range(2)
        ]
        samples = capture_first_layer_inputs(
            model,
            layers[0][1],
            batches,
            device=torch.device("cpu"),
        )
        policy = QuantizationPolicy(
            selections=("linear",),
            method=method,
            num_bits=4,
            group_size=8,
            gptq=GPTQPolicy(desc_act=True),
            awq=AWQPolicy(n_grid=2),
        )

        quantized = quantize_model_sequential(
            layers,
            samples,
            policy,
            device=torch.device("cpu"),
        )

        assert len(quantized) == 7
        assert all(result.method == method for result in quantized.values())


def test_tiny_llama_runs_native_w8a16_autoround_with_gptq_packing():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
            max_position_embeddings=32,
            use_cache=False,
        )
    ).eval()
    layers = decoder_layers(model)
    batches = [
        {
            "input_ids": torch.randint(0, 32, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }
        for _ in range(2)
    ]
    samples = capture_first_layer_inputs(
        model,
        layers[0][1],
        batches,
        device=torch.device("cpu"),
    )
    policy = QuantizationPolicy(
        selections=("linear",),
        method="autoround",
        num_bits=8,
        group_size=8,
        autoround=AutoRoundPolicy(iters=2, batch_size=1),
    )

    quantized = quantize_model_sequential(
        layers,
        samples,
        policy,
        device=torch.device("cpu"),
    )

    assert len(quantized) == 7
    assert all(result.algorithm == "autoround" for result in quantized.values())
    assert all(result.packing == "gptq" for result in quantized.values())


def test_tiny_qwen3_moe_runs_gptq_and_awq_without_model_adapter():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    for method in ("gptq", "awq"):
        torch.manual_seed(13)
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                hidden_size=8,
                intermediate_size=16,
                moe_intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_experts=2,
                num_experts_per_tok=1,
                vocab_size=32,
                max_position_embeddings=32,
                use_cache=False,
            )
        ).eval()
        assert linearize_fused_experts(model) == ["model.layers.0.mlp.experts"]
        layers = decoder_layers(model)
        batches = [
            {
                "input_ids": torch.randint(0, 32, (1, 8)),
                "attention_mask": torch.ones(1, 8, dtype=torch.long),
            }
            for _ in range(2)
        ]
        samples = capture_first_layer_inputs(
            model,
            layers[0][1],
            batches,
            device=torch.device("cpu"),
        )
        policy = QuantizationPolicy(
            selections=("moe.routed",),
            method=method,
            group_size=8,
            awq=AWQPolicy(n_grid=2),
        )

        quantized = quantize_model_sequential(
            layers,
            samples,
            policy,
            device=torch.device("cpu"),
        )

        assert len(quantized) == 6
        assert all(".mlp.experts." in name for name in quantized)
