"""Exercise real vLLM CUDA layers using the pending review fixes."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig

import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.planner import build_convert_plan, build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy, UnselectedWeightsPolicy
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and vLLM")


LINEAR = "layers.0.attn.wo_b.weight"
INDEXER = "layers.0.attn.indexer.wq_b.weight"
ENGRAM = "layers.1.engram.embed.weight"


@pytest.fixture(scope="module", autouse=True)
def real_runtime(tmp_path_factory):
    assert torch.cuda.is_available()
    pytest.importorskip("vllm.models.deepseek_v41.nvidia.engram", reason="requires vLLM with DeepSeek V4.1 support")
    torch.cuda.set_device(0)
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    rendezvous = tmp_path_factory.mktemp("distributed") / "rendezvous"
    with pytest.MonkeyPatch.context() as env, set_current_vllm_config(VllmConfig()):
        env.setenv("FLAGOS_COMPRESSOR_VLLM_SOURCE_FORMATS", "1")
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="nccl",
            distributed_init_method=rendezvous.as_uri(),
        )
        try:
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            from flagos_compressor.integrations.vllm import register

            register()
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def make_source(path, dtype=torch.float32, scale_value=3.):
    path.mkdir()
    save_file({
        LINEAR: torch.ones(64, 64, dtype=torch.float8_e4m3fn),
        LINEAR.replace(".weight", ".weight_scale_inv"): torch.ones(2, 2),
        INDEXER: torch.full((64, 64), 2., dtype=torch.float8_e4m3fn),
        INDEXER.replace(".weight", ".weight_scale_inv"): torch.full((2, 2), scale_value).to(dtype),
        ENGRAM: torch.ones(3, 64, dtype=torch.float8_e4m3fn),
        ENGRAM.replace(".weight", ".weight_scale_inv"): torch.full((3, 2), 4.),
    }, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v41",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32]},
    }))


def policy(preserve=False):
    return QuantizationPolicy(
        selections=("linear",), num_bits=8, activation_num_bits=8,
        strategy="channel", n_candidates=8,
        unselected=(UnselectedWeightsPolicy("preserve", None)
                    if preserve else UnselectedWeightsPolicy()),
    )


def config_context(output, revision="revision-a"):
    from vllm.config import VllmConfig, set_current_vllm_config

    config = json.loads((output / "config.json").read_text())
    current = VllmConfig()
    current.model_config = SimpleNamespace(
        model="owner/repo", revision=revision, download_dir="/different/cache",
        hf_config=PretrainedConfig(**config),
    )
    return config, set_current_vllm_config(current)


@pytest.mark.parametrize("dtype,scale_value,expected", [
    (torch.float32, 3., 6.), (torch.uint8, 127, 2.),
    (torch.float16, 3., 6.), (torch.bfloat16, 3., 6.),
    pytest.param(getattr(torch, "float8_e8m0fnu", None), 2., 4.,
                 marks=pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"),
                                          reason="requires E8M0 dtype")),
])
def test_real_replicated_linear_scale_loading(tmp_path, dtype, scale_value, expected):
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source, dtype, scale_value)
    plan = build_quantize_plan(scan_hf_safetensors(source), policy(True))
    execute_plan(source, output, plan, build_backend("cuda", "cuda:0", fallback_policy="error"))
    validation = validate_artifact(output)
    assert validation["valid"], validation["errors"]
    config, context = config_context(output)
    with context, torch.device("cuda:0"):
        quantization = CompressedTensorsConfig.from_config(config["quantization_config"])
        layer = ReplicatedLinear(
            64, 64, bias=False, params_dtype=torch.bfloat16,
            quant_config=quantization, disable_tp=True, return_bias=False,
            prefix="language_model.model." + INDEXER.removesuffix(".weight"),
        )
        assert layer.quant_method.__class__.__name__ == "PreservedFp8LinearMethod"
        state = load_file(output / "model.safetensors")
        # Invoke vLLM's actual ReplicatedLinear loader, rather than emulating it.
        layer.weight_loader(layer.weight, state[INDEXER])
        layer.weight_loader(layer.weight_scale_inv, state[INDEXER.replace(".weight", ".weight_scale_inv")])
        assert layer.weight_scale_inv.dtype == dtype
        layer.quant_method.process_weights_after_loading(layer)
        inputs = torch.ones(2, 64, device="cuda:0", dtype=torch.bfloat16)
        torch.testing.assert_close(
            layer(inputs), torch.full((2, 64), 64 * expected, device="cuda:0", dtype=torch.bfloat16)
        )


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_real_two_stage_engram_loading_and_lookup(tmp_path, cpu_offload):
    from vllm.models.deepseek_v41.nvidia.engram import ParallelEngramEmbedding

    source, bf16, int8 = (tmp_path / name for name in ("source", "bf16", "int8"))
    make_source(source)
    backend = build_backend("cuda", "cuda:0", fallback_policy="error")
    execute_plan(source, bf16, build_convert_plan(scan_hf_safetensors(source)), backend)
    execute_plan(bf16, int8, build_quantize_plan(scan_hf_safetensors(bf16), policy()), backend)
    assert validate_artifact(int8)["valid"]
    for output in (bf16, int8):
        config, context = config_context(output)
        assert config["flagos_bf16_engram"] is True
        with context, torch.device("cuda:0"):
            layer = ParallelEngramEmbedding(3, 64, (3,), cpu_offload=cpu_offload)
            assert layer.weight.dtype == torch.bfloat16
            assert not hasattr(layer, "weight_scale_inv")
            assert layer.weight.is_cpu == cpu_offload
            state = load_file(output / "model.safetensors")
            layer.weight.weight_loader(layer.weight, state[ENGRAM])
            indices = torch.tensor([[0], [2], [-1], [3]], device="cuda:0")
            expected = torch.zeros(4, 1, 64, device="cuda:0", dtype=torch.bfloat16)
            expected[:2] = 4.
            torch.testing.assert_close(layer(indices), expected, rtol=0, atol=0)
