"""CPU loading regressions with vLLM's parameter-copy/dispatch interfaces."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.planner import build_convert_plan, build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy, UnselectedWeightsPolicy
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


LINEAR = "layers.0.attn.wo_b.weight"
INDEXER = "layers.0.attn.indexer.wq_b.weight"
ENGRAM = "layers.1.engram.embed.weight"


@pytest.fixture
def runtime(monkeypatch):
    """Exercise the real adapters without importing CUDA-only vLLM modules."""
    def module(name, **attrs):
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            if parent not in sys.modules:
                module(parent)
        value = ModuleType(name)
        value.__path__ = []
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        if "." in name:
            monkeypatch.setattr(sys.modules[parent], leaf, value, raising=False)
        return value

    current = SimpleNamespace(model_config=None)
    module("vllm")
    module("vllm.config", get_current_vllm_config_or_none=lambda: current)

    class Linear(torch.nn.Module):
        pass

    class QuantConfig:
        def get_quant_method(self, layer, prefix):
            return "original"

    module("vllm.model_executor.layers.linear", LinearBase=Linear, LinearMethodBase=object)
    module(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        CompressedTensorsConfig=QuantConfig,
    )

    def scale_parameter(cls, sizes, cols, block, loader, scale_dtype):
        rows = sum((size + block[0] - 1) // block[0] for size in sizes)
        return torch.nn.Parameter(
            torch.empty(rows, (cols + block[1] - 1) // block[1], dtype=scale_dtype),
            requires_grad=False,
        )

    module(
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        create_fp8_weight_parameter=lambda rows, cols, loader: torch.nn.Parameter(
            torch.empty(rows, cols, dtype=torch.float8_e4m3fn), requires_grad=False
        ),
        create_fp8_scale_parameter=scale_parameter,
    )
    module("vllm.model_executor.parameter", BlockQuantScaleParameter=torch.nn.Parameter)
    module("vllm.models.deepseek_v41.nvidia.flashmla", deep_gemm_fp8_o_proj=None)
    module(
        "vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant",
        fused_inv_rope_fp8_quant=None,
    )

    class Engram(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dp_shared_memory = self.cpu_offload = False
            self.part_num_embeddings, self.dim = 3, 64
            weight, scale = self._allocate_weights()
            self.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
            self.register_parameter(
                "weight_scale_inv", torch.nn.Parameter(scale, requires_grad=False)
            )

        def _allocate_weights(self):
            return (torch.empty(3, 64, dtype=torch.float8_e4m3fn),
                    torch.empty(3, 2, dtype=torch.uint8))

        def lookup(self, *args):
            raise AssertionError("GPU lookup is not part of a CPU loading test")

    module("vllm.models.deepseek_v41.nvidia.engram", ParallelEngramEmbedding=Engram)
    module("triton", jit=lambda f: f)
    module("triton.language", constexpr=int)
    name = "flagos_compressor.integrations.vllm_bf16_engram"
    path = Path(__file__).resolve().parents[1] / "src" / Path(*name.split(".")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    engram_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, engram_module)
    spec.loader.exec_module(engram_module)
    monkeypatch.setenv("FLAGOS_COMPRESSOR_VLLM_SOURCE_FORMATS", "1")
    from flagos_compressor.integrations.vllm import register

    register()
    return SimpleNamespace(current=current, Linear=Linear, QuantConfig=QuantConfig, Engram=Engram)


def make_source(path, scale_dtype=torch.float32, scale_value=3):
    path.mkdir()
    save_file({
        LINEAR: torch.ones(64, 64, dtype=torch.float8_e4m3fn),
        LINEAR.replace(".weight", ".weight_scale_inv"): torch.ones(2, 2),
        INDEXER: torch.full((64, 64), 2., dtype=torch.float8_e4m3fn),
        INDEXER.replace(".weight", ".weight_scale_inv"): torch.full((2, 2), scale_value, dtype=scale_dtype),
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


def load_config(runtime, output, model="owner/repo", revision="revision-a"):
    config = json.loads((output / "config.json").read_text())
    runtime.current.model_config = SimpleNamespace(
        model=model, revision=revision, download_dir="/different/cache",
        hf_config=SimpleNamespace(**config),
    )
    return config


@pytest.mark.parametrize("model,revision", [
    ("owner/repo", "revision-a"), ("owner/repo", "revision-b"),
    ("/cache/models--owner--repo/snapshots/resolved", "revision-c"),
])
@pytest.mark.parametrize("dtype,value,expected", [
    (torch.float32, 3., 6.), (torch.uint8, 127, 2.),
    (torch.float16, 3., 6.), (torch.bfloat16, 3., 6.),
    (torch.float8_e8m0fnu, 2., 4.),
])
def test_preserved_indexer_export_and_load(tmp_path, runtime, model, revision, dtype, value, expected):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source, dtype, value)
    plan = build_quantize_plan(scan_hf_safetensors(source), policy(preserve=True))
    execute_plan(source, output, plan, build_backend("cpu"))
    assert validate_artifact(output)["valid"]
    config = load_config(runtime, output, model, revision)
    params = config["flagos_source_quantization"]["weights"][INDEXER]["storage_params"]
    assert params["scale_dtype"] == str(dtype).removeprefix("torch.")
    layer = runtime.Linear()
    method = runtime.QuantConfig().get_quant_method(
        layer, "language_model.model." + INDEXER.removesuffix(".weight")
    )
    assert method != "original"
    method.create_weights(layer, 64, [64], 64, 64, torch.bfloat16)
    state = load_file(output / "model.safetensors")
    layer.weight.data.copy_(state[INDEXER])
    # Mirrors ReplicatedLinear.weight_loader, including numeric dtype conversion.
    layer.weight_scale_inv.data.copy_(state[INDEXER.replace(".weight", ".weight_scale_inv")])
    assert layer.weight_scale_inv.dtype == dtype
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight, torch.full((64, 64), expected, dtype=torch.bfloat16))
    inputs = torch.ones(1, 64, dtype=torch.bfloat16)
    torch.testing.assert_close(method.apply(layer, inputs), torch.full((1, 64), 64 * expected, dtype=torch.bfloat16))


def test_two_stage_bf16_engram_export_and_load(tmp_path, runtime):
    source, bf16, int8 = (tmp_path / name for name in ("source", "bf16", "int8"))
    make_source(source)
    execute_plan(source, bf16, build_convert_plan(scan_hf_safetensors(source)), build_backend("cpu"))
    # Exercise both the pure BF16 export and the subsequent W8A8 export.
    for output in (bf16, int8):
        if output == int8:
            plan = build_quantize_plan(scan_hf_safetensors(bf16), policy())
            assert ENGRAM in {t.name for t in plan.kept_tensors}
            execute_plan(bf16, int8, plan, build_backend("cpu"))
        config = load_config(runtime, output)
        assert config["flagos_bf16_engram"] is True
        layer = runtime.Engram()
        assert layer.weight.dtype == torch.bfloat16
        assert not hasattr(layer, "weight_scale_inv")
        state = load_file(output / "model.safetensors")
        layer.weight.data.copy_(state[ENGRAM])
        torch.testing.assert_close(layer.weight, torch.full((3, 64), 4., dtype=torch.bfloat16))
    assert validate_artifact(int8)["valid"]


def test_preserved_indexer_rejects_missing_scale_encoding(tmp_path, runtime):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    execute_plan(source, output, build_quantize_plan(scan_hf_safetensors(source), policy(True)), build_backend("cpu"))
    config = load_config(runtime, output)
    del config["flagos_source_quantization"]["weights"][INDEXER]["storage_params"]["scale_dtype"]
    with pytest.raises(ValueError, match="re-export"):
        runtime.QuantConfig().get_quant_method(runtime.Linear(), INDEXER.removesuffix(".weight"))
