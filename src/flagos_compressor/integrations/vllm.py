"""Opt-in DeepSeek V4.1 source-format support for vLLM.

Enable with FLAGOS_COMPRESSOR_VLLM_SOURCE_FORMATS=1 and include
``flagos_source_formats`` in VLLM_PLUGINS. Exported weights are never modified.
Preserved FP8 indexer projections are dequantized to BF16 at load time, without
introducing another activation quantizer. INT8 grouped output projections use
the existing compressed-tensors linear kernel.
"""

import functools
import json
import os
from pathlib import Path


def _dequantize_preserved_fp8(weight, scale, block_size, dtype):
    expanded = (
        scale.float()
        .repeat_interleave(block_size[0], 0)
        .repeat_interleave(block_size[1], 1)
    )
    if expanded.shape[0] < weight.shape[0] or expanded.shape[1] < weight.shape[1]:
        raise ValueError("Preserved FP8 scale grid does not cover its weight")
    return (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).to(dtype)


def _grouped_linear_projection(inputs, linear, groups, rank):
    """Use a quantized Linear while selecting each group's own output rows."""
    import torch

    tokens = inputs.shape[0]
    projected = linear(inputs.reshape(tokens * groups, -1))
    if isinstance(projected, tuple):
        projected = projected[0]
    projected = projected.reshape(tokens, groups, groups, rank)
    group_ids = torch.arange(groups, device=inputs.device)
    return projected[:, group_ids, group_ids, :].reshape(tokens, groups * rank)


@functools.lru_cache(maxsize=8)
def _source_contract(model):
    path = Path(model) / "config.json"
    if not path.is_file():
        return {}
    config = json.loads(path.read_text())
    if config.get("model_type") != "deepseek_v41":
        return {}
    return config.get("flagos_source_quantization", {})


def register():
    if os.environ.get("FLAGOS_COMPRESSOR_VLLM_SOURCE_FORMATS") != "1":
        return

    import torch
    import torch.nn.functional as F
    from vllm.config import get_current_vllm_config_or_none
    from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        create_fp8_scale_parameter,
        create_fp8_weight_parameter,
    )
    from vllm.model_executor.parameter import BlockQuantScaleParameter
    from vllm.models.deepseek_v41.nvidia import flashmla
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    if getattr(CompressedTensorsConfig, "_flagos_source_formats", False):
        return

    class PreservedFp8LinearMethod(LinearMethodBase):
        def __init__(self, spec):
            self.block_size = spec["storage_params"]["block_size"]

        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            loader = extra_weight_attrs.get("weight_loader")
            layer.orig_dtype = params_dtype
            layer.register_parameter(
                "weight",
                create_fp8_weight_parameter(
                    sum(output_partition_sizes), input_size_per_partition, loader
                ),
            )
            layer.register_parameter(
                "weight_scale_inv",
                create_fp8_scale_parameter(
                    BlockQuantScaleParameter,
                    output_partition_sizes,
                    input_size_per_partition,
                    self.block_size,
                    loader,
                    scale_dtype=torch.float8_e8m0fnu,
                ),
            )

        def process_weights_after_loading(self, layer):
            value = _dequantize_preserved_fp8(
                layer.weight, layer.weight_scale_inv, self.block_size, layer.orig_dtype
            )
            layer.weight = torch.nn.Parameter(value, requires_grad=False)

        def apply(self, layer, x, bias=None):
            return F.linear(x, layer.weight, bias)

    original_method = CompressedTensorsConfig.get_quant_method

    def get_quant_method(self, layer, prefix):
        current = get_current_vllm_config_or_none()
        if current is not None and isinstance(layer, LinearBase):
            contract = _source_contract(current.model_config.model)
            name = (
                prefix.removeprefix("language_model.").removeprefix("model.")
                + ".weight"
            )
            spec = contract.get("weights", {}).get(name)
            if spec is not None and ".indexer." in name:
                if spec["format"] != "fp8_block_e8m0":
                    raise ValueError(
                        f"Unsupported preserved indexer format: {spec['format']}"
                    )
                return PreservedFp8LinearMethod(spec)
        return original_method(self, layer, prefix)

    original_o_proj = flashmla.deep_gemm_fp8_o_proj

    def o_proj(o, positions, cos_sin_cache, wo_a, wo_b, **kw):
        if wo_a.weight.dtype != torch.int8:
            return original_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, **kw)
        inputs, _ = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=kw["n_groups"],
            heads_per_group=kw["heads_per_group"],
            nope_dim=kw["nope_dim"],
            rope_dim=kw["rope_dim"],
            quant_group_size=kw["einsum_recipe"][2],
            tma_aligned_scales=kw["tma_aligned_scales"],
            quantize=False,
        )
        return wo_b(
            _grouped_linear_projection(inputs, wo_a, kw["n_groups"], kw["o_lora_rank"])
        )

    CompressedTensorsConfig.get_quant_method = get_quant_method
    CompressedTensorsConfig._flagos_source_formats = True
    flashmla.deep_gemm_fp8_o_proj = o_proj
