"""Validated source layouts that cannot be inferred from byte sizes alone."""

from __future__ import annotations

import re


def fp8_source_params(
    name: str,
    shape: tuple[int, ...],
    scale_shape: tuple[int, ...] | None,
    config: dict,
) -> dict | None:
    """Resolve a source FP8 layout only when its scale grid is consistent."""
    quant = config.get("quantization_config") or {}
    block = quant.get("weight_block_size", [128, 128])
    if not isinstance(block, (list, tuple)) or len(block) != 2:
        raise ValueError(f"Invalid FP8 weight_block_size: {block!r}")
    block = tuple(int(v) for v in block)
    if any(v <= 0 for v in block):
        raise ValueError(f"Invalid FP8 weight_block_size: {block!r}")
    if len(shape) != 2 or scale_shape is None:
        return None

    model_type = config.get("model_type")
    if model_type == "deepseek_v41" and re.fullmatch(
        r"layers\.\d+\.engram\.embed\.weight", name
    ):
        block = (1, 32)

    # MiMo stores independently padded checkpoint TP chunks, each [Q_i, K_i, V_i].
    # Both the scales and the weight rows must be decoded before reordering.
    match = re.fullmatch(
        r"model\.(?:layers\.(\d+)|mtp\.layers\.\d+)\.self_attn\.qkv_proj\.weight",
        name,
    )
    if (
        model_type == "mimo_v2"
        and match
        and config.get("attention_projection_layout") == "fused_qkv"
    ):
        layer = match.group(1)
        pattern = config.get("hybrid_layer_pattern") or []
        if layer is not None and int(layer) >= len(pattern):
            raise ValueError(f"Missing MiMo attention layout for {name}")
        swa = layer is None or bool(pattern[int(layer)])
        prefix = "swa_" if swa else ""
        heads = int(config[prefix + "num_attention_heads"])
        kv_heads = int(config[prefix + "num_key_value_heads"])
        head_dim = int(config[prefix + "head_dim"])
        v_dim = int(config[prefix + "v_head_dim"])
        # The source TP count is the global-attention KV-head count, including
        # SWA/MTP layers with more KV heads (MiMo-V2.5: 4 chunks, 8 SWA KV heads).
        chunks = int(config["num_key_value_heads"])
        if chunks <= 0 or heads % chunks or kv_heads % chunks:
            raise ValueError(f"Invalid MiMo Q/KV head ratio for {name}")
        sizes = [
            heads // chunks * head_dim,
            kv_heads // chunks * head_dim,
            kv_heads // chunks * v_dim,
        ]
        group_rows = sum(sizes)
        expected = (
            chunks * ((group_rows + block[0] - 1) // block[0]),
            (shape[1] + block[1] - 1) // block[1],
        )
        if shape[0] != chunks * group_rows or tuple(scale_shape) != expected:
            raise ValueError(
                f"Invalid MiMo grouped QKV layout for {name}: weight={shape}, "
                f"scale={scale_shape}, expected scale={expected}"
            )
        return {
            "block_size": list(block),
            "qkv_group_sizes": sizes,
            "qkv_groups": chunks,
        }

    expected = tuple((dim + size - 1) // size for dim, size in zip(shape, block))
    if tuple(scale_shape) == expected or (
        len(scale_shape) == 1 and scale_shape[0] == expected[0] * expected[1]
    ):
        return {"block_size": list(block)}
    return None
