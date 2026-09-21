from __future__ import annotations


ATTENTION_LINEAR_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "query", "key",
    "value", "dense", "c_attn", "c_proj", "qkv_proj", "query_key_value",
    "wq", "wk", "wv", "wo", "wq_a", "wq_b", "wkv", "wkv_a", "wkv_b",
    "wo_a", "wo_b",
    "kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj",
    "kv_proj", "o_b_proj",
    "in_proj_qkv", "in_proj_qkvz", "in_proj_ba", "in_proj_z", "in_proj_b",
    "in_proj_a",
}

MLP_LINEAR_NAMES = {
    "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3",
    "dense_h_to_4h", "dense_4h_to_h",
}

# Fused routed-expert banks store every expert of one projection as a single
# 3D tensor (``[num_experts, out, in]``) whose leaf is the projection name and
# whose parent module is ``experts``. These tensors have no ``.weight`` suffix,
# so they are classified separately from the 2D linear weights above.
FUSED_EXPERT_LEAF_NAMES = {
    "gate_up_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3",
}


def classify_fused_expert(name: str) -> tuple[str | None, tuple[str, ...]]:
    """Classify a fused routed-expert bank tensor (``experts.<proj>``).

    Returns ``("moe_routed_fused", tags)`` when ``name`` is a fused routed
    expert projection (parent module ``experts``, leaf a known projection),
    otherwise ``(None, ())``. These tensors carry no ``.weight`` suffix and are
    stored as 3D ``[num_experts, out, in]`` banks.
    """
    if name.endswith(".weight"):
        return None, ()
    parts = name.lower().split(".")
    if len(parts) < 2:
        return None, ()
    leaf = parts[-1]
    parent = parts[-2]
    shared_parts = {"shared_expert", "shared_experts"}
    if any(part in shared_parts for part in parts):
        return None, ()
    if parent == "experts" and leaf in FUSED_EXPERT_LEAF_NAMES:
        return "moe_routed_fused", ("linear", "moe", "moe.routed")
    return None, ()


def classify_weight(name: str) -> tuple[str | None, tuple[str, ...]]:
    """Classify common linear weights without making conversion decisions."""
    if not name.endswith(".weight"):
        return classify_fused_expert(name)
    parts = name[: -len(".weight")].lower().split(".")
    leaf = parts[-1]
    tags: set[str] = set()

    # DeepSeek-V4's stateful attention compressor/indexer must remain in its
    # source floating-point format. In particular, its ``gate_proj`` name is
    # MLP-like but is not an MLP projection and must not match ``linear``.
    if "self_attn" in parts and "compressor" in parts:
        return "attention_compressor", ("attention.compressor",)

    shared_parts = {"shared_expert", "shared_experts"}
    is_shared = any(part in shared_parts for part in parts)
    is_routed = "experts" in parts and not is_shared
    is_mlp_linear = leaf in MLP_LINEAR_NAMES
    is_attention_linear = leaf in ATTENTION_LINEAR_NAMES

    # DeepSeek-V4's state compressor happens to use a ``wkv`` leaf, but vLLM
    # instantiates ``compressor.wkv`` + ``compressor.wgate`` as one
    # ``MergedColumnParallelLinear`` with ``quant_config=None``. Treating the
    # nested ``wkv`` as an ordinary attention projection would quantize only
    # half of that fused runtime module and produce an unloadable checkpoint.
    if "compressor" in parts and leaf in {"wkv", "wgate"}:
        return None, ()

    if is_mlp_linear or is_attention_linear:
        tags.add("linear")
    if is_routed and is_mlp_linear:
        tags.update({"moe", "moe.routed"})
        return "moe_routed_linear", tuple(sorted(tags))
    if is_shared and is_mlp_linear:
        tags.update({"moe", "moe.shared"})
        return "moe_shared_linear", tuple(sorted(tags))
    if is_attention_linear:
        tags.add("attention")
        return "attention_linear", tuple(sorted(tags))
    if is_mlp_linear:
        tags.add("mlp")
        return "mlp_linear", tuple(sorted(tags))
    if leaf == "lm_head":
        return "lm_head", ("lm_head",)
    if leaf in {"embed_tokens", "embedding", "word_embeddings"}:
        return "embedding", ("embedding",)
    return None, ()


def infer_logical_shape(
    storage_shape: tuple[int, ...], storage_format: str | None
) -> tuple[int, ...]:
    if storage_format == "fp4_e2m1_e8m0" and len(storage_shape) == 2:
        return (storage_shape[0], storage_shape[1] * 2)
    return storage_shape


def infer_storage_format(
    weight_shape: tuple[int, ...],
    element_size: int,
    scale_shape: tuple[int, ...] | None,
) -> str | None:
    """Infer a source quantization format using strict layout validation.

    A byte-sized weight paired with a same-rows 2D scale is treated as
    MXFP4 E2M1 + E8M0 (per-row groups). A byte-sized weight paired with a
    smaller or 1D scale is treated as block-FP8 + E8M0. Anything else
    (multi-byte dtype, no scale, non-2D weight) is not a recognized
    quantized format.
    """
    if element_size != 1 or scale_shape is None:
        return None
    if len(weight_shape) != 2:
        return None
    # MXFP4 stores two values per weight byte and one scale per 32 logical
    # values. Merely checking that rows match would misclassify row-wise INT8.
    if (
        len(scale_shape) == 2
        and scale_shape[0] == weight_shape[0]
        and scale_shape[1] > 0
        and weight_shape[1] * 2 == scale_shape[1] * 32
    ):
        return "fp4_e2m1_e8m0"

    # Block FP8 has one scale per padded 128x128 tile. Flat scale tensors are
    # accepted when their element count is exact.
    rows, cols = weight_shape
    expected_scales = ((rows + 127) // 128) * ((cols + 127) // 128)
    scale_elements = 1
    for dim in scale_shape:
        scale_elements *= dim
    if scale_elements == expected_scales:
        return "fp8_block_e8m0"
    return None
