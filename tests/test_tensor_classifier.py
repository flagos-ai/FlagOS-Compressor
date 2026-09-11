from flagos_compressor.inspect.tensor_classifier import (
    classify_weight,
    infer_storage_format,
)


def test_no_scale_returns_none():
    assert infer_storage_format((16, 32), element_size=2, scale_shape=None) is None
    assert infer_storage_format((16, 32), element_size=1, scale_shape=None) is None


def test_multibyte_weight_returns_none():
    assert infer_storage_format((16, 32), element_size=2, scale_shape=(16, 1)) is None


def test_byte_weight_with_per_row_2d_scale_is_fp4():
    # MXFP4 typical layout: same number of rows as weight, one scale per group along cols.
    assert (
        infer_storage_format((128, 16), element_size=1, scale_shape=(128, 1))
        == "fp4_e2m1_e8m0"
    )


def test_rowwise_int8_layout_is_not_misclassified_as_fp4():
    assert infer_storage_format((128, 32), element_size=1, scale_shape=(128, 1)) is None


def test_byte_weight_with_block_scale_is_fp8():
    # Block-FP8 typical layout: a scale block per 128x128 tile.
    assert (
        infer_storage_format((256, 512), element_size=1, scale_shape=(2, 4))
        == "fp8_block_e8m0"
    )
    assert (
        infer_storage_format((256, 512), element_size=1, scale_shape=(8,))
        == "fp8_block_e8m0"
    )


def test_deepseek_v4_attention_compressor_is_not_selectable_linear():
    kind, tags = classify_weight(
        "model.layers.0.self_attn.compressor.indexer.gate_proj.weight"
    )

    assert kind == "attention_compressor"
    assert "linear" not in tags


def test_deepseek_v4_mla_projection_names_are_attention_linears():
    for leaf in ("kv_proj", "q_a_proj", "q_b_proj", "o_b_proj"):
        kind, tags = classify_weight(
            f"model.layers.0.self_attn.{leaf}.weight"
        )
        assert kind == "attention_linear"
        assert {"attention", "linear"} <= set(tags)


def test_deepseek_v4_attention_projection_names_are_selectable():
    for leaf in ("wkv", "wo_a", "wo_b"):
        kind, tags = classify_weight(f"layers.0.attn.{leaf}.weight")
        assert kind == "attention_linear"
        assert set(tags) == {"attention", "linear"}


def test_deepseek_v4_runtime_compressor_is_not_quantizable_attention():
    for leaf in ("wkv", "wgate"):
        assert classify_weight(
            f"layers.2.attn.compressor.{leaf}.weight"
        ) == (None, ())
