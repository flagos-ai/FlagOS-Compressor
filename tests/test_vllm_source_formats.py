import pytest
import torch
import torch.nn.functional as F

from flagos_compressor.integrations.vllm import (
    _dequantize_preserved_fp8,
    _grouped_linear_projection,
    register,
)


def test_preserved_fp8_uses_rectangular_scales_and_crops_padding():
    weight = torch.ones(3, 5).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    actual = _dequantize_preserved_fp8(weight, scale, [2, 4], torch.bfloat16)
    expected = torch.tensor(
        [[1, 1, 1, 1, 2], [1, 1, 1, 1, 2], [4, 4, 4, 4, 8]], dtype=torch.bfloat16
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("groups", [1, 2, 8])
@pytest.mark.parametrize("return_bias", [False, True])
def test_grouped_projection_matches_independent_group_gemms(groups, return_bias):
    torch.manual_seed(29)
    x = torch.randn(3, groups, 7)
    weight = torch.randn(groups * 5, 7)

    def linear(value):
        result = F.linear(value, weight)
        return (result, None) if return_bias else result

    actual = _grouped_linear_projection(x, linear, groups, 5)
    expected = torch.cat(
        [
            F.linear(x[:, group], weight[group * 5 : (group + 1) * 5])
            for group in range(groups)
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)


def test_runtime_plugin_is_opt_in(monkeypatch):
    monkeypatch.delenv("FLAGOS_COMPRESSOR_VLLM_SOURCE_FORMATS", raising=False)
    register()  # Works without vLLM installed.
