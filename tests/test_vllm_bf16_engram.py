from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA lookup kernel")
@pytest.mark.parametrize("head_start,local_heads", [(0, 3), (2, 2)])
def test_bf16_engram_lookup_matches_gather_with_shard_and_padding(
    head_start, local_heads
):
    pytest.importorskip("triton")
    from flagos_compressor.integrations.vllm_bf16_engram import bf16_lookup

    weight = (
        torch.arange(10 * 32, device="cuda", dtype=torch.float32)
        .reshape(10, 32)
        .bfloat16()
    )
    indices = torch.tensor([[5, 9, 14], [4, 15, 7], [10, -1, 12]], device="cuda")
    out = torch.empty((3, local_heads, 32), device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        weight=weight,
        cpu_offload=False,
        part_n_hash_cols=local_heads,
        _num_sms=8,
        vocab_start_idx=5,
        vocab_end_idx=15,
        head_start=head_start,
        n_hash_cols=3,
        dim=32,
    )
    bf16_lookup(layer, indices, out)
    expected = torch.zeros_like(out)
    for token in range(3):
        for local_head in range(local_heads):
            head = head_start + local_head
            if head < 3 and 5 <= int(indices[token, head]) < 15:
                expected[token, local_head] = weight[int(indices[token, head]) - 5]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
