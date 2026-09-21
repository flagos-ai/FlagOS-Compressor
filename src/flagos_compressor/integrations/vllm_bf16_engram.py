"""BF16 Engram storage for the opt-in DeepSeek V4.1 CUDA adapter."""

import torch
import triton
import triton.language as tl


@triton.jit
def _lookup_bf16(
    weight,
    ids,
    out,
    start,
    end,
    rows,
    stride_t,
    stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID: tl.constexpr,
):
    cols = tl.arange(0, DIM)
    for base in tl.range(tl.program_id(0) * BLOCK_R, rows, GRID * BLOCK_R):
        row = base + tl.arange(0, BLOCK_R)
        valid = row < rows
        head = HEAD_START + row % LOCAL_HEADS
        token = (row // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * stride_t + head * stride_h,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS) & (index >= start) & (index < end)
        local = tl.where(owned, index - start, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )
        tl.store(out + row[:, None] * DIM + cols[None, :], values, mask=valid[:, None])


def bf16_lookup(layer, indices, out, background=False):
    rows = indices.shape[0] * layer.part_n_hash_cols
    if not rows:
        return
    weight = layer.weight.data
    if layer.cpu_offload:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        pointer = weight.data_ptr()
        if getattr(layer, "_flagos_bf16_view_src", None) != pointer:
            layer._flagos_bf16_view = get_accelerator_view_from_cpu_tensor(weight)
            layer._flagos_bf16_view_src = pointer
        weight = layer._flagos_bf16_view
    grid = min(
        triton.cdiv(rows, 16), max(1, layer._num_sms // (2 if background else 1))
    )
    _lookup_bf16[(grid,)](
        weight,
        indices,
        out,
        layer.vocab_start_idx,
        layer.vocab_end_idx,
        rows,
        indices.stride(0),
        indices.stride(1),
        HEAD_START=layer.head_start,
        LOCAL_HEADS=layer.part_n_hash_cols,
        TOTAL_HEADS=layer.n_hash_cols,
        DIM=layer.dim,
        BLOCK_R=16,
        GRID=grid,
    )


def register_bf16_engram():
    from vllm.config import get_current_vllm_config_or_none
    from vllm.models.deepseek_v41.nvidia.engram import ParallelEngramEmbedding

    cls = ParallelEngramEmbedding
    if getattr(cls, "_flagos_bf16_adapter", False):
        return
    original_init = cls.__init__
    original_allocate = cls._allocate_weights
    original_lookup = cls.lookup

    def initialize(self, *args, **kwargs):
        current = get_current_vllm_config_or_none()
        self._flagos_bf16 = bool(
            current
            and getattr(current.model_config.hf_config, "flagos_bf16_engram", False)
        )
        original_init(self, *args, **kwargs)
        if self._flagos_bf16:
            # The upstream constructor attaches loaders to two parameters.
            # The BF16 checkpoint has no source scales; retain only its weight.
            del self.weight_scale_inv

    def allocate(self):
        if not self._flagos_bf16:
            return original_allocate(self)
        if self.dp_shared_memory:
            raise ValueError("BF16 Engram does not support DP-shared host storage")
        options = {"device": "cpu", "pin_memory": True} if self.cpu_offload else {}
        return (
            torch.empty(
                self.part_num_embeddings, self.dim, dtype=torch.bfloat16, **options
            ),
            torch.empty(0, dtype=torch.uint8, **options),
        )

    def lookup(self, indices, out, background=False):
        if self._flagos_bf16:
            return bf16_lookup(self, indices, out, background)
        return original_lookup(self, indices, out, background)

    cls.__init__ = initialize
    cls._allocate_weights = allocate
    cls.lookup = lookup
    cls._flagos_bf16_adapter = True
