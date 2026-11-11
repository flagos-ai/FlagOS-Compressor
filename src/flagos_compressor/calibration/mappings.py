"""Lightweight structure mappings; model forward remains owned by Transformers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class AWQMapping:
    previous_name: str
    quantized_names: tuple[str, ...]
    linear_names: tuple[str, ...]
    inspect_name: str
    input_name: str


def gptq_sequential_groups(names: list[str]) -> list[list[str]]:
    """Architecture-aware ordering with a deterministic catch-all."""
    stages = (
        (
            "q_proj", "k_proj", "v_proj", "qkv_proj", "query_key_value",
            "q_a_proj", "kv_a_proj_with_mqa", "kv_proj", "wq_a", "wkv_a",
            "in_proj_",
        ),
        ("q_b_proj", "kv_b_proj", "wq_b", "wkv_b"),
        ("o_proj", "out_proj", "o_b_proj"),
        ("gate_proj", "up_proj", "gate_up_proj", "fc1", "w1", "w3"),
        ("down_proj", "fc2", "w2"),
    )
    remaining = list(names)
    groups: list[list[str]] = []
    for suffixes in stages:
        group = [
            name
            for name in remaining
            if any(name.rsplit(".", 1)[-1].startswith(suffix) for suffix in suffixes)
        ]
        if group:
            groups.append(group)
            remaining = [name for name in remaining if name not in group]
    # Unknown projections are still quantized, but each receives a fresh forward
    # so true-sequential error propagation remains conservative.
    groups.extend([[name] for name in remaining])
    return groups


def _first_existing(modules: dict[str, nn.Module], names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in modules), None)


def _is_weight_consumer(module: nn.Module | None) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, nn.Parameter) and weight.dim() == 2


def _can_balance_linear_pair(
    previous: nn.Module | None,
    consumer: nn.Module | None,
) -> bool:
    """Whether the producer has enough output channels for consumer scales.

    A fused QKV producer may have extra leading channels, which AutoAWQ handles
    by scaling the trailing value slice. GQA/MQA value projections can instead
    be narrower than the repeated attention input consumed by ``o_proj``; that
    relation cannot be equalized with a single producer-weight transform.
    """
    previous_weight = getattr(previous, "weight", None)
    consumer_weight = getattr(consumer, "weight", None)
    return (
        isinstance(previous_weight, nn.Parameter)
        and isinstance(consumer_weight, nn.Parameter)
        and previous_weight.dim() == 2
        and consumer_weight.dim() == 2
        and previous_weight.shape[0] >= consumer_weight.shape[1]
    )


def _deepseek_attention_consumers(modules, root, *, latent=False):
    """Include floating-point branches fed by the same scaled activation.

    A module can be excluded from quantization and still require balancing.
    V4's compressed attention and indexer share both the normalized hidden
    input and the normalized low-rank query with the selected projections.
    """
    if modules[root].__class__.__name__ not in {
        "DeepseekV4Attention", "DeepseekV41Attention",
    }:
        return ()
    leaves = (
        ("q_b_proj", "compressor.indexer.q_b_proj", "indexer.q_b_proj")
        if latent else (
            "q_a_proj", "kv_proj", "compressor.kv_proj", "compressor.gate_proj",
            "compressor.indexer.kv_proj", "compressor.indexer.gate_proj",
            "compressor.indexer.scorer.weights_proj", "indexer.weights_proj",
        )
    )
    return tuple(f"{root}.{leaf}" for leaf in leaves
                 if _is_weight_consumer(modules.get(f"{root}.{leaf}")))


def _has_clipped_up_activation(module):
    # Linearized experts retain their original bound _apply_gate method;
    # shared experts expose limit directly.
    gate = getattr(module, "apply_gate", None)
    owner = getattr(gate, "__self__", module)
    limit = getattr(owner, "limit", None)
    if limit is None:
        limit = getattr(getattr(owner, "config", None), "swiglu_limit", None)
    return isinstance(limit, (int, float)) and 0 < limit < float("inf")


def infer_awq_mappings(layer: nn.Module, selected: set[str]) -> list[AWQMapping]:
    """Infer AutoAWQ scale relations from common Transformers module structure."""
    modules = dict(layer.named_modules())
    mappings: list[AWQMapping] = []

    attention_roots = [
        name
        for name in modules
        if name and name.rsplit(".", 1)[-1] in {"self_attn", "attention", "attn", "linear_attn"}
    ]
    for root in attention_roots:
        qkv = tuple(
            name
            for name in sorted(selected)
            if name.startswith(root + ".")
            and name.rsplit(".", 1)[-1]
            in {
                "q_proj", "k_proj", "v_proj", "qkv_proj", "query_key_value",
                "q_a_proj", "kv_a_proj_with_mqa", "kv_proj", "wq_a", "wkv_a",
            }
        )
        qkv += tuple(
            name
            for name in sorted(selected)
            if name.startswith(root + ".")
            and name.rsplit(".", 1)[-1].startswith("in_proj_")
        )
        previous = _first_existing(
            modules,
            (
                "input_layernorm",
                "attention_norm",
                "attn_norm",
                "pre_attention_layernorm",
            ),
        )
        if previous and qkv:
            balance = tuple(dict.fromkeys(qkv + _deepseek_attention_consumers(modules, root)))
            mappings.append(AWQMapping(previous, qkv, balance, root, root))

        # DeepSeek/GLM MLA uses low-rank A -> RMSNorm -> B projection pairs.
        # Balance the B projection against its own latent-space norm instead of
        # treating the topology as a standard Q/K/V projection.
        latent_pairs = (
            (("q_a_layernorm", "q_a_norm", "wq_a_layernorm"), ("q_b_proj", "wq_b")),
            (("kv_a_layernorm", "kv_a_norm", "wkv_a_layernorm"), ("kv_b_proj", "wkv_b")),
        )
        for norm_leaves, projection_leaves in latent_pairs:
            latent_norm = _first_existing(
                modules,
                tuple(f"{root}.{leaf}" for leaf in norm_leaves),
            )
            projection = next(
                (
                    f"{root}.{leaf}"
                    for leaf in projection_leaves
                    if f"{root}.{leaf}" in selected
                ),
                None,
            )
            if latent_norm and projection:
                balance = (projection,)
                if projection.rsplit(".", 1)[-1] in {"q_b_proj", "wq_b"}:
                    balance = tuple(dict.fromkeys(
                        balance + _deepseek_attention_consumers(modules, root, latent=True)))
                mappings.append(
                    AWQMapping(
                        latent_norm,
                        (projection,),
                        balance,
                        projection,
                        projection,
                    )
                )

        value = next(
            (
                name
                for name in sorted(selected)
                if name.endswith(".v_proj") and name.startswith(root)
            ),
            None,
        )
        if value is None:
            value = next(
                (
                    name
                    for name in sorted(selected)
                    if name.startswith(root)
                    and name.rsplit(".", 1)[-1]
                    in {"qkv_proj", "query_key_value"}
                ),
                None,
            )
        output = next(
            (
                name
                for name in sorted(selected)
                if name.startswith(root)
                and name.rsplit(".", 1)[-1]
                in {"o_proj", "out_proj", "o_b_proj", "dense"}
            ),
            None,
        )
        if value and output and _can_balance_linear_pair(
            modules.get(value),
            modules.get(output),
        ):
            mappings.append(AWQMapping(value, (output,), (output,), output, output))

    mlp_roots = [
        name
        for name in modules
        if name and name.rsplit(".", 1)[-1] in {"mlp", "feed_forward", "ffn"}
    ]
    for root in mlp_roots:
        inputs = tuple(
            name
            for name in sorted(selected)
            if name.startswith(root + ".")
            and name.rsplit(".", 1)[-1]
            in {"gate_proj", "up_proj", "gate_up_proj", "fc1", "w1", "w3"}
        )
        previous = _first_existing(
            modules,
            (
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "ffn_norm",
                "pre_mlp_layernorm",
            ),
        )
        if previous and inputs:
            # MoE routers consume the same normalized hidden states as their
            # experts. They normally stay in floating point, but must receive
            # the inverse equalization transform or routing would change.
            router_names = tuple(
                name
                for name in (
                    f"{root}.gate",
                    f"{root}.router",
                    f"{root}.shared_expert_gate",
                )
                if _is_weight_consumer(modules.get(name)) and name not in inputs
            )
            mappings.append(
                AWQMapping(
                    previous,
                    inputs,
                    inputs + router_names,
                    root,
                    root,
                )
            )

        for up_name in [
            name
            for name in sorted(selected)
            if name.startswith(root + ".")
            and name.rsplit(".", 1)[-1] in {"up_proj", "gate_up_proj", "fc1", "w3"}
        ]:
            parent = up_name.rsplit(".", 1)[0]
            down_name = next(
                (
                    f"{parent}.{leaf}"
                    for leaf in ("down_proj", "fc2", "w2")
                    if f"{parent}.{leaf}" in selected
                ),
                None,
            )
            # clamp(up / s, +/-limit) * s != clamp(up, +/-limit).
            # Keep quantizing down_proj, but do not absorb its input scale into
            # up_proj across a fixed activation clamp. Changing the runtime's
            # clamp would require an explicit per-channel inference contract.
            if down_name and not _has_clipped_up_activation(modules[parent]):
                mappings.append(
                    AWQMapping(
                        up_name,
                        (down_name,),
                        (down_name,),
                        down_name,
                        down_name,
                    )
                )
    return mappings


__all__ = ["AWQMapping", "gptq_sequential_groups", "infer_awq_mappings"]
