from __future__ import annotations


DEFAULT_SCALE_SUFFIXES = (
    ".scale",
    ".weight_scale",
    ".weight_scale_inv",
    "_scale_inv",
    ".scales",
)


def infer_weight_name_from_scale(scale_name: str) -> str | None:
    if scale_name.endswith(".weight_scale"):
        return scale_name[: -len(".weight_scale")] + ".weight_packed"
    if scale_name.endswith(".weight_scale_inv"):
        return scale_name[: -len(".weight_scale_inv")] + ".weight"
    if scale_name.endswith("_scale_inv"):
        return scale_name[: -len("_scale_inv")] + ".weight"
    if scale_name.endswith(".scale"):
        prefix = scale_name[: -len(".scale")]
        if prefix.endswith(".weight"):
            return prefix
        return prefix + ".weight"
    if scale_name.endswith(".scales"):
        prefix = scale_name[: -len(".scales")]
        if prefix.endswith(".weight"):
            return prefix
        return prefix + ".weight"
    return None


def build_scale_map(weight_names: set[str]) -> dict[str, str]:
    scale_map: dict[str, str] = {}
    for name in sorted(weight_names):
        inferred = infer_weight_name_from_scale(name)
        candidates = {inferred} if inferred in weight_names else set()
        if name.endswith(".weight_scale"):
            unpacked = name[: -len(".weight_scale")] + ".weight"
            if unpacked in weight_names:
                candidates.add(unpacked)
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous scale {name!r}: matches {sorted(candidates)}")
        if candidates:
            weight = candidates.pop()
            if weight in scale_map:
                raise ValueError(
                    f"Ambiguous scales for {weight!r}: {scale_map[weight]!r}, {name!r}"
                )
            scale_map[weight] = name
    return scale_map
