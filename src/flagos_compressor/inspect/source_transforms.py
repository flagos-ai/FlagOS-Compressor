"""Guard exports that cannot retain source activation transforms."""


def reject_source_transforms(config: dict) -> None:
    quantization = config.get("quantization_config") or {}
    if config.get("transform_config") or quantization.get("transform_config"):
        raise ValueError(
            "Source transform_config is unsupported: online input rotations "
            "cannot be discarded during BF16 conversion or quantization. "
            "Restore the original weight coordinates before exporting."
        )
