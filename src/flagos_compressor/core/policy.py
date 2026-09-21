from __future__ import annotations

from dataclasses import dataclass, field
import re

from flagos_compressor.core.profile import TensorInfo


BUILTIN_SELECTIONS = {
    "moe": "moe",
    "moe.routed": "moe.routed",
    "moe.shared": "moe.shared",
    "attention": "attention",
    "mlp": "mlp",
    "linear": "linear",
}


@dataclass(frozen=True)
class CalibrationPolicy:
    """Calibration inputs shared by activation-aware quantizers."""

    data: str | tuple[str, ...] | None = None
    samples: int = 128
    sequence_length: int = 512
    seed: int = 42
    split: str = "train"
    text_column: str = "text"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if self.samples <= 0 or self.sequence_length <= 0:
            raise ValueError("calibration samples and sequence_length must be positive")
        if not self.split or not self.text_column:
            raise ValueError("calibration split and text_column must be non-empty")
        if isinstance(self.data, tuple) and not all(
            isinstance(item, str) and item for item in self.data
        ):
            raise ValueError("calibration data entries must be non-empty strings")


@dataclass(frozen=True)
class GPTQPolicy:
    """AutoGPTQ-compatible algorithm controls."""

    block_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = True
    static_groups: bool = False
    true_sequential: bool = True
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("gptq.block_size must be positive")
        if not 0 < self.damp_percent < 1:
            raise ValueError("gptq.damp_percent must be between 0 and 1")


@dataclass(frozen=True)
class AWQPolicy:
    """AutoAWQ-compatible W4A16 GEMM controls."""

    zero_point: bool = True
    version: str = "gemm"
    duo_scaling: bool = True
    apply_clip: bool = True
    n_grid: int = 20
    max_chunk_memory: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        normalized = self.version.lower()
        object.__setattr__(self, "version", normalized)
        if normalized != "gemm":
            raise ValueError("Only AutoAWQ GEMM checkpoint format is currently supported")
        if self.n_grid <= 0 or self.max_chunk_memory <= 0:
            raise ValueError("awq.n_grid and max_chunk_memory must be positive")


@dataclass(frozen=True)
class AutoRoundPolicy:
    """Native AutoRound controls following the reference implementation."""

    iters: int = 200
    lr: float | None = None
    minmax_lr: float | None = None
    batch_size: int = 8
    gradient_accumulate_steps: int = 1
    momentum: float = 0.0
    enable_minmax_tuning: bool = True
    enable_quantized_input: bool = True

    def __post_init__(self) -> None:
        if self.iters <= 0:
            raise ValueError("autoround.iters must be positive")
        if self.lr is not None and self.lr <= 0:
            raise ValueError("autoround.lr must be positive when specified")
        if self.minmax_lr is not None and self.minmax_lr <= 0:
            raise ValueError("autoround.minmax_lr must be positive when specified")
        if self.batch_size <= 0 or self.gradient_accumulate_steps <= 0:
            raise ValueError(
                "autoround.batch_size and gradient_accumulate_steps must be positive"
            )
        if self.momentum < 0:
            raise ValueError("autoround.momentum must be non-negative")


@dataclass(frozen=True)
class UnselectedWeightsPolicy:
    """How source-quantized weights outside the selected set are handled."""

    strategy: str = "convert"
    format: str | None = "bf16"

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or not self.strategy:
            raise ValueError("unselected.strategy must be non-empty")
        if self.format is not None and not isinstance(self.format, str):
            raise ValueError("unselected.format must be a string or null")
        if self.strategy == "convert" and not self.format:
            raise ValueError("unselected.format is required for convert strategy")
        if self.strategy == "preserve" and self.format is not None:
            raise ValueError(
                "unselected.format must be omitted for preserve strategy"
            )


_TARGET_WEIGHT_FORMAT_BITS = {"int4": 4, "int8": 8}


@dataclass(frozen=True)
class TargetSchemeRule:
    """One ordered selector with explicit, existing quantization settings."""

    weight_format: str
    activation_num_bits: int
    strategy: str
    selection: str | None = None
    name_pattern: str | None = None
    group_size: int | None = None
    chunk_size: int | None = None
    scale_dtype: str | None = None

    def __post_init__(self) -> None:
        if (self.selection is None) == (self.name_pattern is None):
            raise ValueError(
                "A per-selector rule requires exactly one selection or name pattern"
            )
        if self.selection is not None and self.selection not in BUILTIN_SELECTIONS:
            raise ValueError(f"Unknown per-selector target: {self.selection}")
        if self.name_pattern is not None:
            re.compile(self.name_pattern)
        normalized_weight_format = self.weight_format.lower()
        if normalized_weight_format not in _TARGET_WEIGHT_FORMAT_BITS:
            supported = ", ".join(sorted(_TARGET_WEIGHT_FORMAT_BITS))
            raise ValueError(
                f"Unsupported target weight format {self.weight_format!r}; "
                f"currently supported: {supported}. Weight formats such as "
                "fp4 and fp8 remain distinct extension points."
            )
        object.__setattr__(self, "weight_format", normalized_weight_format)
        if self.activation_num_bits not in (8, 16):
            raise ValueError("activation-bits must be 8 or 16")
        normalized_strategy = self.strategy.lower()
        if normalized_strategy not in {"group", "channel"}:
            raise ValueError("strategy must be group or channel")
        object.__setattr__(self, "strategy", normalized_strategy)
        if self.activation_num_bits == 8 and (
            self.num_bits != 8 or self.strategy != "channel"
        ):
            raise ValueError(
                "A8 requires weight-format int8 with strategy channel"
            )
        if self.strategy == "channel" and self.num_bits != 8:
            raise ValueError("channel strategy is currently supported only for INT8")
        if self.strategy == "group" and self.group_size is None:
            object.__setattr__(
                self,
                "group_size",
                32 if self.num_bits == 4 else 128,
            )
        elif self.strategy == "channel" and self.group_size is not None:
            raise ValueError("channel strategy does not accept group-size")
        if self.chunk_size is None:
            object.__setattr__(
                self,
                "chunk_size",
                4096 if self.num_bits == 4 else 1024,
            )
        assert self.chunk_size is not None
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group-size must be a positive integer")
        if self.num_bits == 4 and self.group_size is not None and self.group_size % 2:
            raise ValueError("INT4 group-size must be even")
        if self.chunk_size <= 0:
            raise ValueError("chunk-size must be positive")
        scale_dtype_aliases = {
            "fp32": "float32",
            "float32": "float32",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
        }
        if self.is_w8a8:
            requested_scale_dtype = (self.scale_dtype or "float32").lower()
            if requested_scale_dtype not in scale_dtype_aliases:
                raise ValueError(
                    "INT8-A8 scale_dtype must be fp32, float32, bf16, or bfloat16"
                )
            object.__setattr__(
                self,
                "scale_dtype",
                scale_dtype_aliases[requested_scale_dtype],
            )
        elif self.scale_dtype is not None:
            raise ValueError("scale_dtype is supported only for int8-a8")

    @property
    def num_bits(self) -> int:
        return _TARGET_WEIGHT_FORMAT_BITS[self.weight_format]

    @property
    def scheme(self) -> str:
        """Stable artifact label; granularity is recorded separately."""
        return f"{self.weight_format}-a{self.activation_num_bits}"

    @property
    def is_w8a8(self) -> bool:
        return self.num_bits == 8 and self.activation_num_bits == 8

    @property
    def label(self) -> str:
        return self.selection or f"re:{self.name_pattern}"

    def matches_name(self, name: str, tags: tuple[str, ...]) -> bool:
        if self.selection is not None:
            return BUILTIN_SELECTIONS[self.selection] in set(tags)
        assert self.name_pattern is not None
        return re.search(self.name_pattern, name) is not None


@dataclass(frozen=True)
class QuantizationPolicy:
    selections: tuple[str, ...] = ()
    exclude_selections: tuple[str, ...] = ()
    include_names: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ()
    target_scheme_rules: tuple[TargetSchemeRule, ...] = ()
    method: str = "mse"
    format: str | None = None
    num_bits: int = 4
    activation_num_bits: int = 16
    scale_dtype: str = "float32"
    strategy: str = "group"
    group_size: int | None = None
    n_candidates: int = 200
    chunk_size: int | None = None
    calibration: CalibrationPolicy = field(default_factory=CalibrationPolicy)
    gptq: GPTQPolicy = field(default_factory=GPTQPolicy)
    awq: AWQPolicy = field(default_factory=AWQPolicy)
    autoround: AutoRoundPolicy = field(default_factory=AutoRoundPolicy)
    unselected: UnselectedWeightsPolicy = field(
        default_factory=UnselectedWeightsPolicy
    )

    def __post_init__(self) -> None:
        scale_dtype_aliases = {
            "fp32": "float32",
            "float32": "float32",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
        }
        if self.scale_dtype not in scale_dtype_aliases:
            raise ValueError(
                "scale_dtype must be one of: fp32, float32, bf16, bfloat16"
            )
        object.__setattr__(
            self,
            "scale_dtype",
            scale_dtype_aliases[self.scale_dtype],
        )
        unknown = sorted(
            (set(self.selections) | set(self.exclude_selections)) - set(BUILTIN_SELECTIONS)
        )
        if unknown:
            raise ValueError(f"Unknown selections: {', '.join(unknown)}")
        if self.method not in {"mse", "gptq", "awq", "autoround"}:
            raise ValueError("method must be one of: mse, gptq, awq, autoround")
        if self.target_scheme_rules:
            if self.method != "mse":
                raise ValueError("Selector-local rules currently require method='mse'")
            if self.selections or self.include_names:
                raise ValueError(
                    "Selector-local rules cannot be combined with legacy selections"
                )
        expected_format = {
            "mse": "compressed-tensors",
            "gptq": "gptq",
            "awq": "awq",
            "autoround": "gptq",
        }[self.method]
        if self.format is None:
            object.__setattr__(self, "format", expected_format)
        elif self.format != expected_format:
            raise ValueError(
                f"method={self.method!r} requires format={expected_format!r}"
            )
        if self.num_bits not in (4, 8):
            raise ValueError("num_bits must be 4 or 8")
        if self.activation_num_bits not in (8, 16):
            raise ValueError("activation_num_bits must be 8 or 16")
        if self.method in {"gptq", "awq", "autoround"} and self.activation_num_bits != 16:
            raise ValueError(f"{self.method.upper()} currently supports weight-only A16")
        if self.method == "awq" and self.num_bits != 4:
            raise ValueError("AutoAWQ GEMM currently supports only 4-bit weights")
        if self.method == "awq" and not self.awq.zero_point:
            raise ValueError("Native AutoAWQ GEMM requires awq.zero_point=true")
        if self.method == "autoround" and not self.gptq.symmetric:
            raise ValueError("Native AutoRound currently requires symmetric weights")
        if self.strategy not in {"group", "channel"}:
            raise ValueError("strategy must be 'group' or 'channel'")
        if self.method in {"gptq", "awq", "autoround"} and self.strategy != "group":
            raise ValueError(f"{self.method.upper()} requires group strategy")
        if self.activation_num_bits == 8 and (
            self.num_bits != 8 or self.strategy != "channel"
        ):
            raise ValueError(
                "W8A8 requires 8-bit weights with channel strategy"
            )
        if self.strategy == "channel" and self.num_bits != 8:
            raise ValueError("channel strategy is currently supported only for INT8")
        if self.strategy == "channel" and self.group_size is not None:
            raise ValueError("group_size must be omitted for channel strategy")
        if self.strategy == "group" and self.group_size is None:
            object.__setattr__(
                self,
                "group_size",
                (
                    128
                    if self.method in {"gptq", "awq", "autoround"}
                    else (32 if self.num_bits == 4 else 128)
                ),
            )
        if self.chunk_size is None:
            object.__setattr__(
                self,
                "chunk_size",
                4096 if self.num_bits == 4 else 1024,
            )
        assert self.chunk_size is not None
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if (
            self.num_bits == 4
            and self.group_size is not None
            and self.group_size % 2
        ):
            raise ValueError("INT4 group_size must be an even integer")
        if self.n_candidates <= 0 or self.chunk_size <= 0:
            raise ValueError("n_candidates and chunk_size must be positive")
        for pattern in (*self.include_names, *self.exclude_names):
            re.compile(pattern)

    @property
    def is_w8a8(self) -> bool:
        return self.num_bits == 8 and self.activation_num_bits == 8

    def selects(self, tensor: TensorInfo) -> bool:
        if tensor.role != "weight":
            return False
        return self.selects_name(tensor.name, tensor.tags)

    def selects_name(self, name: str, tags: tuple[str, ...]) -> bool:
        """Apply the selector contract to a live Transformers module weight."""
        tags = set(tags)
        if self.target_scheme_rules:
            selected = any(
                rule.matches_name(name, tuple(tags))
                for rule in self.target_scheme_rules
            )
        else:
            selected = any(
                BUILTIN_SELECTIONS[item] in tags for item in self.selections
            )
            selected = selected or any(
                re.search(pattern, name) for pattern in self.include_names
            )
        if not selected:
            return False
        if any(BUILTIN_SELECTIONS[item] in tags for item in self.exclude_selections):
            return False
        return not any(re.search(pattern, name) for pattern in self.exclude_names)

    def target_scheme_rule_for(self, tensor: TensorInfo) -> TargetSchemeRule | None:
        """Return the last matching rule, after applying global exclusions."""
        if tensor.role != "weight" or not self.target_scheme_rules:
            return None
        tags = tuple(tensor.tags)
        if any(
            BUILTIN_SELECTIONS[item] in set(tags)
            for item in self.exclude_selections
        ):
            return None
        if any(re.search(pattern, tensor.name) for pattern in self.exclude_names):
            return None
        return next(
            (
                rule
                for rule in reversed(self.target_scheme_rules)
                if rule.matches_name(tensor.name, tags)
            ),
            None,
        )
