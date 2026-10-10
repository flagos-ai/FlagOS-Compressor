"""Import this module to register built-in format ops."""

from flagos_compressor.formats import fp4_e2m1 as _fp4_e2m1  # noqa: F401
from flagos_compressor.formats import fp8_e8m0 as _fp8_e8m0  # noqa: F401
from flagos_compressor.formats import microscaling as _microscaling  # noqa: F401
from flagos_compressor.formats import bf16 as _bf16  # noqa: F401
from flagos_compressor.formats import floating as _floating  # noqa: F401
from flagos_compressor.formats import int4_pack as _int4_pack  # noqa: F401
from flagos_compressor.formats import int8_pack as _int8_pack  # noqa: F401
from flagos_compressor.formats import compressed_tensors as _compressed_tensors  # noqa: F401
from flagos_compressor.formats import (  # noqa: F401
    compressed_tensors_moe as _compressed_tensors_moe,
)
