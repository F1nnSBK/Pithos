"""
Pithos: High-Performance Model-Isomorphic Vector Database Engine
"""

from .core import (
    VectorDb,
    Index,
    DeltaBuffer,
    SearchResult,
    IndexInfo,
    QuantizationMode,
    SidecarMode,
    FpgaDescriptor,
    reset_isolate,
    shrink_to_fit,
)
from .ffi import PithosNativeError

__version__ = "2.2.1"
__all__ = [
    "VectorDb",
    "Index",
    "DeltaBuffer",
    "SearchResult",
    "IndexInfo",
    "QuantizationMode",
    "SidecarMode",
    "FpgaDescriptor",
    "PithosNativeError",
    "reset_isolate",
    "shrink_to_fit",
    "__version__",
]

