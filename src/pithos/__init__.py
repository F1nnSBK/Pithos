# Copyright (c) 2026 Pithos Authors and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
================================================================================
Pithos: High-Performance Model-Isomorphic Vector Database Engine
================================================================================
"""

from .core import (
    VectorDb,
    Index,
    DeltaBuffer,
    SearchResult,
    IndexInfo,
    PlanetaryGridResult,
    QuantizationMode,
    SidecarMode,
    FpgaDescriptor,
    reset_isolate,
    shrink_to_fit,
)
from .ffi import PithosNativeError

__version__ = "2.2.4"
__all__ = [
    "VectorDb",
    "Index",
    "DeltaBuffer",
    "SearchResult",
    "IndexInfo",
    "PlanetaryGridResult",
    "QuantizationMode",
    "SidecarMode",
    "FpgaDescriptor",
    "PithosNativeError",
    "reset_isolate",
    "shrink_to_fit",
    "__version__",
]

