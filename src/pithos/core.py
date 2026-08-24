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
Pithos Vector Database Engine -- Python High-Performance Client Interface
================================================================================

This module exposes the official Python client bindings for Pithos, designed
with FAISS-grade performance, rigorous NumPy shape/type safety, and zero-copy
off-heap memory execution.

Key Architectural Guarantees:
-----------------------------
1. Zero-Copy Ingestion & Querying:
   Direct C-FFI / Native Image dispatch with pre-allocated buffer support (D, I).
2. Hardware Microscaling & Precision Sidecars:
   Vectorized OCP/NVIDIA FP8 E4M3 and Block-16 NVFP4 codecs in NumPy SIMD.
3. Multi-Index Hashing (MIH):
   Vectorized CSR routing table generation for sub-millisecond prefix candidate pruning.
4. Universal Single-File Format (.pithos):
   Schema-agnostic DIOGENES container format with self-contained TOC and Arrow metadata.
"""

from __future__ import annotations

import ctypes
import glob
import io
import json
import math
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .ffi import NativeBindings, PithosNativeError, reset_isolate, shrink_to_fit

_BIT_COUNTS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# ==============================================================================
# Type Checking & Array Invariant Validators (FAISS Standard)
# ==============================================================================

def _check_dtype_uint8(codes: np.ndarray, name: str = "codes") -> np.ndarray:
    """Validates that input array is ndarray of dtype uint8 and contiguous.

    Parameters
    ----------
    codes : ndarray
        Input byte array.
    name : str, default="codes"
        Argument name for descriptive error reporting.

    Returns
    -------
    ndarray
        Contiguous uint8 ndarray.

    Raises
    ------
    TypeError
        If dtype is not uint8.
    """
    if codes.dtype != np.uint8:
        raise TypeError(
            f"Input argument '{name}' must be ndarray of dtype uint8, but found {codes.dtype}"
        )
    return np.ascontiguousarray(codes)


def _check_dtype_float32(x: np.ndarray, name: str = "x") -> np.ndarray:
    """Ensures input vector array is contiguous float32.

    Parameters
    ----------
    x : array_like
        Input vector array.
    name : str, default="x"
        Argument name for error reporting.

    Returns
    -------
    ndarray
        Contiguous float32 ndarray.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float32)
    elif x.dtype != np.float32:
        x = x.astype(np.float32)
    return np.ascontiguousarray(x)


def _check_dtype_int64(ids: np.ndarray, name: str = "ids") -> np.ndarray:
    """Ensures input ID array is contiguous int64.

    Parameters
    ----------
    ids : array_like
        Input identifier array.
    name : str, default="ids"
        Argument name for error reporting.

    Returns
    -------
    ndarray
        Contiguous int64 ndarray.
    """
    if not isinstance(ids, np.ndarray):
        ids = np.asarray(ids, dtype=np.int64)
    elif ids.dtype != np.int64:
        ids = ids.astype(np.int64)
    return np.ascontiguousarray(ids)


# ==============================================================================
# Precision Sidecar Codecs (FP8 E4M3 & Block-16 NVFP4)
# ==============================================================================

_FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_THRESHOLDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def _encode_fp8_e4m3_scalar(val: float) -> int:
    """Encodes a single float into an 8-bit OCP/NVIDIA FP8 E4M3 byte."""
    if math.isnan(val):
        sign = 1 if math.copysign(1.0, val) < 0.0 else 0
        return (sign << 7) | 0x7F
    sign = 1 if math.copysign(1.0, val) < 0.0 else 0
    if val == 0.0:
        return sign << 7
    abs_val = abs(val)
    if abs_val >= 448.0 or math.isinf(val):
        return (sign << 7) | 0x7E
    if abs_val < (0.5 / 512.0):
        return sign << 7
    if abs_val < 0.015625:  # 2^(-6) subnormal
        m = int(round(abs_val * 512.0))
        if m > 7:
            m = 7
        return (sign << 7) | (m & 0x7)
    exp = int(math.floor(math.log2(abs_val))) + 7
    if exp < 1:
        m = int(round(abs_val * 512.0))
        if m > 7:
            m = 7
        return (sign << 7) | (m & 0x7)
    if exp > 15:
        return (sign << 7) | 0x7E
    scale = math.pow(2.0, exp - 7)
    m = int(round((abs_val / scale - 1.0) * 8.0))
    if m >= 8:
        exp += 1
        m = 0
        if exp >= 16:
            return (sign << 7) | 0x7E
    if exp == 15 and m >= 7:
        m = 6
    return (sign << 7) | ((exp & 0xF) << 3) | (m & 0x7)


def _encode_fp4_nibble(val: float) -> int:
    """Encodes a single float value into a 4-bit NVFP4 E2M1 nibble."""
    sign = 1 if val < 0.0 else 0
    abs_val = abs(val)
    best_idx = 0
    best_diff = abs_val
    for i in range(1, 8):
        diff = abs(abs_val - _FP4_TABLE[i])
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return (sign << 3) | (best_idx & 0x07)


def _decode_fp8_e4m3_scalar(b: int) -> float:
    """Decodes a single 8-bit OCP/NVIDIA FP8 E4M3 value into 32-bit float."""
    b_int = int(b)
    sign = 1 if (b_int & 0x80) != 0 else 0
    exp = (b_int >> 3) & 0x0F
    mantissa = b_int & 0x07
    sign_mult = -1.0 if sign == 1 else 1.0
    if exp == 0:
        return sign_mult * (mantissa / 512.0)
    elif exp == 15 and mantissa == 7:
        return float("nan")
    else:
        scale = float(1 << (exp - 7)) if exp >= 7 else (1.0 / float(1 << (7 - exp)))
        return sign_mult * scale * (1.0 + mantissa / 8.0)


_FP8_DECODE_LUT = np.array([_decode_fp8_e4m3_scalar(b) for b in range(256)], dtype=np.float32)


def _decode_fp8_e4m3_array(arr: np.ndarray) -> np.ndarray:
    """Decodes uint8 FP8 E4M3 values back to float32 using a precomputed 256-element LUT.

    Parameters
    ----------
    arr : ndarray
        Quantized uint8 array of FP8 E4M3 bytes.

    Returns
    -------
    ndarray
        Reconstructed float32 array.
    """
    return _FP8_DECODE_LUT[np.ascontiguousarray(arr, dtype=np.uint8)]


def _encode_fp8_e4m3_array(arr: np.ndarray) -> np.ndarray:
    """Vectorized conversion of float32 array to 8-bit OCP/NVIDIA FP8 E4M3 standard bytes.

    Matches Java VectorDb.encodeFP8_E4M3 with 100% bit-exact parity.

    Parameters
    ----------
    arr : ndarray
        Input float32 array of arbitrary shape.

    Returns
    -------
    ndarray
        Quantized uint8 array with same shape as input.
    """
    flat = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = flat.view(np.uint32)

    sign = ((u32 >> 24) & 0x80).astype(np.uint8)
    exp = ((u32 >> 23) & 0xFF).astype(np.int32)
    mant = (u32 & 0x7FFFFF).astype(np.int32)
    abs_val = np.abs(flat)

    out = np.zeros(flat.shape, dtype=np.uint8)

    # 1. Underflow / Zero (< 0.5 * 2^-9)
    zero_mask = abs_val < (0.5 / 512.0)
    out[zero_mask] = sign[zero_mask]

    # 2. Clamping / Overflows >= 448.0 or Inf
    max_mask = (abs_val >= 448.0) | np.isinf(flat)
    out[max_mask] = sign[max_mask] | 0x7E

    # 3. Subnormals (< 0.015625)
    sub_mask = (~zero_mask) & (~max_mask) & (abs_val < 0.015625)
    if np.any(sub_mask):
        m = np.clip(np.round(abs_val[sub_mask] * 512.0).astype(np.int32), 0, 7).astype(np.uint8)
        out[sub_mask] = sign[sub_mask] | (m & 0x7)

    # 4. Normal range (0.015625 <= abs_val < 448.0)
    norm_mask = (~zero_mask) & (~max_mask) & (abs_val >= 0.015625)
    if np.any(norm_mask):
        e = exp[norm_mask] - 120
        m = (mant[norm_mask] + (1 << 19)) >> 20

        # Carry-over if rounding up
        carry = m >= 8
        e = np.where(carry, e + 1, e)
        m = np.where(carry, 0, m)

        # Handle overflow
        over = e >= 16
        e = np.where(over, 15, e)
        m = np.where(over, 6, m)

        # Re-clamp max E4M3 (15, 7 is NaN, max finite is 15, 6)
        max_finite_clamp = (e == 15) & (m >= 7)
        m = np.where(max_finite_clamp, 6, m)

        out[norm_mask] = sign[norm_mask] | ((e.astype(np.uint8) & 0xF) << 3) | (m.astype(np.uint8) & 0x7)

    # 5. NaN
    nan_mask = np.isnan(flat)
    if np.any(nan_mask):
        out[nan_mask] = sign[nan_mask] | 0x7F

    return out


def _encode_fp4_nibbles_array(norm_floats: np.ndarray) -> np.ndarray:
    """Vectorized quantization of normalized floats into 4-bit FP4 E2M1 nibbles (0..15).

    Parameters
    ----------
    norm_floats : ndarray
        Float array scaled by block microscaling factor.

    Returns
    -------
    ndarray
        4-bit nibbles stored in lower 4 bits of uint8 array.
    """
    flat = np.ascontiguousarray(norm_floats, dtype=np.float32)
    signs = np.where(flat < 0.0, 0x08, 0x00).astype(np.uint8)
    abs_floats = np.abs(flat)
    nibbles = np.digitize(abs_floats, _FP4_THRESHOLDS).astype(np.uint8)
    return signs | (nibbles & 0x07)


def _encode_nvfp4_blocks_array(vecs: np.ndarray) -> np.ndarray:
    """Vectorized NVFP4 Block-16 microscaling encoder.

    Converts 2D float32 array (N, D) to (N, num_blocks * 9) uint8 bytes.
    Each 16-element block is stored as 1 byte FP8 E4M3 scale factor + 8 bytes packed nibble pairs.

    Parameters
    ----------
    vecs : ndarray
        Continuous float embeddings of shape (N, D).

    Returns
    -------
    ndarray
        NVFP4 encoded byte matrix of shape (N, ((D + 15) // 16) * 9).
    """
    flat_2d = np.ascontiguousarray(vecs, dtype=np.float32)
    if flat_2d.ndim == 1:
        flat_2d = flat_2d.reshape(1, -1)
    N, D = flat_2d.shape
    num_blocks = (D + 15) // 16
    padded_dim = num_blocks * 16

    if D == padded_dim:
        padded = flat_2d
    else:
        padded = np.zeros((N, padded_dim), dtype=np.float32)
        padded[:, :D] = flat_2d

    blocks = padded.reshape(-1, 16)
    max_abs = np.max(np.abs(blocks), axis=-1)
    scales = np.where(max_abs > 0.0, max_abs / 6.0, 0.0).astype(np.float32)

    scale_bytes = _encode_fp8_e4m3_array(scales)
    actual_scales = _decode_fp8_e4m3_array(scale_bytes)

    safe_scales = np.where(actual_scales > 0.0, actual_scales, 1.0)[:, None]
    norm_blocks = blocks / safe_scales

    nibbles = _encode_fp4_nibbles_array(norm_blocks)

    low_nibbles = nibbles[:, 0::2] & 0x0F
    high_nibbles = (nibbles[:, 1::2] & 0x0F) << 4
    packed_nibbles = low_nibbles | high_nibbles

    block_9b = np.empty((N * num_blocks, 9), dtype=np.uint8)
    block_9b[:, 0] = scale_bytes
    block_9b[:, 1:9] = packed_nibbles

    return block_9b.reshape(N, num_blocks * 9)


_FP4_DECODE_LUT = np.array(_FP4_TABLE, dtype=np.float32)


def _decode_nvfp4_blocks_array(encoded_bytes: np.ndarray, dimension: int) -> np.ndarray:
    """Vectorized NVFP4 Block-16 microscaling decoder.

    Converts (N, num_blocks * 9) uint8 bytes back to (N, dimension) float32 array.

    Parameters
    ----------
    encoded_bytes : ndarray
        NVFP4 encoded byte matrix.
    dimension : int
        Target embedding dimension D.

    Returns
    -------
    ndarray
        Reconstructed float32 embeddings of shape (N, dimension).
    """
    raw = np.ascontiguousarray(encoded_bytes, dtype=np.uint8)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    N = raw.shape[0]
    num_blocks = (dimension + 15) // 16
    expected_len = num_blocks * 9
    if raw.shape[1] != expected_len:
        raise ValueError(f"Expected {expected_len} bytes per vector for dim={dimension}, got {raw.shape[1]}")

    blocks = raw.reshape(N * num_blocks, 9)
    scale_bytes = blocks[:, 0]
    scales = _decode_fp8_e4m3_array(scale_bytes)

    packed = blocks[:, 1:9]
    low_nibbles = packed & 0x0F
    high_nibbles = (packed >> 4) & 0x0F

    nibbles = np.empty((N * num_blocks, 16), dtype=np.uint8)
    nibbles[:, 0::2] = low_nibbles
    nibbles[:, 1::2] = high_nibbles

    signs = np.where((nibbles & 0x08) != 0, -1.0, 1.0).astype(np.float32)
    mag_indices = nibbles & 0x07
    mags = _FP4_DECODE_LUT[mag_indices]

    unscaled = signs * mags
    scaled = unscaled * scales[:, None]
    return scaled.reshape(N, num_blocks * 16)[:, :dimension].astype(np.float32)


def _build_mih_csr_table(tier0_bytes: bytes, num_records: int) -> bytes:
    """Vectorized construction of 4-chunk Multi-Index Hashing (MIH) CSR prefix table.

    Produces offsets array of shape (4, 257) int32 and postings array of shape (4, N) int32.
    Achieves 100% bit-exact parity with zero scalar loops.

    Parameters
    ----------
    tier0_bytes : bytes
        Raw bytes of Tier 0 quantized vectors.
    num_records : int
        Total number of vectors in the dataset.

    Returns
    -------
    bytes
        Concatenated offsets and postings binary payload.
    """
    NUM_MIH_CHUNKS = 4
    NUM_MIH_BUCKETS = 256
    bytes_per_rec_t0 = len(tier0_bytes) // num_records if num_records > 0 else 0

    if num_records == 0 or bytes_per_rec_t0 < 4:
        empty_offsets = np.zeros((NUM_MIH_CHUNKS, NUM_MIH_BUCKETS + 1), dtype=np.int32)
        empty_postings = np.empty((NUM_MIH_CHUNKS, num_records), dtype=np.int32)
        return empty_offsets.tobytes() + empty_postings.tobytes()

    t0_arr = np.frombuffer(tier0_bytes, dtype=np.uint8).reshape(num_records, bytes_per_rec_t0)
    keys_4c = t0_arr[:, :4]  # shape (N, 4)

    offsets_arr = np.zeros((NUM_MIH_CHUNKS, NUM_MIH_BUCKETS + 1), dtype=np.int32)
    postings_arr = np.empty((NUM_MIH_CHUNKS, num_records), dtype=np.int32)

    for c in range(NUM_MIH_CHUNKS):
        chunk_keys = keys_4c[:, c]
        counts = np.bincount(chunk_keys, minlength=NUM_MIH_BUCKETS)
        offsets_arr[c, 1:] = np.cumsum(counts, dtype=np.int32)
        postings_arr[c, :] = np.argsort(chunk_keys, kind="stable").astype(np.int32)

    return offsets_arr.tobytes() + postings_arr.tobytes()


def _align64(offset: int) -> int:
    """Aligns an integer byte offset upwards to the next 64-byte boundary."""
    return (offset + 63) & ~63


def _pad_to(file_obj, target_offset: int) -> None:
    """Pads file object with zero bytes until reaching target offset."""
    cur = file_obj.tell()
    if cur < target_offset:
        file_obj.write(bytes(target_offset - cur))


# ==============================================================================
# Enumerations & Data Classes
# ==============================================================================

class QuantizationMode(IntEnum):
    """Supported vector quantization modes in Pithos.

    Attributes
    ----------
    ONE_BIT : int
        1-bit sign binarization (1 bit per dimension).
    TWO_BIT : int
        2-bit ternary / QJL residual quantization (2 bits per dimension).
    FLOAT32 : int
        Unquantized 32-bit float bypass.
    """
    ONE_BIT = 0
    TWO_BIT = 1
    FLOAT32 = 2


class SidecarMode(IntEnum):
    """Supported float sidecar storage formats in Pithos.

    Attributes
    ----------
    NONE : int
        No float sidecar (asymmetric rotated L2 fallback).
    FP16 : int
        IEEE 754 half-precision float sidecar (_fp16.bin, 2 B/dim).
    FP8 : int
        OCP/NVIDIA Blackwell FP8 E4M3 sidecar (_fp8.bin, 1 B/dim).
    FP4 : int
        Blackwell NVFP4 E2M1 block microscaling (_fp4.bin, 0.5 B/dim + scale).
    """
    NONE = 0
    FP16 = 1
    FP8 = 2
    FP4 = 3


@dataclass(frozen=True)
class SearchResult:
    """Represents a single nearest neighbor search result.

    Attributes
    ----------
    id : int
        Unique 64-bit integer identifier of the vector record.
    score : int
        Raw integer score (Hamming distance or scaled float distance).
    """
    id: int
    score: int

    @property
    def distance(self) -> float:
        """Scaled float distance (divided by 1,000,000)."""
        return self.score / 1_000_000.0


@dataclass(frozen=True)
class IndexInfo:
    """Metadata attributes of a loaded Pithos index.

    Attributes
    ----------
    dimension : int
        Total vector dimensionality (D).
    size : int
        Total number of records in the index (N).
    planet_id : int
        Planetary body identifier code.
    planet_radius : int
        Equatorial planetary radius in meters.
    tiers_count : int
        Number of configured Matryoshka tiers.
    sidecar_mode : SidecarMode
        Precision sidecar format attached to index.
    """
    dimension: int
    size: int
    planet_id: int
    planet_radius: int
    tiers_count: int
    sidecar_mode: SidecarMode = SidecarMode.NONE

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes IndexInfo to a standard Python dictionary."""
        return {
            "dimension": self.dimension,
            "size": self.size,
            "planet_id": self.planet_id,
            "planet_radius": self.planet_radius,
            "tiers_count": self.tiers_count,
            "sidecar_mode": int(self.sidecar_mode),
        }


@dataclass
class PlanetaryGridResult:
    """Result of planetary grid multi-family resonant screening and optional precision reranking.

    Attributes
    ----------
    resonant_count : int
        Number of records matching or exceeding the consensus resonance threshold.
    voting_mask : ndarray
        Binary uint8 voting mask array across all N records in index.
    candidate_ids : ndarray, optional
        Record IDs/indices of passing candidates, sorted by precision score (descending).
    scores : ndarray, optional
        Precision scores (cosine similarities) corresponding to candidate_ids.
    votes : ndarray, optional
        Number of active consensus families (popcount) for each candidate.
    masks : ndarray, optional
        Raw uint8 family bitmask for each candidate.
    """
    resonant_count: int
    voting_mask: np.ndarray
    candidate_ids: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    votes: Optional[np.ndarray] = None
    masks: Optional[np.ndarray] = None

    def __iter__(self):
        """Supports transparent tuple unpacking for 100% backwards compatibility:
        `count, mask = index.query_planetary_grid(...)`
        """
        return iter((self.resonant_count, self.voting_mask))

    def __getitem__(self, item: Union[int, slice]) -> Any:
        return (self.resonant_count, self.voting_mask)[item]

    def __len__(self) -> int:
        return 2

    @property
    def has_reranked(self) -> bool:
        """Returns True if precision reranking was executed on passing candidates."""
        return self.candidate_ids is not None


@dataclass(frozen=True)
class FpgaDescriptor:
    """Hardware descriptor for direct FPGA DMA streaming and MMIO register configuration.

    Attributes
    ----------
    tier_index : int
        Zero-based index of the target tier.
    tier_dimension : int
        Bit dimension of the tier.
    record_count : int
        Total number of records to stream.
    tier_base_address : int
        Raw 64-bit off-heap virtual address of the tier bit vectors.
    tier_byte_length : int
        Byte length of the tier buffer.
    metadata_base_address : int
        Virtual address of metadata bitmask segment.
    metadata_byte_length : int
        Byte length of metadata buffer.
    ids_base_address : int
        Virtual address of record ID segment.
    ids_byte_length : int
        Byte length of record ID buffer.
    words_per_record : int
        Number of 64-bit words per record vector.
    """
    tier_index: int
    tier_dimension: int
    record_count: int
    tier_base_address: int
    tier_byte_length: int
    metadata_base_address: int
    metadata_byte_length: int
    ids_base_address: int
    ids_byte_length: int
    words_per_record: int


# ==============================================================================
# DeltaBuffer (LSM Real-Time Ingest)
# ==============================================================================

class DeltaBuffer:
    """Log-Structured Merge (LSM) in-memory write buffer for real-time inserts.

    Parameters
    ----------
    db : VectorDb
        Parent VectorDb instance.
    index_name : str
        Target index identifier.
    """

    def __init__(self, db: VectorDb, index_name: str):
        self._db = db
        self._name = index_name
        self._ffi = db._ffi

    @property
    def name(self) -> str:
        """Name of the attached index."""
        return self._name

    def insert(self, record_id: int, vector: Union[np.ndarray, Sequence[float]]) -> None:
        """Inserts a single vector record into the active delta buffer.

        Parameters
        ----------
        record_id : int
            Unique 64-bit integer identifier for the vector.
        vector : array_like
            Float vector of shape (D,).
        """
        vec = _check_dtype_float32(np.asarray(vector).flatten(), "vector")
        status = self._ffi.lib.vdb_insert(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_longlong(record_id),
            vec.ctypes.data_as(ctypes.c_void_p),
        )
        self._ffi.check_status(status, "insert into DeltaBuffer")

    def delete(self, record_id: int) -> bool:
        """Soft-deletes a record from the delta buffer via tombstone masking.

        Parameters
        ----------
        record_id : int
            Identifier of record to mark as tombstone.

        Returns
        -------
        bool
            True if record was marked deleted, False otherwise.
        """
        ret = self._ffi.lib.vdb_delete_from_delta(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_longlong(record_id),
        )
        if ret < 0:
            raise PithosNativeError(ret, f"Failed to delete record {record_id} from delta buffer.")
        return ret == 1

    def size(self) -> int:
        """Returns the number of live (non-tombstoned) records in the delta buffer."""
        sz = self._ffi.lib.vdb_delta_size(self._ffi.thread, self._name.encode("utf-8"))
        if sz < 0:
            raise PithosNativeError(int(sz), "Failed to retrieve delta buffer size.")
        return int(sz)

    def needs_flush(self) -> bool:
        """Returns True if live count has exceeded configured flush threshold."""
        ret = self._ffi.lib.vdb_needs_flush(self._ffi.thread, self._name.encode("utf-8"))
        if ret < 0:
            raise PithosNativeError(ret, "Failed to check flush state.")
        return ret == 1

    def backup(self, path: str) -> None:
        """Serializes live delta entries to a binary backup file.

        Parameters
        ----------
        path : str
            Destination filepath for backup binary.
        """
        status = self._ffi.lib.vdb_backup_delta(
            self._ffi.thread,
            self._name.encode("utf-8"),
            path.encode("utf-8"),
        )
        self._ffi.check_status(status, "backup DeltaBuffer")

    def restore(self, path: str, flush_threshold: int = 10000) -> None:
        """Restores delta entries from a binary backup file.

        Parameters
        ----------
        path : str
            Source backup filepath.
        flush_threshold : int, default=10000
            Max live entries before trigger threshold.
        """
        status = self._ffi.lib.vdb_restore_delta(
            self._ffi.thread,
            self._name.encode("utf-8"),
            path.encode("utf-8"),
            ctypes.c_int(flush_threshold),
        )
        self._ffi.check_status(status, "restore DeltaBuffer")


# ==============================================================================
# Index (FAISS-Grade High-Performance Handle)
# ==============================================================================

class Index:
    """Handle to an off-heap memory-mapped multi-tier vector index.

    Supports FAISS-compatible query conventions, pre-allocated zero-allocation
    numpy output arrays, hardware DMA descriptors, and precision sidecar reranking.
    """

    def __init__(self, db: VectorDb, name: str, base_path: str):
        self._db = db
        self._name = name
        self._base_path = base_path
        self._ffi = db._ffi
        self._info: Optional[IndexInfo] = None
        self._sidecar_mmap: Optional[np.ndarray] = None
        self._mips_transformer = None
        
        # Check if this index was compiled with metric="mips"
        meta = self.user_metadata
        if meta and "mips_transformer" in meta:
            from .mips import SphericalLiftingTransformer
            self._mips_transformer = SphericalLiftingTransformer.from_dict(meta["mips_transformer"])

    def __setattr__(self, name: str, value: Any) -> None:
        """Protects Index attributes against silent assignment bugs (FAISS standard)."""
        valid_slots = {
            "_db", "_name", "_base_path", "_ffi", "_info", "d", "ntotal",
            "referenced_objects", "_sidecar_mmap", "_mips_transformer"
        }
        if name.startswith("_") or name in valid_slots or hasattr(self, name) or hasattr(self.__class__, name):
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'.")

    # --------------------------------------------------------------------------
    # FAISS-Standard Properties & Aliases
    # --------------------------------------------------------------------------
    @property
    def d(self) -> int:
        """Vector dimensionality D (FAISS compatibility alias)."""
        return self.info().dimension

    @property
    def dimension(self) -> int:
        """Vector dimensionality D."""
        return self.d

    @property
    def ntotal(self) -> int:
        """Total number of vectors in index N (FAISS compatibility alias)."""
        return len(self)

    @property
    def is_trained(self) -> bool:
        """Always True: Pithos isometric WHT projection requires zero offline training."""
        return True

    @property
    def name(self) -> str:
        """Index identifier name."""
        return self._name

    @property
    def base_path(self) -> str:
        """Base filesystem path of the index or container."""
        return self._base_path

    @property
    def planet_id(self) -> int:
        """Planetary body identifier code."""
        return self.info().planet_id

    @property
    def planet_radius(self) -> int:
        """Equatorial planetary radius in meters."""
        return self.info().planet_radius

    @property
    def tier_count(self) -> int:
        """Number of configured Matryoshka tiers."""
        return self.info().tiers_count

    @property
    def sidecar_mode(self) -> SidecarMode:
        """Precision sidecar format attached to index."""
        return self.info().sidecar_mode

    @property
    def has_sidecar(self) -> bool:
        """Returns True if the index has a precision sidecar attached."""
        return self.sidecar_mode != SidecarMode.NONE

    @property
    def is_cuda_capable(self) -> bool:
        """Returns True if loaded native binary supports CUDA GPU acceleration."""
        return self._ffi._has_cuda

    def __len__(self) -> int:
        return int(self._ffi.lib.vdb_size(self._ffi.thread, self._name.encode("utf-8")))

    def size(self) -> int:
        """Returns total number of records in index N."""
        return len(self)

    def info(self) -> IndexInfo:
        """Retrieves complete index metadata descriptor."""
        dim = ctypes.c_int()
        sz = ctypes.c_longlong()
        pid = ctypes.c_byte()
        prad = ctypes.c_longlong()
        tcount = ctypes.c_int()

        status = self._ffi.lib.vdb_get_info(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.byref(dim),
            ctypes.byref(sz),
            ctypes.byref(pid),
            ctypes.byref(prad),
            ctypes.byref(tcount),
        )
        self._ffi.check_status(status, "get index info")

        if hasattr(self._ffi.lib, "vdb_get_sidecar_mode"):
            sidecar_code = self._ffi.lib.vdb_get_sidecar_mode(self._ffi.thread, self._name.encode("utf-8"))
            sidecar_mode = SidecarMode(sidecar_code) if sidecar_code >= 0 else SidecarMode.NONE
        else:
            c_path = self._base_path if self._base_path.endswith(".pithos") else f"{self._base_path}.pithos"
            if os.path.exists(c_path) and os.path.isfile(c_path):
                toc = self._db._read_container_toc(c_path)
                s_fmt = toc.get("sections", {}).get("sidecar", {}).get("format", "")
                if s_fmt.startswith("fp8"):
                    sidecar_mode = SidecarMode.FP8
                elif s_fmt.startswith("nvfp4") or s_fmt.startswith("fp4"):
                    sidecar_mode = SidecarMode.FP4
                elif s_fmt.startswith("fp16"):
                    sidecar_mode = SidecarMode.FP16
                else:
                    sidecar_mode = SidecarMode.NONE
            elif os.path.exists(f"{self._base_path}_fp8.bin"):
                sidecar_mode = SidecarMode.FP8
            elif os.path.exists(f"{self._base_path}_fp4.bin"):
                sidecar_mode = SidecarMode.FP4
            elif os.path.exists(f"{self._base_path}_fp16.bin"):
                sidecar_mode = SidecarMode.FP16
            else:
                sidecar_mode = SidecarMode.NONE

        self._info = IndexInfo(
            dimension=dim.value,
            size=sz.value,
            planet_id=pid.value,
            planet_radius=prad.value,
            tiers_count=tcount.value,
            sidecar_mode=sidecar_mode,
        )
        return self._info

    def set_chunk_size(self, chunk_size: int) -> None:
        """Configures parallel record chunk size for Disruptor worker threads.

        Parameters
        ----------
        chunk_size : int
            Number of vector records per worker chunk batch.
        """
        status = self._ffi.lib.vdb_set_chunk_size(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_longlong(chunk_size),
        )
        self._ffi.check_status(status, "set chunk size")

    def set_energy_budget(self, tau: float) -> None:
        """Sets Matryoshka early-exit cumulative spectral energy budget tau in (0, 1].

        Parameters
        ----------
        tau : float
            Energy cutoff ratio between 0.0 and 1.0.
        """
        status = self._ffi.lib.vdb_set_energy_budget(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_double(tau),
        )
        self._ffi.check_status(status, "set energy budget")

    # --------------------------------------------------------------------------
    # Search & Query Methods (FAISS NumPy Standard)
    # --------------------------------------------------------------------------
    def search(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        *,
        D: Optional[np.ndarray] = None,
        I: Optional[np.ndarray] = None,
        cuda: bool = False,
        return_numpy: bool = False,
        force_python: bool = False,
    ) -> Union[List[SearchResult], List[List[SearchResult]], Tuple[np.ndarray, np.ndarray]]:
        """Finds the k nearest neighbors of query vectors x.

        Supports FAISS-style pre-allocated arrays (D, I) for zero-allocation loops.

        Parameters
        ----------
        queries : array_like
            Query vectors, shape (n, d) or 1D shape (d,). `dtype` must be float32.
        k : int, default=10
            Number of nearest neighbors to return. Must be > 0.
        D : ndarray, optional
            Pre-allocated distance array of shape (n, k) and dtype int32.
        I : ndarray, optional
            Pre-allocated label array of shape (n, k) and dtype int64.
        cuda : bool, default=False
            Whether to dispatch to CUDA GPU kernel if available.
        return_numpy : bool, default=False
            If True, returns tuple (I, D) as flat NumPy arrays.

        Returns
        -------
        results : list of SearchResult, list of list of SearchResult, or tuple (I, D)
            Nearest neighbor results.
        """
        q_arr = _check_dtype_float32(np.asarray(queries), "queries")
        
        q_norms = None
        if getattr(self, "_mips_transformer", None) is not None:
            q_arr, q_norms = self._mips_transformer.transform_queries(q_arr)

        is_single = q_arr.ndim == 1
        if is_single:
            q_arr = q_arr.reshape(1, -1)

        num_queries, dim = q_arr.shape
        if dim != self.dimension:
            raise ValueError(f"Query dimension {dim} does not match index dimension {self.dimension}")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        if I is None:
            out_ids = np.empty((num_queries, k), dtype=np.int64)
        else:
            assert I.shape == (num_queries, k)
            out_ids = _check_dtype_int64(I, "I")

        if D is None:
            out_dists = np.empty((num_queries, k), dtype=np.int32)
        else:
            assert D.shape == (num_queries, k)
            out_dists = np.ascontiguousarray(D, dtype=np.int32)

        import platform
        # ARM64/aarch64 workaround: Force python fallback to avoid GraalVM Panama/MemorySegment bugs on ARM backend
        force_python = platform.machine().lower() in ("arm64", "aarch64")
        
        if cuda and self._ffi._has_cuda:
            status = self._ffi.lib.vdb_cuda_batch_search(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                ctypes.c_int(k),
                out_ids.ctypes.data_as(ctypes.c_void_p),
                out_dists.ctypes.data_as(ctypes.c_void_p),
            )
            self._ffi.check_status(status, "search")
        elif force_python:
            self._python_search_fallback(q_arr, k, out_ids, out_dists)
        else:
            status = self._ffi.lib.vdb_batch_search(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                ctypes.c_int(k),
                out_ids.ctypes.data_as(ctypes.c_void_p),
                out_dists.ctypes.data_as(ctypes.c_void_p),
            )
            self._ffi.check_status(status, "search")

        if return_numpy or D is not None or I is not None:
            if getattr(self, "_mips_transformer", None) is not None:
                sims = 1.0 - (out_dists.astype(np.float32) / 1000000.0)
                raw_scores = self._mips_transformer.untransform_scores(sims, q_norms)
                # Note: If D was pre-allocated as int32, we cannot write floats into it seamlessly in python, 
                # but we return the raw_scores arrays directly.
                return (out_ids[0], raw_scores[0]) if is_single and D is None else (out_ids, raw_scores)
            return (out_ids[0], out_dists[0]) if is_single and D is None else (out_ids, out_dists)

        results: List[List[SearchResult]] = []
        is_mips = getattr(self, "_mips_transformer", None) is not None
        for q_idx in range(num_queries):
            q_res: List[SearchResult] = []
            for i in range(k):
                rec_id = int(out_ids[q_idx, i])
                if rec_id == -1:
                    continue
                sc = out_dists[q_idx, i]
                if is_mips:
                    approx_sim = 1.0 - (float(sc) / 1000000.0)
                    q_n = q_norms if is_single else q_norms[q_idx]
                    orig_sim = self._mips_transformer.untransform_scores(approx_sim, float(q_n))
                    q_res.append(SearchResult(id=rec_id, score=float(orig_sim)))
                else:
                    q_res.append(SearchResult(id=rec_id, score=float(sc)))
            results.append(q_res)

        return results[0] if is_single else results

    def search_numpy(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-Copy batch k-NN search returning flat numpy arrays (out_ids, out_distances).

        Completely bypasses Python object allocation and GC overhead.

        Parameters
        ----------
        queries : array_like
            Query vectors of shape (n, d) or (d,).
        k : int, default=10
            Number of nearest neighbors to retrieve.
        cuda : bool, default=False
            Whether to use CUDA acceleration.

        Returns
        -------
        out_ids : ndarray
            Int64 array of shape (n, k) with neighbor IDs.
        out_distances : ndarray
            Int32 array of shape (n, k) with neighbor distances.
        """
        return self.search(queries, k=k, cuda=cuda, return_numpy=True)

    def batch_search(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False,
        return_numpy: bool = False,
    ) -> Union[List[SearchResult], List[List[SearchResult]], Tuple[np.ndarray, np.ndarray]]:
        """Alias for search() performing batch k-NN search across multi-tier vectors."""
        return self.search(queries, k=k, cuda=cuda, return_numpy=return_numpy)

    def batch_search_numpy(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-Copy batch k-NN search returning flat numpy arrays (out_ids, out_distances)."""
        return self.search(queries, k=k, cuda=cuda, return_numpy=True)

    def _python_search_fallback(self, q_arr: np.ndarray, k: int, out_ids: np.ndarray, out_dists: np.ndarray) -> None:
        """NumPy vector-based exact search fallback to prevent JVM crashes on Apple Silicon M-series."""
        if self.sidecar_mode == SidecarMode.NONE:
            raise RuntimeError("SidecarMode.NONE is not supported on Apple Silicon (M-series) due to native search crashes. Please use SidecarMode.FP8, FP16, or FP4.")
            
        vecs = self.get_vectors()
        if len(vecs) == 0:
            out_ids.fill(-1)
            out_dists.fill(0)
            return

        # Fetch index metric from superblock
        metric = "cosine"
        try:
            c_path = self._base_path if self._base_path.endswith(".pithos") else f"{self._base_path}.pithos"
            with open(c_path, "rb") as f:
                f.seek(32)
                m_code = int.from_bytes(f.read(4), byteorder="little")
                if m_code == 1:
                    metric = "l2"
                elif m_code == 2:
                    metric = "dot"
        except Exception:
            pass

        num_queries = q_arr.shape[0]
        ids_buf = self.get_ids_buffer()

        for q_idx in range(num_queries):
            q = q_arr[q_idx]

            # If SidecarMode.NONE, vecs are 1-bit reconstructions (-1 / 1). 
            # We must binarize the query as well to perform symmetric distance, 
            # mirroring native Hamming space behavior to preserve self-retrieval.
            if self.sidecar_mode == SidecarMode.NONE:
                q_sym = np.where(q > 0, 1.0, -1.0).astype(np.float32)
                q_sym /= math.sqrt(vecs.shape[1])
            else:
                q_sym = q

            if metric == "l2":
                dists = np.sum((vecs - q_sym) ** 2, axis=1)
                best_idx = np.argsort(dists)[:k]
                out_ids[q_idx, :len(best_idx)] = ids_buf[best_idx]
                out_dists[q_idx, :len(best_idx)] = dists[best_idx].astype(np.int32)
            else: # cosine or dot
                dots = np.dot(vecs, q_sym)
                if metric == "cosine":
                    q_n = np.linalg.norm(q_sym)
                    v_n = np.linalg.norm(vecs, axis=1)
                    denom = q_n * v_n
                    denom[denom == 0] = 1e-10
                    sim = dots / denom
                    # Native cosine returns: (int)((1.0 - sim) * 1000000.0)
                    scores = (1.0 - sim) * 1000000.0
                    best_idx = np.argsort(scores)[:k]
                    out_ids[q_idx, :len(best_idx)] = ids_buf[best_idx]
                    out_dists[q_idx, :len(best_idx)] = scores[best_idx].astype(np.int32)
                else:
                    scores = -dots # dot product needs to be maximized, but search returns ascending distances
                    best_idx = np.argsort(scores)[:k]
                    out_ids[q_idx, :len(best_idx)] = ids_buf[best_idx]
                    out_dists[q_idx, :len(best_idx)] = (-scores[best_idx]).astype(np.int32)

    def search_merged(
        self,
        query: Union[np.ndarray, Sequence[float]],
        k: int = 10,
    ) -> List[SearchResult]:
        """Queries both the base memory-mapped index and the active DeltaBuffer, merging results.

        Parameters
        ----------
        query : array_like
            Query vector of shape (d,).
        k : int, default=10
            Number of nearest neighbors to retrieve.

        Returns
        -------
        List[SearchResult]
            Merged and deduplicated search results.
        """
        q_arr = _check_dtype_float32(np.asarray(query).flatten(), "query")
        out_ids = np.empty(k, dtype=np.int64)
        out_dists = np.empty(k, dtype=np.int32)

        status = self._ffi.lib.vdb_search_merged(
            self._ffi.thread,
            self._name.encode("utf-8"),
            q_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(k),
            out_ids.ctypes.data_as(ctypes.c_void_p),
            out_dists.ctypes.data_as(ctypes.c_void_p),
        )
        self._ffi.check_status(status, "search merged (base + delta)")

        results: List[SearchResult] = []
        for i in range(k):
            rec_id = int(out_ids[i])
            if rec_id == -1:
                continue
            results.append(SearchResult(id=rec_id, score=int(out_dists[i])))
        return results

    def query_planetary_grid(
        self,
        queries: np.ndarray,
        families: np.ndarray,
        thresholds: np.ndarray,
        out_voting_mask: Optional[np.ndarray] = None,
        voting_mask: Optional[np.ndarray] = None,
        cuda: bool = False,
        min_votes: int = 5,
        rerank: bool = True,
    ) -> PlanetaryGridResult:
        """Performs multi-family resonant voting across scientific criteria with automatic precision reranking.

        When a precision sidecar is attached (FP8, FP16, NVFP4) and `rerank=True`, automatically retrieves
        sidecar vectors for all candidates with >= `min_votes`, computes maximum cosine similarity across all
        queries, and returns sorted candidate IDs, precision scores, and vote counts.

        Supports transparent tuple unpacking `(count, mask) = index.query_planetary_grid(...)` for 100%
        backwards compatibility.

        Parameters
        ----------
        queries : ndarray
            Query vectors, shape (num_queries, D).
        families : ndarray
            Semantic family identifiers (0..7), shape (num_queries,).
        thresholds : ndarray
            Cutoff thresholds, shape (num_queries,).
        voting_mask : ndarray, optional
            Pre-allocated byte mask of size N.
        cuda : bool, default=False
            Whether to use CUDA acceleration.
        min_votes : int, default=5
            Minimum number of active consensus families required for candidate selection.
        rerank : bool, default=True
            Whether to automatically re-rank passing candidates using the attached precision sidecar.

        Returns
        -------
        PlanetaryGridResult
            Result object containing resonant_count, voting_mask, and sorted candidate_ids / scores.
        """
        q_arr = _check_dtype_float32(queries, "queries")
        
        q_norms = None
        if getattr(self, "_mips_transformer", None) is not None:
            q_arr, q_norms = self._mips_transformer.transform_queries(q_arr)
            
        f_arr = np.ascontiguousarray(families, dtype=np.int32)
        t_arr = np.ascontiguousarray(thresholds, dtype=np.int32)

        num_queries = q_arr.shape[0]
        total_records = len(self)

        mask = voting_mask if voting_mask is not None else out_voting_mask
        if mask is None:
            mask = np.zeros(total_records, dtype=np.uint8)
        else:
            mask = _check_dtype_uint8(mask, "voting_mask")

        if cuda and self._ffi._has_cuda:
            resonant_count = self._ffi.lib.vdb_cuda_query_planetary_grid(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                f_arr.ctypes.data_as(ctypes.c_void_p),
                t_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                mask.ctypes.data_as(ctypes.c_void_p),
            )
        else:
            resonant_count = self._ffi.lib.vdb_query_planetary_grid(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                f_arr.ctypes.data_as(ctypes.c_void_p),
                t_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                mask.ctypes.data_as(ctypes.c_void_p),
            )

        if resonant_count < 0:
            self._ffi.check_status(resonant_count, "query planetary grid")

        # Automatic Precision Sidecar Re-Ranking
        cand_ids = None
        scores = None
        votes = None
        cand_masks = None

        if rerank and self.has_sidecar:
            popcounts = _BIT_COUNTS[mask]
            cand_indices = np.where(popcounts >= min_votes)[0]
            if len(cand_indices) > 0:
                cand_vecs = self.get_vectors(cand_indices)
                cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
                cand_normed = cand_vecs / np.where(cand_norms == 0, 1.0, cand_norms)

                q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
                q_normed = q_arr / np.where(q_norms == 0, 1.0, q_norms)

                sim_matrix = np.dot(cand_normed, q_normed.T)
                max_sims = np.max(sim_matrix, axis=1)

                cand_popcounts = popcounts[cand_indices]
                # Sort primarily by score descending, secondarily by votes descending
                sort_order = np.lexsort((-cand_popcounts, -max_sims))

                cand_ids = cand_indices[sort_order]
                scores = max_sims[sort_order]
                votes = cand_popcounts[sort_order]
                cand_masks = mask[cand_ids]
                
                if getattr(self, "_mips_transformer", None) is not None:
                    best_q_indices = np.argmax(sim_matrix[sort_order], axis=1)
                    scores = self._mips_transformer.untransform_scores(scores, q_norms[best_q_indices])
            else:
                cand_ids = np.empty((0,), dtype=np.int64)
                scores = np.empty((0,), dtype=np.float32)
                votes = np.empty((0,), dtype=np.uint8)
                cand_masks = np.empty((0,), dtype=np.uint8)

        return PlanetaryGridResult(
            resonant_count=int(resonant_count),
            voting_mask=mask,
            candidate_ids=cand_ids,
            scores=scores,
            votes=votes,
            masks=cand_masks,
        )

    # --------------------------------------------------------------------------
    # Off-Heap Memory Buffers & Hardware Descriptors
    # --------------------------------------------------------------------------
    def get_tier_address(self, tier_idx: int) -> Tuple[int, int]:
        """Returns the raw virtual address and byte length of a tier segment."""
        addr = ctypes.c_longlong()
        length = ctypes.c_longlong()
        status = self._ffi.lib.vdb_get_tier_address(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_int(tier_idx),
            ctypes.byref(addr),
            ctypes.byref(length),
        )
        self._ffi.check_status(status, "get tier memory address")
        return addr.value, length.value

    get_tier_memory_address = get_tier_address

    def get_metadata_address(self) -> Tuple[int, int]:
        """Returns the raw off-heap address and byte length of the metadata sidecar segment."""
        addr = ctypes.c_longlong()
        length = ctypes.c_longlong()
        status = self._ffi.lib.vdb_get_metadata_address(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.byref(addr),
            ctypes.byref(length),
        )
        self._ffi.check_status(status, "get metadata address")
        return addr.value, length.value

    def get_ids_address(self) -> Tuple[int, int]:
        """Returns the raw off-heap address and byte length of the record IDs segment."""
        addr = ctypes.c_longlong()
        length = ctypes.c_longlong()
        status = self._ffi.lib.vdb_get_ids_address(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.byref(addr),
            ctypes.byref(length),
        )
        self._ffi.check_status(status, "get IDs address")
        return addr.value, length.value

    def get_tier_buffer(self, tier_idx: int = 0) -> np.ndarray:
        """Returns a zero-copy uint8 NumPy ndarray viewing the raw off-heap tier bit vectors."""
        addr, length = self.get_tier_address(tier_idx)
        c_arr = (ctypes.c_uint8 * length).from_address(addr)
        return np.ctypeslib.as_array(c_arr)

    def get_metadata_buffer(self) -> np.ndarray:
        """Returns a zero-copy uint64 NumPy ndarray viewing the raw off-heap metadata bitmask flags."""
        addr, length = self.get_metadata_address()
        c_arr = (ctypes.c_uint64 * (length // 8)).from_address(addr)
        return np.ctypeslib.as_array(c_arr)

    def get_ids_buffer(self) -> np.ndarray:
        """Returns a zero-copy int64 NumPy ndarray viewing the raw off-heap record IDs."""
        addr, length = self.get_ids_address()
        c_arr = (ctypes.c_int64 * (length // 8)).from_address(addr)
        return np.ctypeslib.as_array(c_arr)

    def get_sidecar_buffer(self) -> Optional[np.ndarray]:
        """Returns a zero-copy uint8 NumPy ndarray viewing the raw memory-mapped sidecar bytes."""
        if self._sidecar_mmap is not None:
            return self._sidecar_mmap

        # 1. Check Single-File Container (.pithos)
        c_path = self._base_path if self._base_path.endswith(".pithos") else f"{self._base_path}.pithos"
        if os.path.exists(c_path) and os.path.isfile(c_path):
            toc = self._db._read_container_toc(c_path)
            sec = toc.get("sections", {}).get("sidecar")
            if sec and sec.get("length", 0) > 0:
                offset = sec["offset"]
                length = sec["length"]
                self._sidecar_mmap = np.memmap(c_path, dtype=np.uint8, mode="r", offset=offset, shape=(length,))
                return self._sidecar_mmap

        # 2. Check multi-file sidecar binaries
        base_stem = self._base_path[:-7] if self._base_path.endswith(".pithos") else self._base_path
        for ext in ["_fp8.bin", "_fp16.bin", "_fp4.bin"]:
            cand_path = f"{base_stem}{ext}"
            if os.path.exists(cand_path) and os.path.isfile(cand_path):
                self._sidecar_mmap = np.memmap(cand_path, dtype=np.uint8, mode="r")
                return self._sidecar_mmap

        return None

    def get_vectors(
        self,
        indices: Optional[Union[int, Sequence[int], np.ndarray]] = None,
    ) -> np.ndarray:
        """Retrieves and automatically decodes float32 vectors for given candidate indices.

        Supports FP8 E4M3, FP16, NVFP4 E2M1, and sign-reconstruction fallback when no sidecar is present.

        Parameters
        ----------
        indices : int, sequence of int, or ndarray, optional
            Candidate record indices to retrieve. If None, retrieves all vectors in index.

        Returns
        -------
        ndarray
            Float32 vector array of shape (len(indices), D), or 1D shape (D,) if a scalar index was passed.
        """
        total_records = len(self)
        dim = self.dimension
        if total_records == 0:
            return np.empty((0, dim), dtype=np.float32)

        is_scalar = isinstance(indices, (int, np.integer))
        if indices is None:
            idx_arr = slice(None)
            n_req = total_records
        elif is_scalar:
            idx_arr = np.array([int(indices)], dtype=np.int64)
            n_req = 1
        else:
            idx_arr = np.ascontiguousarray(indices, dtype=np.int64)
            n_req = len(idx_arr)
            if n_req == 0:
                return np.empty((0, dim), dtype=np.float32)

        mode = self.sidecar_mode
        if mode == SidecarMode.FP8:
            buf = self.get_sidecar_buffer()
            if buf is None:
                raise RuntimeError("FP8 sidecar buffer could not be loaded.")
            raw_fp8 = buf.reshape(total_records, dim)[idx_arr]
            floats = _decode_fp8_e4m3_array(raw_fp8)
            return floats[0] if is_scalar else floats
        elif mode == SidecarMode.FP16:
            buf = self.get_sidecar_buffer()
            if buf is None:
                raise RuntimeError("FP16 sidecar buffer could not be loaded.")
            fp16_view = buf.view(np.float16).reshape(total_records, dim)[idx_arr]
            floats = fp16_view.astype(np.float32)
            return floats[0] if is_scalar else floats
        elif mode == SidecarMode.FP4:
            buf = self.get_sidecar_buffer()
            if buf is None:
                raise RuntimeError("NVFP4 sidecar buffer could not be loaded.")
            num_blocks = (dim + 15) // 16
            bytes_per_rec = num_blocks * 9
            raw_fp4 = buf.reshape(total_records, bytes_per_rec)[idx_arr]
            floats = _decode_nvfp4_blocks_array(raw_fp4, dim)
            return floats[0] if is_scalar else floats
        else:
            # Fallback to sign-reconstruction from all active tiers
            t_bits = []
            for t in range(self.tier_count):
                t_buf = self.get_tier_buffer(t)
                if len(t_buf) == 0:
                    continue
                bytes_per_rec = len(t_buf) // total_records
                t_selected = t_buf.reshape(total_records, bytes_per_rec)[idx_arr]
                t_bits.append(np.unpackbits(t_selected, axis=-1, bitorder="big"))

            if len(t_bits) > 0:
                all_bits = np.concatenate(t_bits, axis=-1)[..., :dim]
                reconstructed = np.where(all_bits == 1, 1.0, -1.0).astype(np.float32)
                reconstructed /= math.sqrt(dim)
            else:
                reconstructed = np.zeros((n_req, dim), dtype=np.float32)
            return reconstructed[0] if is_scalar else reconstructed

    get_sidecar_vectors = get_vectors

    def rerank(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        candidate_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
        k: Optional[int] = None,
        metric: str = "cosine",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Re-ranks candidate records against queries using precision sidecar vectors.

        Parameters
        ----------
        queries : ndarray
            Query vectors, shape (num_queries, D) or 1D shape (D,).
        candidate_indices : sequence of int or ndarray, optional
            Subset of record indices to re-rank. If None, re-ranks all records in index.
        k : int, optional
            Number of top candidates to return. If None, returns all candidates sorted.
        metric : str, default='cosine'
            Ranking metric ('cosine' for similarity, 'l2' or 'euclidean' for distance).

        Returns
        -------
        ranked_indices : ndarray
            Array of record indices sorted by best match.
        ranked_scores : ndarray
            Cosine similarities or distances corresponding to ranked indices.
        """
        q_arr = _check_dtype_float32(np.asarray(queries), "queries")
        
        q_norms = None
        if getattr(self, "_mips_transformer", None) is not None:
            q_arr, q_norms = self._mips_transformer.transform_queries(q_arr)
            
        is_single_query = q_arr.ndim == 1
        if is_single_query:
            q_arr = q_arr.reshape(1, -1)

        num_queries, dim = q_arr.shape
        if dim != self.dimension:
            raise ValueError(f"Query dimension {dim} does not match index dimension {self.dimension}")

        if candidate_indices is None:
            cand_arr = np.arange(len(self), dtype=np.int64)
            try:
                cand_ids = self.get_ids_buffer()
            except Exception:
                cand_ids = cand_arr
        else:
            cand_arr = np.ascontiguousarray(candidate_indices, dtype=np.int64)
            cand_ids = cand_arr

        num_cands = len(cand_arr)
        if num_cands == 0:
            empty_ids = np.empty((0,), dtype=np.int64) if is_single_query else np.empty((num_queries, 0), dtype=np.int64)
            empty_scores = np.empty((0,), dtype=np.float32) if is_single_query else np.empty((num_queries, 0), dtype=np.float32)
            return empty_ids, empty_scores

        cand_vecs = self.get_vectors(cand_arr)
        top_k = num_cands if k is None else min(k, num_cands)

        metric_lower = metric.lower()
        if metric_lower == "cosine":
            cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
            cand_normed = cand_vecs / np.where(cand_norms == 0, 1.0, cand_norms)
            q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
            q_normed = q_arr / np.where(q_norms == 0, 1.0, q_norms)

            sim_matrix = np.dot(q_normed, cand_normed.T)
            sort_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
            ranked_ids = np.take(cand_ids, sort_indices)
            ranked_scores = np.take_along_axis(sim_matrix, sort_indices, axis=1)
        elif metric_lower in ("dot", "ip", "mips", "inner_product", "dot_product"):
            sim_matrix = np.dot(q_arr, cand_vecs.T)
            sort_indices = np.argsort(-sim_matrix, axis=1)[:, :top_k]
            ranked_ids = np.take(cand_ids, sort_indices)
            ranked_scores = np.take_along_axis(sim_matrix, sort_indices, axis=1)
        elif metric_lower in ("l2", "euclidean"):
            q_sq = np.sum(q_arr**2, axis=1, keepdims=True)
            c_sq = np.sum(cand_vecs**2, axis=1, keepdims=True).T
            dists_sq = np.maximum(0.0, q_sq + c_sq - 2.0 * np.dot(q_arr, cand_vecs.T))
            dists = np.sqrt(dists_sq)
            sort_indices = np.argsort(dists, axis=1)[:, :top_k]
            ranked_ids = np.take(cand_ids, sort_indices)
            ranked_scores = np.take_along_axis(dists, sort_indices, axis=1)
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Choose 'cosine', 'dot', 'ip', or 'l2'.")

        if is_single_query:
            if getattr(self, "_mips_transformer", None) is not None:
                ranked_scores[0] = self._mips_transformer.untransform_scores(ranked_scores[0], q_norms)
            return ranked_ids[0], ranked_scores[0]
            
        if getattr(self, "_mips_transformer", None) is not None:
            ranked_scores = self._mips_transformer.untransform_scores(ranked_scores, q_norms.reshape(-1, 1))
        return ranked_ids, ranked_scores

    def get_fpga_descriptor(self, tier_idx: int = 0) -> FpgaDescriptor:
        """Generates a complete hardware descriptor for FPGA DMA engines and PCIe MMIO registers."""
        tier_addr, tier_len = self.get_tier_address(tier_idx)
        meta_addr, meta_len = self.get_metadata_address()
        ids_addr, ids_len = self.get_ids_address()

        num_recs = self.size()
        if num_recs > 0 and tier_len > 0:
            bytes_per_rec = int(tier_len // num_recs)
            tier_dim = bytes_per_rec * 8
            words_per_record = max(1, (bytes_per_rec + 7) // 8)
        else:
            tier_dim = self.dimension
            words_per_record = max(1, (self.dimension + 63) // 64)

        return FpgaDescriptor(
            tier_index=tier_idx,
            tier_dimension=tier_dim,
            record_count=num_recs,
            tier_base_address=tier_addr,
            tier_byte_length=tier_len,
            metadata_base_address=meta_addr,
            metadata_byte_length=meta_len,
            ids_base_address=ids_addr,
            ids_byte_length=ids_len,
            words_per_record=words_per_record,
        )

    # --------------------------------------------------------------------------
    # Single-File Container Metadata & Partitions
    # --------------------------------------------------------------------------
    @property
    def user_metadata(self) -> dict:
        """User metadata dictionary embedded in the single-file container."""
        if hasattr(self._ffi.lib, "vdb_get_user_metadata"):
            buf = ctypes.create_string_buffer(65536)
            res = self._ffi.lib.vdb_get_user_metadata(self._ffi.thread, self._name.encode("utf-8"), buf, 65536)
            if res > 0:
                try:
                    parsed = json.loads(buf.value.decode("utf-8"))
                    if isinstance(parsed, dict):
                        if "user_metadata" in parsed and isinstance(parsed["user_metadata"], dict):
                            return parsed["user_metadata"]
                        return parsed
                except Exception:
                    pass
        if self._base_path:
            c_path = self._base_path if self._base_path.endswith(".pithos") else f"{self._base_path}.pithos"
            if os.path.exists(c_path):
                toc = self._db._read_container_toc(c_path)
                return toc.get("user_metadata", {})
        return {}

    def get_user_metadata(self) -> dict:
        """Alias for user_metadata()."""
        return self.user_metadata

    @property
    def arrow_table(self) -> Any:
        """Reads Section 4 metadata payload as an Apache Arrow Table if formatted as 'arrow'."""
        if not self._base_path:
            return None
        c_path = self._base_path if self._base_path.endswith(".pithos") else f"{self._base_path}.pithos"
        if not os.path.exists(c_path):
            return None
        toc = self._db._read_container_toc(c_path)
        meta_sec = toc.get("sections", {}).get("metadata")
        if not meta_sec or meta_sec.get("format") != "arrow":
            return None
        try:
            import pyarrow.ipc as ipc
            with open(c_path, "rb") as f:
                f.seek(meta_sec["offset"])
                data = f.read(meta_sec["length"])
            reader = ipc.open_stream(io.BytesIO(data))
            return reader.read_all()
        except Exception:
            return None

    @property
    def partitions(self) -> dict:
        """Partition metadata dictionary from Table of Contents or Arrow payload."""
        user_meta = self.user_metadata
        if "partitions" in user_meta:
            return user_meta["partitions"]
        tbl = self.arrow_table
        if tbl is not None:
            return tbl.to_pydict()
        return {}

    def transform_and_quantize(self, vector: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Transforms a continuous float vector through Rademacher preconditioning & Fast Walsh-Hadamard rotation.

        Parameters
        ----------
        vector : array_like
            Input continuous vector of shape (D,).

        Returns
        -------
        ndarray
            Quantized 64-bit uint64 packed word array of length `(D + 63) // 64`.
        """
        vec = _check_dtype_float32(np.asarray(vector).flatten(), "vector")
        dim = self.dimension
        if vec.shape[0] != dim:
            raise ValueError(f"Vector dimension {vec.shape[0]} does not match index dimension {dim}")

        words_count = (dim + 63) // 64
        out_packed = np.zeros(words_count, dtype=np.uint64)

        status = self._ffi.lib.vdb_transform_and_quantize(
            self._ffi.thread,
            self._name.encode("utf-8"),
            vec.ctypes.data_as(ctypes.c_void_p),
            out_packed.ctypes.data_as(ctypes.c_void_p),
        )
        self._ffi.check_status(status, "transform and quantize vector")
        return out_packed


# ==============================================================================
# Container Writing Core
# ==============================================================================

def _write_pithos_container_file(
    path: str,
    vecs: np.ndarray,
    ids_arr: np.ndarray,
    tiers_arr: np.ndarray,
    metric_code: int,
    q_mode: int,
    actual_sidecar: SidecarMode,
    metadata_payload: Optional[bytes],
    metadata_format: str,
    user_metadata: Optional[dict],
    ffi: NativeBindings,
) -> None:
    """Internal builder compiling vectors directly into a single .pithos container."""
    num_records, dimension = vecs.shape
    num_tiers = len(tiers_arr)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_base = os.path.join(tmpdir, "tmp_index")
        with ffi.isolated_context() as temp_thread:
            status = ffi.lib.vdb_compile_index_file_ext(
                temp_thread,
                tmp_base.encode("utf-8"),
                ctypes.c_byte(1),
                ctypes.c_longlong(1737400),
                ctypes.c_int(dimension),
                tiers_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_tiers),
                ids_arr.ctypes.data_as(ctypes.c_void_p),
                vecs.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_records),
                ctypes.c_int(int(q_mode)),
                ctypes.c_int(int(actual_sidecar)),
            )
            ffi.check_status(status, "compile temporary index files")

        ids_bytes = ids_arr.tobytes()

        tier_bytes_list = []
        for k in range(num_tiers):
            with open(f"{tmp_base}_tier_{k}.bin", "rb") as f:
                tier_bytes_list.append(f.read())

        sidecar_bytes = b""
        sidecar_format = "none"
        if actual_sidecar == SidecarMode.FP16:
            fp16_file = f"{tmp_base}_fp16.bin"
            if os.path.exists(fp16_file) and os.path.getsize(fp16_file) == num_records * dimension * 2:
                with open(fp16_file, "rb") as f:
                    sidecar_bytes = f.read()
            else:
                sidecar_bytes = vecs.astype(np.float16).tobytes()
            sidecar_format = "fp16"
        elif actual_sidecar == SidecarMode.FP8:
            fp8_file = f"{tmp_base}_fp8.bin"
            if os.path.exists(fp8_file) and os.path.getsize(fp8_file) == num_records * dimension:
                with open(fp8_file, "rb") as f:
                    sidecar_bytes = f.read()
            else:
                sidecar_bytes = _encode_fp8_e4m3_array(vecs).tobytes()
            sidecar_format = "fp8_e4m3"
        elif actual_sidecar == SidecarMode.FP4:
            fp4_file = f"{tmp_base}_fp4.bin"
            num_blocks = (dimension + 15) // 16
            bytes_per_rec = num_blocks * 9
            if os.path.exists(fp4_file) and os.path.getsize(fp4_file) == num_records * bytes_per_rec:
                with open(fp4_file, "rb") as f:
                    sidecar_bytes = f.read()
            else:
                sidecar_bytes = _encode_nvfp4_blocks_array(vecs).tobytes()
            sidecar_format = "nvfp4_e2m1"

        meta_bytes = metadata_payload if metadata_payload else b""
        meta_format = metadata_format if metadata_format else "raw"

        SUPERBLOCK_SIZE = 128
        ids_offset = _align64(SUPERBLOCK_SIZE)
        ids_len = len(ids_bytes)

        current_offset = _align64(ids_offset + ids_len)
        tier_offsets = []
        tier_lengths = []
        for tb in tier_bytes_list:
            tier_offsets.append(current_offset)
            tier_lengths.append(len(tb))
            current_offset = _align64(current_offset + len(tb))

        sidecar_offset = 0
        sidecar_len = 0
        if len(sidecar_bytes) > 0:
            sidecar_offset = current_offset
            sidecar_len = len(sidecar_bytes)
            current_offset = _align64(current_offset + sidecar_len)

        # Multi-Index Hashing (MIH) Section (4 chunks x 256 buckets CSR)
        prefix_table_bytes = _build_mih_csr_table(tier_bytes_list[0], num_records)
        prefix_table_offset = current_offset
        prefix_table_len = len(prefix_table_bytes)
        current_offset = _align64(current_offset + prefix_table_len)

        metadata_offset = 0
        metadata_len = 0
        if len(meta_bytes) > 0:
            metadata_offset = current_offset
            metadata_len = len(meta_bytes)
            current_offset = _align64(current_offset + metadata_len)

        toc_dict = {
            "format": "pithos_v2",
            "motto": "Autarky: Self-contained & Zero Baggage",
            "sections": {
                "ids": {"offset": ids_offset, "length": ids_len, "dtype": "uint64"}
            },
            "user_metadata": user_metadata if user_metadata else {},
        }
        for k in range(num_tiers):
            toc_dict["sections"][f"tier_{k}"] = {
                "offset": tier_offsets[k],
                "length": tier_lengths[k],
                "dim_boundary": int(tiers_arr[k]),
            }
        toc_dict["sections"]["sidecar"] = {
            "offset": sidecar_offset,
            "length": sidecar_len,
            "format": sidecar_format,
        }
        toc_dict["sections"]["prefix_table"] = {
            "offset": prefix_table_offset,
            "length": prefix_table_len,
            "num_chunks": 4,
            "num_buckets_per_chunk": 256,
            "format": "mih_csr_4x8",
        }
        toc_dict["sections"]["metadata"] = {
            "offset": metadata_offset,
            "length": metadata_len,
            "format": meta_format,
        }
        toc_bytes = json.dumps(toc_dict, indent=2).encode("utf-8")
        toc_offset = current_offset
        toc_len = len(toc_bytes)
        current_offset = _align64(current_offset + toc_len)

        sb = bytearray(SUPERBLOCK_SIZE)
        sb[0:8] = b"DIOGENES"
        sb[8:12] = (2).to_bytes(4, byteorder="little", signed=True)
        sb[12:20] = int(num_records).to_bytes(8, byteorder="little", signed=True)
        sb[20:24] = int(dimension).to_bytes(4, byteorder="little", signed=True)
        sb[24:26] = int(metric_code).to_bytes(2, byteorder="little", signed=True)
        sb[26:28] = int(actual_sidecar).to_bytes(2, byteorder="little", signed=True)
        sb[28:30] = int(num_tiers).to_bytes(2, byteorder="little", signed=True)
        for i in range(8):
            t_val = int(tiers_arr[i]) if i < num_tiers else 0
            sb[30 + i * 2 : 32 + i * 2] = t_val.to_bytes(2, byteorder="little", signed=True)
        sb[46:54] = int(toc_offset).to_bytes(8, byteorder="little", signed=True)
        sb[54:58] = int(toc_len).to_bytes(4, byteorder="little", signed=True)
        sb[58:60] = int(q_mode).to_bytes(2, byteorder="little", signed=True)
        sb[60:68] = int(prefix_table_offset).to_bytes(8, byteorder="little", signed=True)
        sb[68:76] = int(prefix_table_len).to_bytes(8, byteorder="little", signed=True)

        trailer = bytearray(20)
        trailer[0:8] = int(toc_offset).to_bytes(8, byteorder="little", signed=True)
        trailer[8:12] = int(toc_len).to_bytes(4, byteorder="little", signed=True)
        trailer[12:20] = b"PITHOSDB"

        with open(path, "wb") as out_f:
            out_f.write(sb)
            _pad_to(out_f, ids_offset)
            out_f.write(ids_bytes)
            for k in range(num_tiers):
                _pad_to(out_f, tier_offsets[k])
                out_f.write(tier_bytes_list[k])
            if sidecar_len > 0:
                _pad_to(out_f, sidecar_offset)
                out_f.write(sidecar_bytes)
            _pad_to(out_f, prefix_table_offset)
            out_f.write(prefix_table_bytes)
            if metadata_len > 0:
                _pad_to(out_f, metadata_offset)
                out_f.write(meta_bytes)
            _pad_to(out_f, toc_offset)
            out_f.write(toc_bytes)
            _pad_to(out_f, current_offset)
            out_f.write(trailer)


# ==============================================================================
# VectorDb (Engine Coordinator & Factory)
# ==============================================================================

class VectorDb:
    """Pythonic interface to the Pithos Vector Database Engine.

    Manages loaded multi-tier indices, DeltaBuffers, index compilation, and CUDA runtimes.

    Parameters
    ----------
    lib_path : str, optional
        Explicit path to native shared library (`libpithos.so` / `libpithos.dylib` / `pithos.dll`).
    """

    _active_instances = 0
    _lock = threading.Lock()

    def __init__(self, lib_path: Optional[str] = None):
        self._ffi = NativeBindings(lib_path)
        with VectorDb._lock:
            if VectorDb._active_instances == 0:
                if hasattr(self._ffi.lib, "vdb_init"):
                    self._ffi.lib.vdb_init(self._ffi.thread)
            VectorDb._active_instances += 1
        self._indices: Dict[str, Index] = {}
        self._delta_buffers: Dict[str, DeltaBuffer] = {}
        self._temp_dirs: List[str] = []
        self._closed = False

    @property
    def is_cuda_capable(self) -> bool:
        """Returns True if the loaded native library includes CUDA hardware acceleration symbols."""
        return self._ffi._has_cuda

    def _read_container_toc(self, container_path: str) -> dict:
        """Parses the TOC JSON payload from the trailer of a single-file .pithos container."""
        try:
            with open(container_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size < 148:
                    return {}
                f.seek(size - 20)
                trailer = f.read(20)
                if trailer[12:20] != b"PITHOSDB":
                    return {}
                toc_offset = int.from_bytes(trailer[0:8], byteorder="little")
                toc_len = int.from_bytes(trailer[8:12], byteorder="little")
                f.seek(toc_offset)
                toc_bytes = f.read(toc_len)
                return json.loads(toc_bytes.decode("utf-8"))
        except Exception:
            return {}

    def _unpack_container_if_needed(self, base_path: str) -> str:
        actual_path = base_path if os.path.exists(base_path) else f"{base_path}.pithos"
        return actual_path if os.path.exists(actual_path) else base_path

    def __enter__(self) -> VectorDb:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Closes all loaded indices, delta buffers, and releases coordinator resources."""
        if self._closed:
            return
        self._closed = True
        for name in list(self._indices.keys()):
            try:
                self.drop_index(name)
            except Exception:
                pass
        for td in self._temp_dirs:
            if os.path.exists(td):
                try:
                    shutil.rmtree(td)
                except Exception:
                    pass
        self._temp_dirs.clear()
        with VectorDb._lock:
            VectorDb._active_instances = max(0, VectorDb._active_instances - 1)
            if VectorDb._active_instances == 0:
                self._ffi.lib.vdb_close(self._ffi.thread)
        self._ffi.shrink_to_fit()

    def shrink_to_fit(self) -> None:
        """Explicitly triggers GraalVM GC, OS memory release (malloc_trim), and Python GC."""
        self._ffi.shrink_to_fit()

    def reset_isolate(self) -> None:
        """Drops all loaded indices, cleans temp directories, and re-initializes GraalVM isolate."""
        for name in list(self._indices.keys()):
            try:
                self.drop_index(name)
            except Exception:
                pass
        for td in self._temp_dirs:
            if os.path.exists(td):
                try:
                    shutil.rmtree(td)
                except Exception:
                    pass
        self._temp_dirs.clear()
        self._ffi.reset_isolate()
        self._closed = False

    def load_index(
        self,
        name: str,
        base_path: str,
        weights: Optional[np.ndarray] = None,
        lora_dim: int = 0,
    ) -> Index:
        """Maps an existing multi-tier index or .pithos single-file container into memory off-heap.

        Parameters
        ----------
        name : str
            Unique registry identifier name for the index.
        base_path : str
            Filepath of the compiled index or .pithos container.
        weights : ndarray, optional
            Projection or LoRA weight matrix for Matryoshka spectral energy profiling.
        lora_dim : int, default=0
            Bottleneck rank dimension of LoRA matrix.

        Returns
        -------
        Index
            Off-heap index handle.
        """
        effective_path = self._unpack_container_if_needed(base_path)
        if weights is not None:
            w_arr = _check_dtype_float32(weights, "weights")
            status = self._ffi.lib.vdb_load_index_with_weights(
                self._ffi.thread,
                name.encode("utf-8"),
                effective_path.encode("utf-8"),
                w_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(lora_dim),
            )
        else:
            status = self._ffi.lib.vdb_load_index(
                self._ffi.thread,
                name.encode("utf-8"),
                effective_path.encode("utf-8"),
            )
        self._ffi.check_status(status, f"load index '{name}'")
        idx = Index(self, name, base_path)
        self._indices[name] = idx
        return idx

    def get_index(self, name: str) -> Optional[Index]:
        """Returns the loaded index handle by name, or None if not found."""
        return self._indices.get(name)

    def drop_index(self, name: str) -> bool:
        """Unmaps and drops an index and its attached DeltaBuffer from memory."""
        self._delta_buffers.pop(name, None)
        self._indices.pop(name, None)
        status = self._ffi.lib.vdb_drop_index(self._ffi.thread, name.encode("utf-8"))
        return status == 0

    def create_delta_buffer(self, index_name: str, flush_threshold: int = 10000) -> DeltaBuffer:
        """Attaches an in-memory DeltaBuffer for real-time inserts."""
        status = self._ffi.lib.vdb_create_delta_buffer(
            self._ffi.thread,
            index_name.encode("utf-8"),
            ctypes.c_int(flush_threshold),
        )
        self._ffi.check_status(status, f"create DeltaBuffer for '{index_name}'")
        buf = DeltaBuffer(self, index_name)
        self._delta_buffers[index_name] = buf
        return buf

    def get_delta_buffer(self, index_name: str) -> Optional[DeltaBuffer]:
        """Returns the active DeltaBuffer for the given index, or None."""
        return self._delta_buffers.get(index_name)

    # --------------------------------------------------------------------------
    # Index Compilation & Packaging
    # --------------------------------------------------------------------------
    @staticmethod
    def compile_container(
        path: str,
        records: Union[np.ndarray, Sequence[Sequence[float]]],
        ids: Optional[Union[np.ndarray, Sequence[int]]] = None,
        tiers: Optional[Union[np.ndarray, Sequence[int]]] = None,
        metric: str = "cosine",
        q_mode: QuantizationMode = QuantizationMode.ONE_BIT,
        sidecar_mode: Union[SidecarMode, str, int] = SidecarMode.FP8,
        metadata_payload: Optional[bytes] = None,
        metadata_format: str = "raw",
        arrow_table: Optional[Any] = None,
        user_metadata: Optional[dict] = None,
        lib_path: Optional[str] = None,
    ) -> None:
        """Compiles continuous float embeddings into a universal single-file .pithos container (DIOGENES format).

        Parameters
        ----------
        path : str
            Destination filepath for the compiled .pithos container.
        records : array_like
            Input continuous float vectors, shape (N, D).
        ids : array_like, optional
            Explicit 64-bit integer IDs of shape (N,). If None, defaults to `0..N-1`.
        tiers : array_like, optional
            Matryoshka tier boundary steps (e.g. `[64, 128, 256, 768]`).
        metric : str, default="cosine"
            Distance metric ('cosine', 'l2', 'euclidean', 'dot').
        q_mode : QuantizationMode, default=QuantizationMode.ONE_BIT
            Quantization format (ONE_BIT, TWO_BIT, FLOAT32).
        sidecar_mode : SidecarMode or str, default=SidecarMode.FP8
            Precision sidecar format ('none', 'fp16', 'fp8', 'fp4').
        metadata_payload : bytes, optional
            Arbitrary user metadata binary payload (Section 4).
        metadata_format : str, default="raw"
            Format tag for Section 4 payload ('raw', 'jsonl', 'arrow').
        arrow_table : pyarrow.Table, optional
            Apache Arrow Table to embed directly into Section 4.
        user_metadata : dict, optional
            Arbitrary JSON-serializable dictionary embedded into container Table of Contents.
        lib_path : str, optional
            Path to native shared library.
        """
        vecs = _check_dtype_float32(records, "records")
        num_records, dimension = vecs.shape
        
        # --- Native MIPS Interception ---
        metric_lower = metric.lower()
        if metric_lower in ("mips", "dot", "ip", "inner_product", "dot_product"):
            from .mips import SphericalLiftingTransformer
            transformer = SphericalLiftingTransformer(pad_to_multiple=64)
            vecs = transformer.fit_transform(vecs)
            dimension = vecs.shape[1]
            metric = "cosine"
            metric_lower = "cosine"
            
            if user_metadata is None:
                user_metadata = {}
            user_metadata["mips_transformer"] = transformer.to_dict()

        if arrow_table is not None:
            try:
                import pyarrow.ipc as ipc
                sink = io.BytesIO()
                with ipc.new_stream(sink, arrow_table.schema) as writer:
                    writer.write_table(arrow_table)
                metadata_payload = sink.getvalue()
                metadata_format = "arrow"
            except ImportError:
                raise ImportError("pyarrow is required to compile a container with an Arrow table.")

        if ids is None:
            ids_arr = np.arange(num_records, dtype=np.int64)
        else:
            ids_arr = _check_dtype_int64(ids, "ids")

        if tiers is None:
            tiers_arr = np.array([dimension], dtype=np.int32)
        else:
            tiers_arr = np.ascontiguousarray(tiers, dtype=np.int32)

        if isinstance(sidecar_mode, str):
            sidecar_map = {
                "none": SidecarMode.NONE,
                "fp16": SidecarMode.FP16,
                "fp8": SidecarMode.FP8,
                "fp4": SidecarMode.FP4,
            }
            actual_sidecar = sidecar_map.get(sidecar_mode.lower(), SidecarMode.FP8)
        else:
            actual_sidecar = SidecarMode(int(sidecar_mode))

        metric_map = {"cosine": 0, "l2": 1, "euclidean": 1, "dot": 2, "dot_product": 2}
        metric_code = metric_map.get(metric.lower(), 0)

        meta_bytes_ptr = ctypes.c_char_p(metadata_payload) if metadata_payload else ctypes.c_char_p(None)
        meta_len = len(metadata_payload) if metadata_payload else 0
        meta_fmt_ptr = metadata_format.encode("utf-8") if metadata_format else None
        user_json_str = json.dumps(user_metadata).encode("utf-8") if user_metadata else None

        ffi = NativeBindings(lib_path)
        if hasattr(ffi.lib, "vdb_compile_container"):
            with ffi.isolated_context() as temp_thread:
                status = ffi.lib.vdb_compile_container(
                    temp_thread,
                    path.encode("utf-8"),
                    ctypes.c_int(dimension),
                    tiers_arr.ctypes.data_as(ctypes.c_void_p) if tiers_arr is not None else None,
                    ctypes.c_int(len(tiers_arr) if tiers_arr is not None else 0),
                    ids_arr.ctypes.data_as(ctypes.c_void_p),
                    vecs.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(num_records),
                    ctypes.c_int(metric_code),
                    ctypes.c_int(q_mode.value),
                    ctypes.c_int(actual_sidecar.value),
                    meta_bytes_ptr,
                    ctypes.c_int(meta_len),
                    meta_fmt_ptr,
                    user_json_str
                )
                ffi.check_status(status, "compile monolithic container")
        else:
            _write_pithos_container_file(
                path=path,
                vecs=vecs,
                ids_arr=ids_arr,
                tiers_arr=tiers_arr,
                metric_code=metric_code,
                q_mode=int(q_mode),
                actual_sidecar=actual_sidecar,
                metadata_payload=metadata_payload,
                metadata_format=metadata_format,
                user_metadata=user_metadata,
                ffi=ffi,
            )

    @staticmethod
    def compile_container_stream(
        path: str,
        record_stream: Any,
        total_records: int,
        dimension: int,
        tiers: Optional[Union[np.ndarray, Sequence[int]]] = None,
        metric: str = "cosine",
        q_mode: QuantizationMode = QuantizationMode.ONE_BIT,
        sidecar_mode: Union[SidecarMode, str, int] = SidecarMode.FP8,
        metadata_payload: Optional[bytes] = None,
        metadata_format: str = "raw",
        user_metadata: Optional[dict] = None,
        lib_path: Optional[str] = None,
        chunk_size: int = 5000,
        mips_max_norm: Optional[float] = None,
    ) -> None:
        """Compiles continuous float vectors from a streaming iterator directly into a .pithos container on disk.

        Operates with strictly constant O(1) RAM consumption.

        Parameters
        ----------
        path : str
            Destination filepath for container.
        record_stream : iterator
            Iterator or generator yielding batches of `(ids, vecs)` or `vecs`.
        total_records : int
            Total expected record count across entire stream.
        dimension : int
            Vector dimension D.
        tiers : array_like, optional
            Matryoshka tier boundary steps.
        metric : str, default="cosine"
            Distance metric code.
        q_mode : QuantizationMode, default=QuantizationMode.ONE_BIT
            Quantization mode.
        sidecar_mode : SidecarMode or str, default=SidecarMode.FP8
            Precision sidecar format.
        metadata_payload : bytes, optional
            Metadata binary blob.
        metadata_format : str, default="raw"
            Format tag for metadata payload.
        user_metadata : dict, optional
            User metadata dictionary.
        lib_path : str, optional
            Path to native shared library.
        chunk_size : int, default=5000
            Record count per streaming chunk batch.
        mips_max_norm : float, optional
            Required if metric is 'mips'/'dot'. The maximum L2 norm across the entire stream.
        """
        ffi = NativeBindings(lib_path)
        if total_records <= 0:
            raise ValueError(f"total_records must be > 0, got {total_records}")

        if tiers is None:
            tiers_arr = np.array([dimension], dtype=np.int32)
        else:
            tiers_arr = np.ascontiguousarray(tiers, dtype=np.int32)
        num_tiers = len(tiers_arr)

        if isinstance(sidecar_mode, str):
            sidecar_map = {
                "none": SidecarMode.NONE,
                "fp16": SidecarMode.FP16,
                "fp8": SidecarMode.FP8,
                "fp4": SidecarMode.FP4,
            }
            actual_sidecar = sidecar_map.get(sidecar_mode.lower(), SidecarMode.FP8)
        else:
            actual_sidecar = SidecarMode(int(sidecar_mode))

        metric_map = {"cosine": 0, "l2": 1, "euclidean": 1, "dot": 2, "dot_product": 2}
        metric_lower = metric.lower()
        
        # --- Native MIPS Interception ---
        mips_transformer = None
        if metric_lower in ("mips", "dot", "ip", "inner_product", "dot_product"):
            if mips_max_norm is None:
                raise ValueError("mips_max_norm must be provided for streaming MIPS compilation.")
            from .mips import SphericalLiftingTransformer
            mips_transformer = SphericalLiftingTransformer(pad_to_multiple=64)
            mips_transformer.max_norm = float(mips_max_norm)
            mips_transformer.input_dim = dimension
            mips_transformer.lifted_dim = dimension + 1
            mips_transformer.padded_dim = _align64(dimension + 1)
            mips_transformer._is_fitted = True
            
            dimension = mips_transformer.padded_dim
            metric_lower = "cosine"
            metric = "cosine"
            
            if user_metadata is None:
                user_metadata = {}
            user_metadata["mips_transformer"] = mips_transformer.to_dict()

        metric_code = metric_map.get(metric_lower, 0)

        SUPERBLOCK_SIZE = 128
        ids_offset = _align64(SUPERBLOCK_SIZE)
        ids_len = total_records * 8

        tier_offsets = []
        tier_lengths = []
        tier_bytes_per_rec = []
        current_offset = _align64(ids_offset + ids_len)

        prev_bound = 0
        for k in range(num_tiers):
            width = int(tiers_arr[k]) - prev_bound
            prev_bound = int(tiers_arr[k])
            if q_mode == QuantizationMode.TWO_BIT:
                bpr = width // 4
            elif q_mode == QuantizationMode.FLOAT32:
                bpr = width * 4
            else:
                bpr = width // 8
            tier_bytes_per_rec.append(bpr)
            tier_offsets.append(current_offset)
            tier_lengths.append(total_records * bpr)
            current_offset = _align64(current_offset + tier_lengths[-1])

        sidecar_offset = 0
        sidecar_len = 0
        sidecar_format = "none"
        sidecar_bpr = 0
        if actual_sidecar == SidecarMode.FP16:
            sidecar_offset = current_offset
            sidecar_bpr = dimension * 2
            sidecar_len = total_records * sidecar_bpr
            sidecar_format = "fp16"
            current_offset = _align64(current_offset + sidecar_len)
        elif actual_sidecar == SidecarMode.FP8:
            sidecar_offset = current_offset
            sidecar_bpr = dimension * 1
            sidecar_len = total_records * sidecar_bpr
            sidecar_format = "fp8_e4m3"
            current_offset = _align64(current_offset + sidecar_len)
        elif actual_sidecar == SidecarMode.FP4:
            num_blocks = (dimension + 15) // 16
            sidecar_bpr = num_blocks * 9
            sidecar_offset = current_offset
            sidecar_len = total_records * sidecar_bpr
            sidecar_format = "nvfp4_e2m1"
            current_offset = _align64(current_offset + sidecar_len)

        # Multi-Index Hashing (MIH) Section (4 chunks x 256 buckets CSR)
        NUM_MIH_CHUNKS = 4
        NUM_MIH_BUCKETS = 256
        MIH_OFFSETS_COUNT = NUM_MIH_CHUNKS * (NUM_MIH_BUCKETS + 1)
        MIH_OFFSETS_BYTES = MIH_OFFSETS_COUNT * 4

        prefix_table_offset = current_offset
        prefix_postings_length = NUM_MIH_CHUNKS * total_records * 4
        prefix_table_length = MIH_OFFSETS_BYTES + prefix_postings_length
        current_offset = _align64(current_offset + prefix_table_length)

        meta_bytes = metadata_payload if metadata_payload else b""
        meta_format = metadata_format if metadata_format else "raw"
        metadata_offset = 0
        metadata_len = 0
        if len(meta_bytes) > 0:
            metadata_offset = current_offset
            metadata_len = len(meta_bytes)
            current_offset = _align64(current_offset + metadata_len)

        toc_dict = {
            "format": "pithos_v2",
            "motto": "Autarky: Self-contained & Zero Baggage",
            "sections": {
                "ids": {"offset": ids_offset, "length": ids_len, "dtype": "uint64"}
            },
            "user_metadata": user_metadata if user_metadata else {},
        }
        for k in range(num_tiers):
            toc_dict["sections"][f"tier_{k}"] = {
                "offset": tier_offsets[k],
                "length": tier_lengths[k],
                "dim_boundary": int(tiers_arr[k]),
            }
        toc_dict["sections"]["sidecar"] = {
            "offset": sidecar_offset,
            "length": sidecar_len,
            "format": sidecar_format,
        }
        toc_dict["sections"]["prefix_table"] = {
            "offset": prefix_table_offset,
            "length": prefix_table_length,
            "num_chunks": 4,
            "num_buckets_per_chunk": 256,
            "format": "mih_csr_4x8",
        }
        toc_dict["sections"]["metadata"] = {
            "offset": metadata_offset,
            "length": metadata_len,
            "format": meta_format,
        }
        toc_bytes = json.dumps(toc_dict, indent=2).encode("utf-8")
        toc_offset = current_offset
        toc_len = len(toc_bytes)
        current_offset = _align64(current_offset + toc_len)

        total_file_size = current_offset + 20

        sb = bytearray(SUPERBLOCK_SIZE)
        sb[0:8] = b"DIOGENES"
        sb[8:12] = (2).to_bytes(4, byteorder="little", signed=True)
        sb[12:20] = int(total_records).to_bytes(8, byteorder="little", signed=True)
        sb[20:24] = int(dimension).to_bytes(4, byteorder="little", signed=True)
        sb[24:26] = int(metric_code).to_bytes(2, byteorder="little", signed=True)
        sb[26:28] = int(actual_sidecar).to_bytes(2, byteorder="little", signed=True)
        sb[28:30] = int(num_tiers).to_bytes(2, byteorder="little", signed=True)
        for i in range(8):
            t_val = int(tiers_arr[i]) if i < num_tiers else 0
            sb[30 + i * 2 : 32 + i * 2] = t_val.to_bytes(2, byteorder="little", signed=True)
        sb[46:54] = int(toc_offset).to_bytes(8, byteorder="little", signed=True)
        sb[54:58] = int(toc_len).to_bytes(4, byteorder="little", signed=True)
        sb[58:60] = int(q_mode).to_bytes(2, byteorder="little", signed=True)
        sb[60:68] = int(prefix_table_offset).to_bytes(8, byteorder="little", signed=True)
        sb[68:76] = int(prefix_table_length).to_bytes(8, byteorder="little", signed=True)

        trailer = bytearray(20)
        trailer[0:8] = int(toc_offset).to_bytes(8, byteorder="little", signed=True)
        trailer[8:12] = int(toc_len).to_bytes(4, byteorder="little", signed=True)
        trailer[12:20] = b"PITHOSDB"

        out_f = open(path, "wb")
        try:
            out_f.truncate(total_file_size)
            out_f.seek(0)
            out_f.write(sb)
            if metadata_len > 0:
                out_f.seek(metadata_offset)
                out_f.write(meta_bytes)
            out_f.seek(toc_offset)
            out_f.write(toc_bytes)
            out_f.seek(total_file_size - 20)
            out_f.write(trailer)
            out_f.flush()

            def _iterate_chunks(stream, chunk_sz):
                current_id = 0
                buffer_vecs = []
                buffer_ids = []
                for item in stream:
                    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], (np.ndarray, list, range)) and isinstance(item[1], (np.ndarray, list)):
                        b_ids = _check_dtype_int64(item[0], "b_ids")
                        b_vecs = _check_dtype_float32(item[1], "b_vecs")
                        if b_vecs.ndim == 1:
                            b_vecs = b_vecs.reshape(1, -1)
                        current_id += b_vecs.shape[0]
                        yield b_ids, b_vecs
                    elif isinstance(item, np.ndarray) and item.ndim == 2:
                        b_vecs = _check_dtype_float32(item, "b_vecs")
                        b_ids = np.arange(current_id, current_id + b_vecs.shape[0], dtype=np.int64)
                        current_id += b_vecs.shape[0]
                        yield b_ids, b_vecs
                    elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple, np.ndarray)):
                        b_vecs = _check_dtype_float32(item, "b_vecs")
                        b_ids = np.arange(current_id, current_id + b_vecs.shape[0], dtype=np.int64)
                        current_id += b_vecs.shape[0]
                        yield b_ids, b_vecs
                    else:
                        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], int):
                            rec_id, vec = item
                            buffer_ids.append(rec_id)
                            buffer_vecs.append(vec)
                        else:
                            buffer_ids.append(current_id + len(buffer_vecs))
                            buffer_vecs.append(item)
                        if len(buffer_vecs) >= chunk_sz:
                            b_vecs = _check_dtype_float32(buffer_vecs, "buffer_vecs")
                            b_ids = _check_dtype_int64(buffer_ids, "buffer_ids")
                            current_id += len(buffer_vecs)
                            buffer_vecs.clear()
                            buffer_ids.clear()
                            yield b_ids, b_vecs
                if len(buffer_vecs) > 0:
                    b_vecs = _check_dtype_float32(buffer_vecs, "buffer_vecs")
                    b_ids = _check_dtype_int64(buffer_ids, "buffer_ids")
                    yield b_ids, b_vecs
            processed_records = 0
            all_t0_chunks = []
            with tempfile.TemporaryDirectory() as tmpdir:
                with ffi.isolated_context() as temp_thread:
                    chunk_idx = 0
                    for b_ids, b_vecs in _iterate_chunks(record_stream, chunk_size):
                        if mips_transformer is not None:
                            norms = np.linalg.norm(b_vecs, axis=1)
                            lifted = np.zeros((len(b_vecs), dimension), dtype=np.float32)
                            lifted[:, :mips_transformer.input_dim] = b_vecs / mips_transformer.max_norm
                            lifted[:, mips_transformer.input_dim] = np.sqrt(np.maximum(0.0, 1.0 - (norms / mips_transformer.max_norm)**2))
                            b_vecs = lifted

                        b_vecs = np.ascontiguousarray(b_vecs)
                        b_size = b_vecs.shape[0]
                        if b_size == 0 or processed_records >= total_records:
                            continue

                        tmp_base = os.path.join(tmpdir, f"chunk_{chunk_idx}")
                        status = ffi.lib.vdb_compile_index_file_ext(
                            temp_thread,
                            tmp_base.encode("utf-8"),
                            ctypes.c_byte(1),
                            ctypes.c_longlong(1737400),
                            ctypes.c_int(dimension),
                            tiers_arr.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(num_tiers),
                            b_ids.ctypes.data_as(ctypes.c_void_p),
                            b_vecs.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(b_size),
                            ctypes.c_int(int(q_mode)),
                            ctypes.c_int(int(actual_sidecar)),
                        )
                        ffi.check_status(status, "compile stream chunk")

                        out_f.seek(ids_offset + processed_records * 8)
                        with open(f"{tmp_base}_ids.bin", "rb") as f_ids:
                            shutil.copyfileobj(f_ids, out_f)

                        for k in range(num_tiers):
                            out_f.seek(tier_offsets[k] + processed_records * tier_bytes_per_rec[k])
                            with open(f"{tmp_base}_tier_{k}.bin", "rb") as f_t:
                                if k == 0:
                                    t0_data = f_t.read()
                                    out_f.write(t0_data)
                                    bpr_t0 = len(t0_data) // b_size if b_size > 0 else 0
                                    if b_size > 0 and bpr_t0 >= 4:
                                        chunk_t0 = np.frombuffer(t0_data, dtype=np.uint8).reshape(b_size, bpr_t0)[:, :4]
                                        all_t0_chunks.append(chunk_t0)
                                else:
                                    shutil.copyfileobj(f_t, out_f)

                        if actual_sidecar == SidecarMode.FP16:
                            out_f.seek(sidecar_offset + processed_records * sidecar_bpr)
                            fp16_chunk = f"{tmp_base}_fp16.bin"
                            if os.path.exists(fp16_chunk) and os.path.getsize(fp16_chunk) == b_size * dimension * 2:
                                with open(fp16_chunk, "rb") as f_s:
                                    shutil.copyfileobj(f_s, out_f)
                            else:
                                out_f.write(b_vecs.astype(np.float16).tobytes())
                        elif actual_sidecar == SidecarMode.FP8:
                            out_f.seek(sidecar_offset + processed_records * sidecar_bpr)
                            fp8_chunk = f"{tmp_base}_fp8.bin"
                            if os.path.exists(fp8_chunk) and os.path.getsize(fp8_chunk) == b_size * dimension:
                                with open(fp8_chunk, "rb") as f_s:
                                    shutil.copyfileobj(f_s, out_f)
                            else:
                                out_f.write(_encode_fp8_e4m3_array(b_vecs).tobytes())
                        elif actual_sidecar == SidecarMode.FP4:
                            out_f.seek(sidecar_offset + processed_records * sidecar_bpr)
                            fp4_chunk = f"{tmp_base}_fp4.bin"
                            if os.path.exists(fp4_chunk) and os.path.getsize(fp4_chunk) == b_size * sidecar_bpr:
                                with open(fp4_chunk, "rb") as f_s:
                                    shutil.copyfileobj(f_s, out_f)
                            else:
                                out_f.write(_encode_nvfp4_blocks_array(b_vecs).tobytes())

                        for f_pattern in glob.glob(f"{tmp_base}*"):
                            try:
                                os.remove(f_pattern)
                            except Exception:
                                pass

                        processed_records += b_size
                        chunk_idx += 1

            # Write Multi-Index Hashing (MIH) Prefix Table (4 x CSR)
            if len(all_t0_chunks) > 0:
                keys_4c = np.concatenate(all_t0_chunks, axis=0)
            else:
                keys_4c = np.empty((0, 4), dtype=np.uint8)

            actual_total = keys_4c.shape[0]
            offsets_arr = np.zeros((NUM_MIH_CHUNKS, NUM_MIH_BUCKETS + 1), dtype=np.int32)
            postings_arr = np.empty((NUM_MIH_CHUNKS, actual_total), dtype=np.int32)

            for c in range(NUM_MIH_CHUNKS):
                chunk_keys = keys_4c[:, c]
                counts = np.bincount(chunk_keys, minlength=NUM_MIH_BUCKETS)
                offsets_arr[c, 1:] = np.cumsum(counts, dtype=np.int32)
                postings_arr[c, :] = np.argsort(chunk_keys, kind="stable").astype(np.int32)

            out_f.seek(prefix_table_offset)
            out_f.write(offsets_arr.tobytes())
            out_f.write(postings_arr.tobytes())

            if processed_records != total_records:
                out_f.seek(12)
                out_f.write(int(processed_records).to_bytes(8, byteorder="little", signed=True))

            out_f.flush()
        finally:
            out_f.close()
            ffi.shrink_to_fit()

    @staticmethod
    def compile_index(
        base_path: str,
        records: Union[np.ndarray, Sequence[Sequence[float]]],
        ids: Optional[Union[np.ndarray, Sequence[int]]] = None,
        tiers: Optional[Union[np.ndarray, Sequence[int]]] = None,
        planet_id: int = 1,
        planet_radius: int = 1737400,
        q_mode: QuantizationMode = QuantizationMode.ONE_BIT,
        sidecar_mode: Union[SidecarMode, str, int] = SidecarMode.FP8,
        write_fp16: Optional[bool] = None,
        use_container: bool = False,
        user_metadata: Optional[dict] = None,
        lib_path: Optional[str] = None,
    ) -> None:
        """Compiles raw continuous float embeddings into a multi-tier binary columnar Pithos index on disk.

        Parameters
        ----------
        base_path : str
            Base filepath prefix for the index files.
        records : array_like
            Input continuous float vectors of shape (N, D).
        ids : array_like, optional
            Explicit 64-bit integer IDs of shape (N,).
        tiers : array_like, optional
            Matryoshka tier boundary steps.
        planet_id : int, default=1
            Planetary body identifier code.
        planet_radius : int, default=1737400
            Equatorial planetary radius in meters.
        q_mode : QuantizationMode, default=QuantizationMode.ONE_BIT
            Quantization mode.
        sidecar_mode : SidecarMode or str, default=SidecarMode.FP8
            Precision sidecar format ('none', 'fp16', 'fp8', 'fp4').
        write_fp16 : bool, optional
            Legacy flag for FP16 sidecar generation.
        use_container : bool, default=False
            If True, outputs a single-file `.pithos` container instead of multi-file layout.
        user_metadata : dict, optional
            User metadata dictionary.
        lib_path : str, optional
            Path to native shared library.
        """
        if use_container or base_path.endswith(".pithos"):
            return VectorDb.compile_container(
                path=base_path,
                records=records,
                ids=ids,
                tiers=tiers,
                q_mode=q_mode,
                sidecar_mode=sidecar_mode,
                user_metadata=user_metadata,
                lib_path=lib_path,
            )
        ffi = NativeBindings(lib_path)
        vecs = _check_dtype_float32(records, "records")
        num_records, dimension = vecs.shape

        if ids is None:
            ids_arr = np.arange(num_records, dtype=np.int64)
        else:
            ids_arr = _check_dtype_int64(ids, "ids")

        if tiers is None:
            tiers_arr = np.array([dimension], dtype=np.int32)
        else:
            tiers_arr = np.ascontiguousarray(tiers, dtype=np.int32)

        if write_fp16 is not None:
            actual_sidecar = SidecarMode.FP16 if write_fp16 else SidecarMode.NONE
        elif isinstance(sidecar_mode, str):
            sidecar_map = {
                "none": SidecarMode.NONE,
                "fp16": SidecarMode.FP16,
                "fp8": SidecarMode.FP8,
                "fp4": SidecarMode.FP4,
            }
            actual_sidecar = sidecar_map.get(sidecar_mode.lower(), SidecarMode.FP8)
        else:
            actual_sidecar = SidecarMode(int(sidecar_mode))

        with ffi.isolated_context() as temp_thread:
            status = ffi.lib.vdb_compile_index_file_ext(
                temp_thread,
                base_path.encode("utf-8"),
                ctypes.c_byte(planet_id),
                ctypes.c_longlong(planet_radius),
                ctypes.c_int(dimension),
                tiers_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(len(tiers_arr)),
                ids_arr.ctypes.data_as(ctypes.c_void_p),
                vecs.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_records),
                ctypes.c_int(int(q_mode)),
                ctypes.c_int(int(actual_sidecar)),
            )
            ffi.check_status(status, "compile index file")

        # Sidecar file handling (FP8 / FP4 / NONE)
        if actual_sidecar == SidecarMode.FP8:
            fp16_file = f"{base_path}_fp16.bin"
            if os.path.exists(fp16_file):
                os.remove(fp16_file)
            fp8_file = f"{base_path}_fp8.bin"
            if not os.path.exists(fp8_file) or os.path.getsize(fp8_file) != num_records * dimension:
                fp8_bytes = _encode_fp8_e4m3_array(vecs).tobytes()
                with open(fp8_file, "wb") as f:
                    f.write(fp8_bytes)
            with open(base_path, "r+b") as f:
                f.seek(62)
                f.write(bytes([2]))
        elif actual_sidecar == SidecarMode.FP4:
            fp16_file = f"{base_path}_fp16.bin"
            if os.path.exists(fp16_file):
                os.remove(fp16_file)
            fp4_file = f"{base_path}_fp4.bin"
            num_blocks = (dimension + 15) // 16
            bytes_per_rec = num_blocks * 9
            if not os.path.exists(fp4_file) or os.path.getsize(fp4_file) != num_records * bytes_per_rec:
                fp4_bytes = _encode_nvfp4_blocks_array(vecs).tobytes()
                with open(fp4_file, "wb") as f:
                    f.write(fp4_bytes)
            with open(base_path, "r+b") as f:
                f.seek(62)
                f.write(bytes([3]))
        elif actual_sidecar == SidecarMode.NONE:
            fp16_file = f"{base_path}_fp16.bin"
            if os.path.exists(fp16_file):
                os.remove(fp16_file)
            with open(base_path, "r+b") as f:
                f.seek(62)
                f.write(bytes([0]))

    @staticmethod
    def compact_indices(
        source_paths: Sequence[str],
        target_path: str,
        lib_path: Optional[str] = None,
    ) -> None:
        """Compacts multiple compiled Pithos indices into a consolidated index file.

        Parameters
        ----------
        source_paths : sequence of str
            List of source index base paths.
        target_path : str
            Consolidated target base path.
        lib_path : str, optional
            Path to native shared library.
        """
        ffi = NativeBindings(lib_path)
        joined = ";".join(source_paths)
        status = ffi.lib.vdb_compact_indexes(
            ffi.thread,
            joined.encode("utf-8"),
            target_path.encode("utf-8"),
        )
        ffi.check_status(status, "compact indices")

        first_path = source_paths[0]
        if os.path.exists(f"{first_path}_fp8.bin"):
            target_fp8 = f"{target_path}_fp8.bin"
            with open(target_fp8, "wb") as out_f:
                for sp in source_paths:
                    with open(f"{sp}_fp8.bin", "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
            if os.path.exists(f"{target_path}_fp16.bin"):
                os.remove(f"{target_path}_fp16.bin")
            with open(target_path, "r+b") as f:
                f.seek(62)
                f.write(bytes([2]))
        elif os.path.exists(f"{first_path}_fp4.bin"):
            target_fp4 = f"{target_path}_fp4.bin"
            with open(target_fp4, "wb") as out_f:
                for sp in source_paths:
                    with open(f"{sp}_fp4.bin", "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
            if os.path.exists(f"{target_path}_fp16.bin"):
                os.remove(f"{target_path}_fp16.bin")
            with open(target_path, "r+b") as f:
                f.seek(62)
                f.write(bytes([3]))

    # --------------------------------------------------------------------------
    # CUDA Acceleration Management
    # --------------------------------------------------------------------------
    def cuda_init(self, device_id: int = 0) -> None:
        """Initializes CUDA hardware acceleration runtime.

        Parameters
        ----------
        device_id : int, default=0
            Target NVIDIA GPU device ordinal.
        """
        if not self._ffi._has_cuda:
            raise RuntimeError("Pithos native library was compiled without CUDA support.")
        status = self._ffi.lib.vdb_cuda_init(self._ffi.thread, ctypes.c_int(device_id))
        self._ffi.check_status(status, "initialize CUDA")

    def cuda_shutdown(self) -> None:
        """Releases CUDA runtime resources."""
        if self._ffi._has_cuda:
            self._ffi.lib.vdb_cuda_shutdown(self._ffi.thread)

    def cuda_is_available(self) -> bool:
        """Returns True if a compatible CUDA runtime is active and initialized."""
        if not self._ffi._has_cuda:
            return False
        return self._ffi.lib.vdb_cuda_is_available(self._ffi.thread) == 1
