from __future__ import annotations

import ctypes
import os
import io
import json
import math
import shutil
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Union, Dict, Any, Sequence, Tuple
import numpy as np

from .ffi import NativeBindings, PithosNativeError, reset_isolate, shrink_to_fit

_FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def _encode_fp8_e4m3_scalar(val: float) -> int:
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
        if m > 7: m = 7
        return (sign << 7) | (m & 0x7)
    exp = int(math.floor(math.log2(abs_val))) + 7
    if exp < 1:
        m = int(round(abs_val * 512.0))
        if m > 7: m = 7
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

def _decode_fp8_e4m3_scalar(b: int) -> float:
    b_int = int(b)
    sign = 1 if (b_int & 0x80) != 0 else 0
    exp = (b_int >> 3) & 0x0F
    mantissa = b_int & 0x07
    sign_mult = -1.0 if sign == 1 else 1.0
    if exp == 0:
        return sign_mult * (mantissa / 512.0)
    elif exp == 15 and mantissa == 7:
        return float('nan')
    else:
        scale = float(1 << (exp - 7)) if exp >= 7 else (1.0 / float(1 << (7 - exp)))
        return sign_mult * scale * (1.0 + mantissa / 8.0)

_FP8_DECODE_LUT = np.array([_decode_fp8_e4m3_scalar(b) for b in range(256)], dtype=np.float32)

def _decode_fp8_e4m3_array(arr: np.ndarray) -> np.ndarray:
    """Decodes uint8 FP8 E4M3 values back to float32 using a precomputed 256-element LUT."""
    return _FP8_DECODE_LUT[np.ascontiguousarray(arr, dtype=np.uint8)]

def _encode_fp8_e4m3_array(arr: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of float32 array to 8-bit OCP/NVIDIA FP8 E4M3 standard bytes.
    Matches Java VectorDb.encodeFP8_E4M3 with 100% bit-exact parity.
    """
    flat = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = flat.view(np.uint32)

    sign = ((u32 >> 24) & 0x80).astype(np.uint8)
    exp = ((u32 >> 23) & 0xFF).astype(np.int32)
    mant = (u32 & 0x7FFFFF).astype(np.int32)
    abs_val = np.abs(flat)

    out = np.zeros(flat.shape, dtype=np.uint8)

    # 1. Underflow / Zero
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

_FP4_THRESHOLDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)

def _encode_fp4_nibble(val: float) -> int:
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

def _encode_fp4_nibbles_array(norm_floats: np.ndarray) -> np.ndarray:
    """Vectorized quantization of normalized floats into 4-bit FP4 E2M1 nibbles (0..15)."""
    flat = np.ascontiguousarray(norm_floats, dtype=np.float32)
    signs = np.where(flat < 0.0, 0x08, 0x00).astype(np.uint8)
    abs_floats = np.abs(flat)
    nibbles = np.digitize(abs_floats, _FP4_THRESHOLDS).astype(np.uint8)
    return signs | (nibbles & 0x07)

def _encode_nvfp4_blocks_array(vecs: np.ndarray) -> np.ndarray:
    """
    Vectorized NVFP4 Block-16 microscaling encoder.
    Converts 2D float32 array (N, D) to (N, num_blocks * 9) uint8 bytes.
    Each 16-element block is stored as 1 byte FP8 E4M3 scale factor + 8 bytes packed nibble pairs.
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

class QuantizationMode(IntEnum):
    """Supported vector quantization modes in Pithos."""
    ONE_BIT = 0       # 1-bit sign quantization (1 bit per dimension)
    TWO_BIT = 1       # 2-bit ternary / QJL residual quantization (2 bits per dimension)
    FLOAT32 = 2       # Unquantized 32-bit float bypass

class SidecarMode(IntEnum):
    """Supported float sidecar storage formats in Pithos."""
    NONE = 0          # No float sidecar (asymmetric rotated L2 fallback)
    FP16 = 1          # IEEE 754 half-precision float sidecar (_fp16.bin, 2 B/dim)
    FP8  = 2          # OCP/NVIDIA Blackwell FP8 E4M3 sidecar (_fp8.bin, 1 B/dim)
    FP4  = 3          # Blackwell NVFP4 E2M1 block microscaling (_fp4.bin, 0.5 B/dim + scale)

@dataclass(frozen=True)
class SearchResult:
    """A single nearest neighbor search result."""
    id: int
    score: int
    
    @property
    def distance(self) -> float:
        """Returns the scaled float distance."""
        return self.score / 1_000_000.0

@dataclass(frozen=True)
class IndexInfo:
    """Metadata attributes of a loaded Pithos index."""
    dimension: int
    size: int
    planet_id: int
    planet_radius: int
    tiers_count: int
    sidecar_mode: SidecarMode = SidecarMode.NONE

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "size": self.size,
            "planet_id": self.planet_id,
            "planet_radius": self.planet_radius,
            "tiers_count": self.tiers_count,
            "sidecar_mode": int(self.sidecar_mode),
        }


class DeltaBuffer:
    """
    Log-Structured Merge (LSM) in-memory write buffer for real-time inserts.
    """
    def __init__(self, db: VectorDb, index_name: str):
        self._db = db
        self._name = index_name
        self._ffi = db._ffi

    @property
    def name(self) -> str:
        return self._name

    def insert(self, record_id: int, vector: Union[np.ndarray, Sequence[float]]) -> None:
        """Inserts a new vector into the delta buffer."""
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        status = self._ffi.lib.vdb_insert(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_longlong(record_id),
            vec.ctypes.data_as(ctypes.c_void_p)
        )
        self._ffi.check_status(status, "insert into DeltaBuffer")

    def delete(self, record_id: int) -> bool:
        """Soft-deletes a record from the delta buffer (tombstone)."""
        ret = self._ffi.lib.vdb_delete_from_delta(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_longlong(record_id)
        )
        if ret < 0:
            raise PithosNativeError(ret, f"Failed to delete record {record_id} from delta buffer.")
        return ret == 1

    def size(self) -> int:
        """Returns the number of live (non-tombstoned) records in the delta buffer."""
        size = self._ffi.lib.vdb_delta_size(self._ffi.thread, self._name.encode("utf-8"))
        if size < 0:
            raise PithosNativeError(int(size), "Failed to retrieve delta buffer size.")
        return int(size)

    def needs_flush(self) -> bool:
        """Returns True if the live count has exceeded the configured flush threshold."""
        ret = self._ffi.lib.vdb_needs_flush(self._ffi.thread, self._name.encode("utf-8"))
        if ret < 0:
            raise PithosNativeError(ret, "Failed to check flush state.")
        return ret == 1

    def backup(self, path: str) -> None:
        """Serializes the live delta entries to a binary backup file."""
        status = self._ffi.lib.vdb_backup_delta(
            self._ffi.thread,
            self._name.encode("utf-8"),
            path.encode("utf-8")
        )
        self._ffi.check_status(status, "backup DeltaBuffer")

    def restore(self, path: str, flush_threshold: int = 10000) -> None:
        """Restores delta entries from a binary backup file."""
        status = self._ffi.lib.vdb_restore_delta(
            self._ffi.thread,
            self._name.encode("utf-8"),
            path.encode("utf-8"),
            ctypes.c_int(flush_threshold)
        )
        self._ffi.check_status(status, "restore DeltaBuffer")

@dataclass(frozen=True)
class FpgaDescriptor:
    """
    Hardware descriptor for direct FPGA DMA streaming and MMIO register configuration.
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

class Index:
    """
    Handle to an off-heap memory-mapped multi-tier vector index.
    """
    def __init__(self, db: VectorDb, name: str, base_path: str):

        self._db = db
        self._name = name
        self._base_path = base_path
        self._ffi = db._ffi
        self._info: Optional[IndexInfo] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_path(self) -> str:
        return self._base_path

    def info(self) -> IndexInfo:
        """Retrieves index metadata (dimension, size, tiers, planetId)."""
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
            ctypes.byref(tcount)
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
            sidecar_mode=sidecar_mode
        )
        return self._info

    def __len__(self) -> int:
        return int(self._ffi.lib.vdb_size(self._ffi.thread, self._name.encode("utf-8")))

    def size(self) -> int:
        """Returns the total number of records in the index."""
        return len(self)

    @property
    def dimension(self) -> int:
        return self.info().dimension

    @property
    def planet_id(self) -> int:
        return self.info().planet_id

    @property
    def planet_radius(self) -> int:
        return self.info().planet_radius

    @property
    def tier_count(self) -> int:
        return self.info().tiers_count

    def set_chunk_size(self, chunk_size: int) -> None:
        """Configures the parallel record chunk size for Disruptor worker threads."""
        status = self._ffi.lib.vdb_set_chunk_size(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_longlong(chunk_size)
        )
        self._ffi.check_status(status, "set chunk size")

    def set_energy_budget(self, tau: float) -> None:
        """Sets the Matryoshka early-exit cumulative spectral energy budget tau in (0, 1]."""
        status = self._ffi.lib.vdb_set_energy_budget(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_double(tau)
        )
        self._ffi.check_status(status, "set energy budget")

    def search(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False,
        return_numpy: bool = False
    ) -> Union[List[SearchResult], List[List[SearchResult]], Tuple[np.ndarray, np.ndarray]]:
        """
        Performs high-performance batch k-NN search across multi-tier vectors.
        If return_numpy=True, returns (out_ids, out_distances) directly as flat numpy arrays (zero-copy).
        """
        q_arr = np.ascontiguousarray(queries, dtype=np.float32)
        is_single = q_arr.ndim == 1
        if is_single:
            q_arr = q_arr.reshape(1, -1)
            
        num_queries, dim = q_arr.shape
        out_ids = np.empty((num_queries, k), dtype=np.int64)
        out_dists = np.empty((num_queries, k), dtype=np.int32)

        if cuda and self._ffi._has_cuda:
            status = self._ffi.lib.vdb_cuda_batch_search(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                ctypes.c_int(k),
                out_ids.ctypes.data_as(ctypes.c_void_p),
                out_dists.ctypes.data_as(ctypes.c_void_p)
            )
        else:
            status = self._ffi.lib.vdb_batch_search(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                ctypes.c_int(k),
                out_ids.ctypes.data_as(ctypes.c_void_p),
                out_dists.ctypes.data_as(ctypes.c_void_p)
            )
        self._ffi.check_status(status, "search")

        if return_numpy:
            return (out_ids[0], out_dists[0]) if is_single else (out_ids, out_dists)

        results: List[List[SearchResult]] = []
        for q_idx in range(num_queries):
            q_res: List[SearchResult] = []
            for i in range(k):
                rec_id = int(out_ids[q_idx, i])
                if rec_id == -1:
                    continue
                q_res.append(SearchResult(id=rec_id, score=int(out_dists[q_idx, i])))
            results.append(q_res)

        return results[0] if is_single else results

    def search_numpy(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-Copy batch k-NN search returning flat numpy arrays (out_ids, out_distances).
        Completely bypasses Python object allocation and GC overhead.
        """
        return self.search(queries, k=k, cuda=cuda, return_numpy=True)

    def batch_search(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False,
        return_numpy: bool = False
    ) -> Union[List[SearchResult], List[List[SearchResult]], Tuple[np.ndarray, np.ndarray]]:
        """
        Alias for search() performing batch k-NN search across multi-tier vectors.
        """
        return self.search(queries, k=k, cuda=cuda, return_numpy=return_numpy)

    def batch_search_numpy(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-Copy batch k-NN search returning flat numpy arrays (out_ids, out_distances).
        """
        return self.search(queries, k=k, cuda=cuda, return_numpy=True)

    def search_merged(
        self,
        query: Union[np.ndarray, Sequence[float]],
        k: int = 10
    ) -> List[SearchResult]:
        """
        Queries both the base memory-mapped index and the active DeltaBuffer,
        merging and deduplicating results.
        """
        q_arr = np.ascontiguousarray(query, dtype=np.float32).flatten()
        out_ids = np.empty(k, dtype=np.int64)
        out_dists = np.empty(k, dtype=np.int32)

        status = self._ffi.lib.vdb_search_merged(
            self._ffi.thread,
            self._name.encode("utf-8"),
            q_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(k),
            out_ids.ctypes.data_as(ctypes.c_void_p),
            out_dists.ctypes.data_as(ctypes.c_void_p)
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
        cuda: bool = False
    ) -> tuple[int, np.ndarray]:
        """
        Performs multi-family resonant voting across scientific criteria.
        """
        q_arr = np.ascontiguousarray(queries, dtype=np.float32)
        f_arr = np.ascontiguousarray(families, dtype=np.int32)
        t_arr = np.ascontiguousarray(thresholds, dtype=np.int32)
        
        num_queries = q_arr.shape[0]
        total_records = len(self)
        
        mask = voting_mask if voting_mask is not None else out_voting_mask
        if mask is None:
            mask = np.zeros(total_records, dtype=np.uint8)
        else:
            mask = np.ascontiguousarray(mask, dtype=np.uint8)

        if cuda and self._ffi._has_cuda:
            resonant_count = self._ffi.lib.vdb_cuda_query_planetary_grid(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                f_arr.ctypes.data_as(ctypes.c_void_p),
                t_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                mask.ctypes.data_as(ctypes.c_void_p)
            )
        else:
            resonant_count = self._ffi.lib.vdb_query_planetary_grid(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                f_arr.ctypes.data_as(ctypes.c_void_p),
                t_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                mask.ctypes.data_as(ctypes.c_void_p)
            )
            
        if resonant_count < 0:
            self._ffi.check_status(resonant_count, "query planetary grid")
        return int(resonant_count), mask

    def get_tier_memory_address(self, tier_idx: int) -> tuple[int, int]:
        """Returns the raw virtual address and byte length of a tier segment for direct DMA/FPGA execution."""

        addr = ctypes.c_longlong()
        length = ctypes.c_longlong()
        status = self._ffi.lib.vdb_get_tier_address(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.c_int(tier_idx),
            ctypes.byref(addr),
            ctypes.byref(length)
        )
        self._ffi.check_status(status, "get tier memory address")
        return addr.value, length.value

    get_tier_address = get_tier_memory_address

    def get_metadata_address(self) -> tuple[int, int]:
        """Returns the raw off-heap address and byte length of the metadata sidecar segment."""
        addr = ctypes.c_longlong()
        length = ctypes.c_longlong()
        status = self._ffi.lib.vdb_get_metadata_address(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.byref(addr),
            ctypes.byref(length)
        )
        self._ffi.check_status(status, "get metadata address")
        return addr.value, length.value

    def get_ids_address(self) -> tuple[int, int]:
        """Returns the raw off-heap address and byte length of the record IDs segment."""
        addr = ctypes.c_longlong()
        length = ctypes.c_longlong()
        status = self._ffi.lib.vdb_get_ids_address(
            self._ffi.thread,
            self._name.encode("utf-8"),
            ctypes.byref(addr),
            ctypes.byref(length)
        )
        self._ffi.check_status(status, "get IDs address")
        return addr.value, length.value

    def get_tier_buffer(self, tier_idx: int = 0) -> np.ndarray:
        """
        Returns a zero-copy NumPy ndarray viewing the raw off-heap memory-mapped tier bit vectors.
        """
        addr, length = self.get_tier_address(tier_idx)
        c_arr = (ctypes.c_uint8 * length).from_address(addr)
        return np.ctypeslib.as_array(c_arr)

    def get_metadata_buffer(self) -> np.ndarray:
        """
        Returns a zero-copy uint64 NumPy ndarray viewing the raw off-heap metadata bitmask flags.
        """
        addr, length = self.get_metadata_address()
        c_arr = (ctypes.c_uint64 * (length // 8)).from_address(addr)
        return np.ctypeslib.as_array(c_arr)

    def get_ids_buffer(self) -> np.ndarray:
        """
        Returns a zero-copy int64 NumPy ndarray viewing the raw off-heap record IDs.
        """
        addr, length = self.get_ids_address()
        c_arr = (ctypes.c_int64 * (length // 8)).from_address(addr)
        return np.ctypeslib.as_array(c_arr)

    def get_fpga_descriptor(self, tier_idx: int = 0) -> FpgaDescriptor:
        """
        Generates a complete hardware descriptor for FPGA DMA engines and PCIe MMIO registers.
        """
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
            words_per_record = (tier_dim + 63) // 64
        
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
            words_per_record=words_per_record
        )

    @property
    def user_metadata(self) -> dict:
        """Returns the user metadata dictionary embedded in the .pithos single-file container."""
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

    @property
    def arrow_table(self):
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
        """Returns partition metadata dictionary from the container's Table of Contents or Arrow table."""
        user_meta = self.user_metadata
        if "partitions" in user_meta:
            return user_meta["partitions"]
        tbl = self.arrow_table
        if tbl is not None:
            return tbl.to_pydict()
        return {}

    @property
    def is_cuda_capable(self) -> bool:
        """Returns True if the loaded native library includes CUDA hardware acceleration symbols."""
        return self._ffi._has_cuda


    def transform_and_quantize(self, vector: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """
        Transforms a continuous float vector through Rademacher sign preconditioning
        and block-diagonal Fast Walsh-Hadamard rotation, returning packed 64-bit uint64 words.
        """
        vec = np.ascontiguousarray(vector, dtype=np.float32).flatten()
        dim = self.dimension
        if vec.shape[0] != dim:
            raise ValueError(f"Vector dimension {vec.shape[0]} does not match index dimension {dim}")
        
        words_count = (dim + 63) // 64
        out_packed = np.zeros(words_count, dtype=np.uint64)
        
        status = self._ffi.lib.vdb_transform_and_quantize(
            self._ffi.thread,
            self._name.encode("utf-8"),
            vec.ctypes.data_as(ctypes.c_void_p),
            out_packed.ctypes.data_as(ctypes.c_void_p)
        )
        self._ffi.check_status(status, "transform and quantize vector")
        return out_packed




def _align64(offset: int) -> int:
    return (offset + 63) & ~63

def _pad_to(file_obj, target_offset: int) -> None:
    cur = file_obj.tell()
    if cur < target_offset:
        file_obj.write(bytes(target_offset - cur))

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
    ffi: NativeBindings
) -> None:
    import tempfile
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
                ctypes.c_int(int(actual_sidecar))
            )
            ffi.check_status(status, "compile temporary index files")

        if actual_sidecar == SidecarMode.FP8:
            fp8_file = f"{tmp_base}_fp8.bin"
            if not os.path.exists(fp8_file) or os.path.getsize(fp8_file) != num_records * dimension:
                fp8_bytes = _encode_fp8_e4m3_array(vecs).tobytes()
                with open(fp8_file, "wb") as f:
                    f.write(fp8_bytes)
        elif actual_sidecar == SidecarMode.FP4:
            fp4_file = f"{tmp_base}_fp4.bin"
            num_blocks = (dimension + 15) // 16
            bytes_per_rec = num_blocks * 9
            if not os.path.exists(fp4_file) or os.path.getsize(fp4_file) != num_records * bytes_per_rec:
                fp4_bytes = _encode_nvfp4_blocks_array(vecs).tobytes()
                with open(fp4_file, "wb") as f:
                    f.write(fp4_bytes)

        with open(f"{tmp_base}_ids.bin", "rb") as f:
            ids_bytes = f.read()

        tier_bytes_list = []
        for k in range(num_tiers):
            with open(f"{tmp_base}_tier_{k}.bin", "rb") as f:
                tier_bytes_list.append(f.read())

        sidecar_bytes = b""
        sidecar_format = "none"
        if actual_sidecar == SidecarMode.FP16 and os.path.exists(f"{tmp_base}_fp16.bin"):
            with open(f"{tmp_base}_fp16.bin", "rb") as f:
                sidecar_bytes = f.read()
            sidecar_format = "fp16"
        elif actual_sidecar == SidecarMode.FP8 and os.path.exists(f"{tmp_base}_fp8.bin"):
            with open(f"{tmp_base}_fp8.bin", "rb") as f:
                sidecar_bytes = f.read()
            sidecar_format = "fp8_e4m3"
        elif actual_sidecar == SidecarMode.FP4 and os.path.exists(f"{tmp_base}_fp4.bin"):
            with open(f"{tmp_base}_fp4.bin", "rb") as f:
                sidecar_bytes = f.read()
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
        NUM_MIH_CHUNKS = 4
        NUM_MIH_BUCKETS = 256
        MIH_OFFSETS_COUNT = NUM_MIH_CHUNKS * (NUM_MIH_BUCKETS + 1)
        MIH_OFFSETS_BYTES = MIH_OFFSETS_COUNT * 4

        chunk_bucket_keys = [[0] * num_records for _ in range(NUM_MIH_CHUNKS)]
        chunk_bucket_counts = [[0] * NUM_MIH_BUCKETS for _ in range(NUM_MIH_CHUNKS)]

        bytes_per_rec_t0 = len(tier_bytes_list[0]) // num_records if num_records > 0 else 0
        tier0_raw = tier_bytes_list[0]
        for i in range(num_records):
            rec_off = i * bytes_per_rec_t0
            k0 = tier0_raw[rec_off] if bytes_per_rec_t0 > 0 else 0
            k1 = tier0_raw[rec_off + 1] if bytes_per_rec_t0 > 1 else 0
            k2 = tier0_raw[rec_off + 2] if bytes_per_rec_t0 > 2 else 0
            k3 = tier0_raw[rec_off + 3] if bytes_per_rec_t0 > 3 else 0
            keys = [k0, k1, k2, k3]
            for c in range(NUM_MIH_CHUNKS):
                chunk_bucket_keys[c][i] = keys[c]
                chunk_bucket_counts[c][keys[c]] += 1

        prefix_offsets_bytes = bytearray(MIH_OFFSETS_BYTES)
        prefix_postings_bytes = bytearray(NUM_MIH_CHUNKS * num_records * 4)

        for c in range(NUM_MIH_CHUNKS):
            bucket_offsets = [0] * (NUM_MIH_BUCKETS + 1)
            running = 0
            for b in range(NUM_MIH_BUCKETS):
                bucket_offsets[b] = running
                running += chunk_bucket_counts[c][b]
            bucket_offsets[NUM_MIH_BUCKETS] = num_records

            off_base = c * (NUM_MIH_BUCKETS + 1) * 4
            for b in range(NUM_MIH_BUCKETS + 1):
                prefix_offsets_bytes[off_base + b*4 : off_base + (b+1)*4] = int(bucket_offsets[b]).to_bytes(4, byteorder="little", signed=True)

            current_ptrs = list(bucket_offsets)
            post_base = c * num_records * 4
            for i in range(num_records):
                b = chunk_bucket_keys[c][i]
                dest = current_ptrs[b]
                prefix_postings_bytes[post_base + dest*4 : post_base + (dest+1)*4] = int(i).to_bytes(4, byteorder="little", signed=True)
                current_ptrs[b] += 1

        prefix_table_bytes = prefix_offsets_bytes + prefix_postings_bytes
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
            "user_metadata": user_metadata if user_metadata else {}
        }
        for k in range(num_tiers):
            toc_dict["sections"][f"tier_{k}"] = {
                "offset": tier_offsets[k],
                "length": tier_lengths[k],
                "dim_boundary": int(tiers_arr[k])
            }
        toc_dict["sections"]["sidecar"] = {
            "offset": sidecar_offset,
            "length": sidecar_len,
            "format": sidecar_format
        }
        toc_dict["sections"]["prefix_table"] = {
            "offset": prefix_table_offset,
            "length": prefix_table_len,
            "num_chunks": 4,
            "num_buckets_per_chunk": 256,
            "format": "mih_csr_4x8"
        }
        toc_dict["sections"]["metadata"] = {
            "offset": metadata_offset,
            "length": metadata_len,
            "format": meta_format
        }
        toc_bytes = json.dumps(toc_dict, indent=2).encode("utf-8")
        toc_offset = current_offset
        toc_len = len(toc_bytes)
        current_offset = _align64(current_offset + toc_len)

        total_file_size = current_offset + 20

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


class VectorDb:
    """
    Pythonic interface to the Pithos Vector Database Engine.
    """
    _active_instances = 0
    _lock = threading.Lock()

    def __init__(self, lib_path: Optional[str] = None):
        self._ffi = NativeBindings(lib_path)
        with VectorDb._lock:
            if VectorDb._active_instances == 0:
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
        """
        Explicitly triggers GraalVM GC, OS memory release (malloc_trim),
        and Python garbage collection.
        """
        self._ffi.shrink_to_fit()

    def reset_isolate(self) -> None:
        """
        Drops all loaded indices, cleans temp directories, and re-initializes
        a fresh GraalVM isolate and coordinator.
        """
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
        lora_dim: int = 0
    ) -> Index:
        """
        Maps an existing multi-tier index into memory off-heap.
        """
        effective_path = self._unpack_container_if_needed(base_path)
        if weights is not None:
            w_arr = np.ascontiguousarray(weights, dtype=np.float32)
            status = self._ffi.lib.vdb_load_index_with_weights(
                self._ffi.thread,
                name.encode("utf-8"),
                effective_path.encode("utf-8"),
                w_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(lora_dim)
            )
        else:
            status = self._ffi.lib.vdb_load_index(
                self._ffi.thread,
                name.encode("utf-8"),
                effective_path.encode("utf-8")
            )
        self._ffi.check_status(status, f"load index '{name}'")
        idx = Index(self, name, base_path)
        self._indices[name] = idx
        return idx

    def get_index(self, name: str) -> Optional[Index]:
        """Returns the loaded index handle by name, or None."""
        return self._indices.get(name)

    def drop_index(self, name: str) -> bool:
        """Unmaps and drops an index and its attached DeltaBuffer."""
        self._delta_buffers.pop(name, None)
        self._indices.pop(name, None)
        status = self._ffi.lib.vdb_drop_index(self._ffi.thread, name.encode("utf-8"))
        return status == 0

    def create_delta_buffer(self, index_name: str, flush_threshold: int = 10000) -> DeltaBuffer:
        """Attaches an in-memory DeltaBuffer for real-time inserts."""
        status = self._ffi.lib.vdb_create_delta_buffer(
            self._ffi.thread,
            index_name.encode("utf-8"),
            ctypes.c_int(flush_threshold)
        )
        self._ffi.check_status(status, f"create DeltaBuffer for '{index_name}'")
        buf = DeltaBuffer(self, index_name)
        self._delta_buffers[index_name] = buf
        return buf

    def get_delta_buffer(self, index_name: str) -> Optional[DeltaBuffer]:
        """Returns the active DeltaBuffer for the given index, or None."""
        return self._delta_buffers.get(index_name)

    # -------------------------------------------------------------------------
    # Index Compilation & Compaction
    # -------------------------------------------------------------------------
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
        lib_path: Optional[str] = None
    ) -> None:
        """
        Compiles raw continuous float embeddings into a universal schema-agnostic single-file .pithos container (DIOGENES format).
        """
        ffi = NativeBindings(lib_path)
        vecs = np.ascontiguousarray(records, dtype=np.float32)
        num_records, dimension = vecs.shape

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
            ids_arr = np.ascontiguousarray(ids, dtype=np.int64)

        if tiers is None:
            tiers_arr = np.array([dimension], dtype=np.int32)
        else:
            tiers_arr = np.ascontiguousarray(tiers, dtype=np.int32)

        if isinstance(sidecar_mode, str):
            sidecar_map = {
                "none": SidecarMode.NONE,
                "fp16": SidecarMode.FP16,
                "fp8": SidecarMode.FP8,
                "fp4": SidecarMode.FP4
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

        if hasattr(ffi.lib, "vdb_compile_container"):
            with ffi.isolated_context() as temp_thread:
                status = ffi.lib.vdb_compile_container(
                    temp_thread,
                    path.encode("utf-8"),
                    ctypes.c_int(dimension),
                    tiers_arr.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(len(tiers_arr)),
                    ids_arr.ctypes.data_as(ctypes.c_void_p),
                    vecs.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(num_records),
                    ctypes.c_int(metric_code),
                    ctypes.c_int(int(q_mode)),
                    ctypes.c_int(int(actual_sidecar)),
                    meta_bytes_ptr,
                    ctypes.c_int(meta_len),
                    meta_fmt_ptr,
                    user_json_str
                )
                ffi.check_status(status, "compile single-file container")
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
                ffi=ffi
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
        chunk_size: int = 5000
    ) -> None:
        """
        Compiles continuous float vectors from a streaming iterator/generator directly into a
        universal single-file .pithos container (DIOGENES format) on disk with constant O(1) RAM.
        """
        import tempfile
        import glob

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
                "fp4": SidecarMode.FP4
            }
            actual_sidecar = sidecar_map.get(sidecar_mode.lower(), SidecarMode.FP8)
        else:
            actual_sidecar = SidecarMode(int(sidecar_mode))

        metric_map = {"cosine": 0, "l2": 1, "euclidean": 1, "dot": 2, "dot_product": 2}
        metric_code = metric_map.get(metric.lower(), 0)

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
            "user_metadata": user_metadata if user_metadata else {}
        }
        for k in range(num_tiers):
            toc_dict["sections"][f"tier_{k}"] = {
                "offset": tier_offsets[k],
                "length": tier_lengths[k],
                "dim_boundary": int(tiers_arr[k])
            }
        toc_dict["sections"]["sidecar"] = {
            "offset": sidecar_offset,
            "length": sidecar_len,
            "format": sidecar_format
        }
        toc_dict["sections"]["prefix_table"] = {
            "offset": prefix_table_offset,
            "length": prefix_table_length,
            "num_chunks": 4,
            "num_buckets_per_chunk": 256,
            "format": "mih_csr_4x8"
        }
        toc_dict["sections"]["metadata"] = {
            "offset": metadata_offset,
            "length": metadata_len,
            "format": meta_format
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
                        b_ids = np.ascontiguousarray(item[0], dtype=np.int64)
                        b_vecs = np.ascontiguousarray(item[1], dtype=np.float32)
                        if b_vecs.ndim == 1:
                            b_vecs = b_vecs.reshape(1, -1)
                        current_id += b_vecs.shape[0]
                        yield b_ids, b_vecs
                    elif isinstance(item, np.ndarray) and item.ndim == 2:
                        b_vecs = np.ascontiguousarray(item, dtype=np.float32)
                        b_ids = np.arange(current_id, current_id + b_vecs.shape[0], dtype=np.int64)
                        current_id += b_vecs.shape[0]
                        yield b_ids, b_vecs
                    elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple, np.ndarray)):
                        b_vecs = np.ascontiguousarray(item, dtype=np.float32)
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
                            b_vecs = np.ascontiguousarray(buffer_vecs, dtype=np.float32)
                            b_ids = np.ascontiguousarray(buffer_ids, dtype=np.int64)
                            current_id += len(buffer_vecs)
                            buffer_vecs.clear()
                            buffer_ids.clear()
                            yield b_ids, b_vecs
                if len(buffer_vecs) > 0:
                    b_vecs = np.ascontiguousarray(buffer_vecs, dtype=np.float32)
                    b_ids = np.ascontiguousarray(buffer_ids, dtype=np.int64)
                    yield b_ids, b_vecs

            processed_records = 0
            all_vector_chunk_keys = [[] for _ in range(NUM_MIH_CHUNKS)]
            with tempfile.TemporaryDirectory() as tmpdir:
                with ffi.isolated_context() as temp_thread:
                    chunk_idx = 0
                    for b_ids, b_vecs in _iterate_chunks(record_stream, chunk_size):
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
                            ctypes.c_int(int(actual_sidecar))
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
                                    for r in range(b_size):
                                        r_off = r * bpr_t0
                                        k0 = t0_data[r_off] if bpr_t0 > 0 else 0
                                        k1 = t0_data[r_off + 1] if bpr_t0 > 1 else 0
                                        k2 = t0_data[r_off + 2] if bpr_t0 > 2 else 0
                                        k3 = t0_data[r_off + 3] if bpr_t0 > 3 else 0
                                        keys = [k0, k1, k2, k3]
                                        for c in range(NUM_MIH_CHUNKS):
                                            all_vector_chunk_keys[c].append(keys[c])
                                else:
                                    shutil.copyfileobj(f_t, out_f)

                        if actual_sidecar == SidecarMode.FP16 and os.path.exists(f"{tmp_base}_fp16.bin"):
                            out_f.seek(sidecar_offset + processed_records * sidecar_bpr)
                            with open(f"{tmp_base}_fp16.bin", "rb") as f_s:
                                shutil.copyfileobj(f_s, out_f)
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
            actual_total = len(all_vector_chunk_keys[0])
            prefix_offsets_bytes = bytearray(MIH_OFFSETS_BYTES)
            prefix_postings_bytes = bytearray(NUM_MIH_CHUNKS * actual_total * 4)

            for c in range(NUM_MIH_CHUNKS):
                bucket_counts = [0] * NUM_MIH_BUCKETS
                for b_key in all_vector_chunk_keys[c]:
                    bucket_counts[b_key] += 1

                bucket_offsets = [0] * (NUM_MIH_BUCKETS + 1)
                running = 0
                for b in range(NUM_MIH_BUCKETS):
                    bucket_offsets[b] = running
                    running += bucket_counts[b]
                bucket_offsets[NUM_MIH_BUCKETS] = actual_total

                off_base = c * (NUM_MIH_BUCKETS + 1) * 4
                for b in range(NUM_MIH_BUCKETS + 1):
                    prefix_offsets_bytes[off_base + b*4 : off_base + (b+1)*4] = int(bucket_offsets[b]).to_bytes(4, byteorder="little", signed=True)

                current_ptrs = list(bucket_offsets)
                post_base = c * actual_total * 4
                for i, b_key in enumerate(all_vector_chunk_keys[c]):
                    dest = current_ptrs[b_key]
                    prefix_postings_bytes[post_base + dest*4 : post_base + (dest+1)*4] = int(i).to_bytes(4, byteorder="little", signed=True)
                    current_ptrs[b_key] += 1

            out_f.seek(prefix_table_offset)
            out_f.write(prefix_offsets_bytes)
            out_f.write(prefix_postings_bytes)

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
        lib_path: Optional[str] = None
    ) -> None:
        """
        Compiles raw continuous float embeddings into a multi-tier binary columnar Pithos index on disk.
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
                lib_path=lib_path
            )
        ffi = NativeBindings(lib_path)
        vecs = np.ascontiguousarray(records, dtype=np.float32)
        num_records, dimension = vecs.shape

        if ids is None:
            ids_arr = np.arange(num_records, dtype=np.int64)
        else:
            ids_arr = np.ascontiguousarray(ids, dtype=np.int64)

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
                "fp4": SidecarMode.FP4
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
                ctypes.c_int(int(actual_sidecar))
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
        lib_path: Optional[str] = None
    ) -> None:
        """
        Compacts multiple compiled Pithos indices into a consolidated index file.
        """
        ffi = NativeBindings(lib_path)
        joined = ";".join(source_paths)
        status = ffi.lib.vdb_compact_indexes(
            ffi.thread,
            joined.encode("utf-8"),
            target_path.encode("utf-8")
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

    # -------------------------------------------------------------------------
    # CUDA Acceleration Management
    # -------------------------------------------------------------------------
    def cuda_init(self, device_id: int = 0) -> None:
        """Initializes CUDA hardware acceleration runtime."""
        if not self._ffi._has_cuda:
            raise RuntimeError("Pithos native library was compiled without CUDA support.")
        status = self._ffi.lib.vdb_cuda_init(self._ffi.thread, ctypes.c_int(device_id))
        self._ffi.check_status(status, "initialize CUDA")

    def cuda_shutdown(self) -> None:
        """Releases CUDA runtime resources."""
        if self._ffi._has_cuda:
            self._ffi.lib.vdb_cuda_shutdown(self._ffi.thread)

    def cuda_is_available(self) -> bool:
        """Returns True if a compatible CUDA runtime is active."""
        if not self._ffi._has_cuda:
            return False
        return self._ffi.lib.vdb_cuda_is_available(self._ffi.thread) == 1
