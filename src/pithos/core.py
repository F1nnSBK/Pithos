from __future__ import annotations

import ctypes
import os
import math
import shutil
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Union, Dict, Any, Sequence
import numpy as np

from .ffi import NativeBindings, PithosNativeError

_FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def _encode_fp8_e4m3_scalar(val: float) -> int:
    if val == 0.0:
        return 0
    sign = 1 if val < 0.0 else 0
    abs_val = abs(val)
    if abs_val > 448.0:
        abs_val = 448.0
    if abs_val < 0.015625:  # 2^(-6) subnormal
        mantissa = min(7, int(round(abs_val * 512.0)))
        return (sign << 7) | mantissa
    exp = int(math.floor(math.log2(abs_val))) + 7
    exp = max(1, min(15, exp))
    scale = math.pow(2.0, exp - 7)
    mantissa = min(7, max(0, int(round((abs_val / scale - 1.0) * 8.0))))
    return (sign << 7) | (exp << 3) | mantissa

def _decode_fp8_e4m3_scalar(b: int) -> float:
    sign = 1 if (b & 0x80) != 0 else 0
    exp = (b >> 3) & 0x0F
    mantissa = b & 0x07
    sign_mult = -1.0 if sign == 1 else 1.0
    if exp == 0:
        return sign_mult * (mantissa / 512.0)
    else:
        scale = math.pow(2.0, exp - 7)
        return sign_mult * scale * (1.0 + mantissa / 8.0)

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
            if os.path.exists(f"{self._base_path}_fp8.bin"):
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
        cuda: bool = False
    ) -> Union[List[SearchResult], List[List[SearchResult]]]:
        """
        Performs high-performance batch k-NN search across multi-tier vectors.
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

    def batch_search(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False
    ) -> Union[List[SearchResult], List[List[SearchResult]]]:
        """
        Alias for search() performing batch k-NN search across multi-tier vectors.
        """
        return self.search(queries, k=k, cuda=cuda)

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




class VectorDb:
    """
    Pythonic interface to the Pithos Vector Database Engine.
    """
    def __init__(self, lib_path: Optional[str] = None):
        self._ffi = NativeBindings(lib_path)
        self._ffi.lib.vdb_init(self._ffi.thread)
        self._indices: Dict[str, Index] = {}
        self._delta_buffers: Dict[str, DeltaBuffer] = {}


    def __enter__(self) -> VectorDb:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Closes all loaded indices and delta buffers."""
        for name in list(self._indices.keys()):
            self.drop_index(name)
        self._ffi.lib.vdb_close(self._ffi.thread)

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
        if weights is not None:
            w_arr = np.ascontiguousarray(weights, dtype=np.float32)
            status = self._ffi.lib.vdb_load_index_with_weights(
                self._ffi.thread,
                name.encode("utf-8"),
                base_path.encode("utf-8"),
                w_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(lora_dim)
            )
        else:
            status = self._ffi.lib.vdb_load_index(
                self._ffi.thread,
                name.encode("utf-8"),
                base_path.encode("utf-8")
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
        lib_path: Optional[str] = None
    ) -> None:
        """
        Compiles raw continuous float embeddings into a multi-tier binary columnar Pithos index on disk.
        """
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

        status = ffi.lib.vdb_compile_index_file_ext(
            ffi.thread,
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
                fp8_bytes = bytearray(num_records * dimension)
                flat_vecs = vecs.flatten()
                for i in range(len(flat_vecs)):
                    fp8_bytes[i] = _encode_fp8_e4m3_scalar(flat_vecs[i])
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
            fp4_bytes = bytearray(num_records * bytes_per_rec)
            for r in range(num_records):
                row = vecs[r]
                rec_offset = r * bytes_per_rec
                for b in range(num_blocks):
                    block_start = b * 16
                    block = row[block_start : min(block_start + 16, dimension)]
                    max_val = float(np.max(np.abs(block))) if len(block) > 0 else 0.0
                    scale = max_val / 6.0 if max_val > 0.0 else 1.0
                    fp8_scale = _encode_fp8_e4m3_scalar(scale)
                    actual_scale = _decode_fp8_e4m3_scalar(fp8_scale)
                    if actual_scale == 0.0:
                        actual_scale = 1.0
                    block_offset = rec_offset + b * 9
                    fp4_bytes[block_offset] = fp8_scale
                    for j in range(8):
                        d0 = block_start + j * 2
                        d1 = block_start + j * 2 + 1
                        n0 = _encode_fp4_nibble(row[d0] / actual_scale) if d0 < dimension else 0
                        n1 = _encode_fp4_nibble(row[d1] / actual_scale) if d1 < dimension else 0
                        fp4_bytes[block_offset + 1 + j] = (n0 & 0x0F) | ((n1 & 0x0F) << 4)
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
