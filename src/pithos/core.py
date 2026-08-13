from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Union, Dict, Any, Sequence
import numpy as np

from .ffi import NativeBindings, PithosNativeError

class QuantizationMode(IntEnum):
    """Supported vector quantization modes in Pithos."""
    ONE_BIT = 0       # 1-bit sign quantization (1 bit per dimension)
    TWO_BIT = 1       # 2-bit ternary quantization (sign + mask, 2 bits per dimension)
    FLOAT32 = 2       # Unquantized 32-bit float bypass

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
        
        self._info = IndexInfo(
            dimension=dim.value,
            size=sz.value,
            planet_id=pid.value,
            planet_radius=prad.value,
            tiers_count=tcount.value
        )
        return self._info

    def __len__(self) -> int:
        return int(self._ffi.lib.vdb_size(self._ffi.thread, self._name.encode("utf-8")))

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
        
        if out_voting_mask is None:
            out_voting_mask = np.zeros(total_records, dtype=np.uint8)
        else:
            out_voting_mask = np.ascontiguousarray(out_voting_mask, dtype=np.uint8)

        if cuda and self._ffi._has_cuda:
            resonant_count = self._ffi.lib.vdb_cuda_query_planetary_grid(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                f_arr.ctypes.data_as(ctypes.c_void_p),
                t_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                out_voting_mask.ctypes.data_as(ctypes.c_void_p)
            )
        else:
            resonant_count = self._ffi.lib.vdb_query_planetary_grid(
                self._ffi.thread,
                self._name.encode("utf-8"),
                q_arr.ctypes.data_as(ctypes.c_void_p),
                f_arr.ctypes.data_as(ctypes.c_void_p),
                t_arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(num_queries),
                out_voting_mask.ctypes.data_as(ctypes.c_void_p)
            )
            
        if resonant_count < 0:
            raise PithosNativeError(int(resonant_count), "Planetary grid query failed.")
            
        return int(resonant_count), out_voting_mask

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

class VectorDb:
    """
    High-level Pythonic interface to the Pithos Vector Database Engine.
    """
    def __init__(self, lib_path: Optional[str] = None):
        self._ffi = NativeBindings(lib_path)
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
        write_fp16: bool = True,
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
            ctypes.c_int(1 if write_fp16 else 0)
        )
        ffi.check_status(status, "compile index file")

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
