"""
Backward-compatibility adapter for Pithos benchmarks and verification scripts.
Delegates all operations to the new modern 'pithos' package.
"""

from typing import Optional, List, Union, Dict, Any, Tuple, Sequence
import numpy as np
import os
import sys

# Ensure src/ is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from pithos import VectorDb, Index, DeltaBuffer, SearchResult, QuantizationMode, IndexInfo

def generate_hypersphere_vectors(num_vectors: int, dim: int = 384) -> np.ndarray:
    """Generates L2-normalized synthetic embeddings on a unit hypersphere."""
    raw_samples = np.random.normal(0.0, 1.0, size=(num_vectors, dim)).astype(np.float32)
    magnitudes = np.linalg.norm(raw_samples, axis=1, keepdims=True)
    magnitudes[magnitudes == 0] = 1.0
    return raw_samples / magnitudes

class PithosMIDB:
    """
    Backward-compatibility wrapper replicating the legacy PithosMIDB class interface
    powered directly by the official 'pithos' / 'pithosdb' package.
    """
    _instance = None

    def __new__(cls, lib_path: Optional[str] = None, cuda_enabled: bool = False, device_id: int = 0):
        if cls._instance is None:
            instance = super(PithosMIDB, cls).__new__(cls)
            instance._db = VectorDb(lib_path)
            instance._indices: Dict[str, Index] = {}
            instance._delta_buffers: Dict[str, DeltaBuffer] = {}
            instance._cuda_enabled = cuda_enabled
            if cuda_enabled:
                instance._db.cuda_init(device_id)
            cls._instance = instance
        return cls._instance

    @property
    def lib(self):
        """Direct access to C FFI library."""
        return self._db._ffi.lib

    @property
    def thread(self):
        """Direct access to GraalVM isolate thread."""
        return self._db._ffi.thread

    def load_index(self, name: str, base_path: str, weights: Optional[np.ndarray] = None, lora_dim: int = 0) -> int:
        idx = self._db.load_index(name, base_path, weights, lora_dim)
        self._indices[name] = idx
        return 0

    def get_info(self, name: str) -> IndexInfo:
        idx = self._indices.get(name) or self._db.get_index(name)
        if not idx:
            raise KeyError(f"Index '{name}' not found.")
        return idx.info()

    def size(self, name: str) -> int:
        idx = self._indices.get(name) or self._db.get_index(name)
        if not idx:
            raise KeyError(f"Index '{name}' not found.")
        return idx.size()

    def batch_search(self, name: str, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._indices.get(name) or self._db.get_index(name)
        if not idx:
            raise KeyError(f"Index '{name}' not found.")
        
        q_arr = np.ascontiguousarray(queries, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
            
        num_q = q_arr.shape[0]
        results = idx.search(q_arr, k=k, cuda=self._cuda_enabled)
        
        ids_matrix = np.full((num_q, k), -1, dtype=np.int64)
        dists_matrix = np.full((num_q, k), 2147483647, dtype=np.int32)
        
        for q_idx in range(num_q):
            q_res = results[q_idx]
            for i, r in enumerate(q_res[:k]):
                ids_matrix[q_idx, i] = r.id
                dists_matrix[q_idx, i] = r.score
                
        return ids_matrix, dists_matrix


    def query_planetary_grid(
        self, name: str, queries: np.ndarray, families: np.ndarray, thresholds: np.ndarray, voting_mask: np.ndarray
    ) -> int:
        idx = self._indices.get(name) or self._db.get_index(name)
        if not idx:
            raise KeyError(f"Index '{name}' not found.")
        count, _ = idx.query_planetary_grid(queries, families, thresholds, voting_mask, cuda=self._cuda_enabled)
        return count

    def compile_index_file(
        self, path: str, planet_id: int, planet_radius: int, dimension: int, tiers: np.ndarray,
        ids: np.ndarray, vectors: np.ndarray, q_mode: int = 0, write_fp16: bool = True
    ) -> int:
        VectorDb.compile_index(
            base_path=path,
            records=vectors,
            ids=ids,
            tiers=tiers,
            planet_id=planet_id,
            planet_radius=planet_radius,
            q_mode=QuantizationMode(q_mode),
            write_fp16=write_fp16
        )
        return 0

    def compile_index_file_ext(
        self, path: str, planet_id: int, planet_radius: int, dimension: int, tiers: np.ndarray,
        ids: np.ndarray, vectors: np.ndarray, q_mode: int = 0, write_fp16: bool = True
    ) -> int:
        return self.compile_index_file(
            path, planet_id, planet_radius, dimension, tiers, ids, vectors, q_mode=q_mode, write_fp16=write_fp16
        )


    def create_delta_buffer(self, index_name: str, flush_threshold: int = 10000) -> int:
        buf = self._db.create_delta_buffer(index_name, flush_threshold)
        self._delta_buffers[index_name] = buf
        return 0

    def insert(self, index_name: str, record_id: int, vector: np.ndarray) -> int:
        buf = self._delta_buffers.get(index_name) or self._db.get_delta_buffer(index_name)
        if not buf:
            buf = self._db.create_delta_buffer(index_name)
            self._delta_buffers[index_name] = buf
        buf.insert(record_id, vector)
        return 0

    def delete_from_delta(self, index_name: str, record_id: int) -> int:
        buf = self._delta_buffers.get(index_name) or self._db.get_delta_buffer(index_name)
        if not buf:
            return 0
        success = buf.delete(record_id)
        return 1 if success else 0

    def delta_size(self, index_name: str) -> int:
        buf = self._delta_buffers.get(index_name) or self._db.get_delta_buffer(index_name)
        return buf.size() if buf else 0

    def needs_flush(self, index_name: str) -> bool:
        buf = self._delta_buffers.get(index_name) or self._db.get_delta_buffer(index_name)
        return buf.needs_flush() if buf else False

    def search_merged(self, index_name: str, query: np.ndarray, k: int = 10) -> Tuple[List[int], List[int]]:
        idx = self._indices.get(index_name) or self._db.get_index(index_name)
        if not idx:
            raise KeyError(f"Index '{index_name}' not found.")
        results = idx.search_merged(query, k=k)
        ids = [r.id for r in results]
        dists = [r.score for r in results]
        return ids, dists

    def backup_delta(self, index_name: str, path: str) -> int:
        buf = self._delta_buffers.get(index_name) or self._db.get_delta_buffer(index_name)
        if not buf:
            raise KeyError(f"Delta buffer for '{index_name}' not found.")
        buf.backup(path)
        return 0

    def restore_delta(self, index_name: str, path: str, flush_threshold: int = 10000) -> int:
        buf = self._delta_buffers.get(index_name) or self._db.get_delta_buffer(index_name)
        if not buf:
            buf = self._db.create_delta_buffer(index_name, flush_threshold)
            self._delta_buffers[index_name] = buf
        buf.restore(path, flush_threshold)
        return 0

    def compact_indexes(self, source_paths: Union[str, Sequence[str]], target_path: str) -> int:
        if isinstance(source_paths, str):
            sources = source_paths.split(";")
        else:
            sources = list(source_paths)
        VectorDb.compact_indices(sources, target_path)
        return 0

    def drop_index(self, name: str) -> bool:
        self._indices.pop(name, None)
        self._delta_buffers.pop(name, None)
        return self._db.drop_index(name)

    def close(self) -> None:
        self._db.close()
        PithosMIDB._instance = None
