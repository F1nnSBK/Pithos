"""pithos.mips
~~~~~~~~~~~
Maximum Inner Product Search (MIPS) and Asymmetric Retrieval extensions for PITHOS:
  1. SphericalLiftingTransformer: Exact mathematical reduction of unnormalized MIPS
     to unit-sphere cosine similarity (S^D) with SIMD 64-bit alignment (Neyshabur & Srebro).
  2. MipsIndex: Transparent high-level MIPS search wrapper over Pithos universal containers.
  3. ConcentricShellIndex: Magnitude-bucketing partitioner for heavy-tailed / power-law norm distributions.
"""

from __future__ import annotations

import os
import json
import math
import tempfile
import numpy as np
from typing import Optional, Union, Sequence, Tuple, List, Dict, Any

from .core import Index, VectorDb, SearchResult, PlanetaryGridResult, SidecarMode, QuantizationMode


class SphericalLiftingTransformer:
    """Exact reduction of unnormalized Maximum Inner Product Search (MIPS) to unit-sphere cosine similarity.

    Given a dataset of unnormalized vectors X in R^(N x D), let M = max_i ||x_i||_2 * (1 + eps).
    Each database vector x_i is mapped to unit sphere S^D via:
        \\tilde{x}_i = [ x_i / M,  sqrt(1 - ||x_i||_2^2 / M^2) ]

    Each query vector q in R^D is mapped to S^D via:
        \\tilde{q} = [ q / ||q||_2,  0 ]

    The inner product on S^D satisfies:
        <\\tilde{q}, \\tilde{x}_i> = <q, x_i> / (M * ||q||_2)
    which guarantees exact preservation of Top-K ranking order.

    The lifted dimension D + 1 is padded with zeros to the next multiple of `pad_to_multiple`
    (default 64) for maximum AVX-512 / ARM Neon POPCOUNT and WHT throughput.

    Parameters
    ----------
    pad_to_multiple : int, default=64
        SIMD alignment register multiple (e.g. 64 for 64-bit POPCOUNT words). Set to 1 to disable padding.
    epsilon : float, default=1e-6
        Relative safety factor applied to max norm to guarantee sqrt(1 - (norm/M)^2) is strictly real.
    """

    def __init__(self, pad_to_multiple: int = 64, epsilon: float = 1e-6):
        self.pad_to_multiple = max(1, pad_to_multiple)
        self.epsilon = epsilon
        self.max_norm: float = 1.0
        self.input_dim: int = 0
        self.lifted_dim: int = 0
        self.padded_dim: int = 0
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> SphericalLiftingTransformer:
        """Computes maximum L2 norm and dimension parameters from training/database vectors.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Input unnormalized database vectors.

        Returns
        -------
        self : SphericalLiftingTransformer
        """
        arr = np.ascontiguousarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array of vectors, got shape {arr.shape}")

        num_records, dim = arr.shape
        if num_records == 0 or dim == 0:
            raise ValueError("Input vector array cannot be empty")

        norms = np.linalg.norm(arr, axis=1)
        max_n = float(np.max(norms))
        if max_n == 0.0:
            max_n = 1.0

        self.max_norm = max_n * (1.0 + self.epsilon)
        self.input_dim = dim
        self.lifted_dim = dim + 1
        
        if self.pad_to_multiple > 1:
            self.padded_dim = int(math.ceil(self.lifted_dim / self.pad_to_multiple) * self.pad_to_multiple)
        else:
            self.padded_dim = self.lifted_dim

        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Transforms unnormalized database vectors into unit-norm lifted vectors in S^(padded_dim - 1).

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Input unnormalized database vectors.

        Returns
        -------
        lifted_X : ndarray of shape (N, padded_dim)
            Unit-norm lifted vectors padded to SIMD multiple.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted with fit() before calling transform()")

        arr = np.ascontiguousarray(X, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected 2D array of shape (N, {self.input_dim}), got {arr.shape}")

        num_records = arr.shape[0]
        out = np.zeros((num_records, self.padded_dim), dtype=np.float32)

        # Scale by 1 / M
        scaled = arr / self.max_norm
        out[:, :self.input_dim] = scaled

        # Extra dimension = sqrt(max(0, 1 - ||x||^2 / M^2))
        scaled_sq_norms = np.sum(scaled ** 2, axis=1)
        extra_dim = np.sqrt(np.maximum(0.0, 1.0 - scaled_sq_norms))
        out[:, self.input_dim] = extra_dim

        return out

    def fit_transform(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Fits transformer and maps database vectors to unit sphere in one step."""
        return self.fit(X).transform(X)

    def transform_queries(
        self, queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms unnormalized query vectors into lifted unit-norm query vectors for cosine search.

        Parameters
        ----------
        queries : ndarray of shape (Q, D) or (D,)
            Input query vectors.

        Returns
        -------
        lifted_queries : ndarray of shape (Q, padded_dim)
            Lifted query vectors with 0 in the extra dimension.
        query_norms : ndarray of shape (Q,)
            Raw L2 norms of original queries for score restoration.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted with fit() before calling transform_queries()")

        q_arr = np.ascontiguousarray(queries, dtype=np.float32)
        is_single = q_arr.ndim == 1
        if is_single:
            q_arr = q_arr.reshape(1, -1)

        if q_arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected query dimension {self.input_dim}, got {q_arr.shape[1]}")

        num_queries = q_arr.shape[0]
        q_norms = np.linalg.norm(q_arr, axis=1)
        safe_norms = np.where(q_norms == 0.0, 1.0, q_norms).reshape(-1, 1)

        # Normalized query: q / ||q||_2
        q_normalized = q_arr / safe_norms

        out = np.zeros((num_queries, self.padded_dim), dtype=np.float32)
        out[:, :self.input_dim] = q_normalized
        # Extra dimension is explicitly 0.0 (already zero-initialized)

        return (out[0], q_norms[0]) if is_single else (out, q_norms)

    def untransform_scores(
        self,
        lifted_scores: Union[float, np.ndarray],
        query_norms: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Restores exact unnormalized dot products from unit-sphere cosine similarity scores.

        Raw Score = Lifted Cosine Score * M * ||q||_2

        Parameters
        ----------
        lifted_scores : float or ndarray
            Cosine similarity scores in range [-1.0, 1.0].
        query_norms : float or ndarray
            L2 norms of the corresponding queries.

        Returns
        -------
        raw_dot_products : float or ndarray
            Exact unnormalized inner products <q, x_i>.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before untransforming scores")

        q_n = np.asarray(query_norms, dtype=np.float32)
        s = np.asarray(lifted_scores, dtype=np.float32)

        if s.ndim == 2 and q_n.ndim == 1:
            q_n = q_n.reshape(-1, 1)

        raw = s * self.max_norm * q_n
        if isinstance(lifted_scores, (float, int)) and raw.ndim == 0:
            return float(raw)
        return raw

    def to_dict(self) -> Dict[str, Any]:
        """Serializes transformer state to dictionary."""
        return {
            "pad_to_multiple": self.pad_to_multiple,
            "epsilon": self.epsilon,
            "max_norm": self.max_norm,
            "input_dim": self.input_dim,
            "lifted_dim": self.lifted_dim,
            "padded_dim": self.padded_dim,
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SphericalLiftingTransformer:
        """Restores transformer state from dictionary."""
        obj = cls(
            pad_to_multiple=data.get("pad_to_multiple", 64),
            epsilon=data.get("epsilon", 1e-6),
        )
        obj.max_norm = float(data.get("max_norm", 1.0))
        obj.input_dim = int(data.get("input_dim", 0))
        obj.lifted_dim = int(data.get("lifted_dim", 0))
        obj.padded_dim = int(data.get("padded_dim", 0))
        obj._is_fitted = bool(data.get("is_fitted", False))
        return obj


class MipsIndex:
    """High-level Maximum Inner Product Search (MIPS) vector index.

    Wraps a Pithos `.pithos` container and handles Spherical Lifting, query transformation,
    and exact unnormalized score reconstruction seamlessly.

    Parameters
    ----------
    index : Index
        Underlying Pithos Index handle containing lifted vectors.
    transformer : SphericalLiftingTransformer
        Transformer used to lift and normalize vectors.
    """

    def __init__(self, index: Index, transformer: SphericalLiftingTransformer):
        self.index = index
        self.transformer = transformer

    @property
    def dimension(self) -> int:
        """Original vector dimensionality D before lifting."""
        return self.transformer.input_dim

    @property
    def padded_dimension(self) -> int:
        """Lifted and SIMD-aligned vector dimensionality."""
        return self.transformer.padded_dim

    @property
    def d(self) -> int:
        """Original vector dimensionality (FAISS compatibility)."""
        return self.dimension

    @property
    def ntotal(self) -> int:
        """Total record count in index."""
        return len(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def size(self) -> int:
        return self.index.size()

    @property
    def has_sidecar(self) -> bool:
        return self.index.has_sidecar

    @classmethod
    def from_vectors(
        cls,
        vectors: Union[np.ndarray, Sequence[Sequence[float]]],
        path: Optional[str] = None,
        ids: Optional[Union[np.ndarray, Sequence[int]]] = None,
        tiers: Optional[Union[np.ndarray, Sequence[int]]] = None,
        sidecar_mode: Union[SidecarMode, str, int] = SidecarMode.FP8,
        q_mode: QuantizationMode = QuantizationMode.ONE_BIT,
        pad_to_multiple: int = 64,
        user_metadata: Optional[dict] = None,
        lib_path: Optional[str] = None,
    ) -> MipsIndex:
        """Constructs and compiles a MipsIndex directly from unnormalized float vectors.

        Parameters
        ----------
        vectors : ndarray of shape (N, D)
            Input unnormalized vectors.
        path : str, optional
            Destination filepath for the `.pithos` container. If None, creates a temporary container.
        ids : ndarray of shape (N,), optional
            Record IDs.
        tiers : sequence of int, optional
            Matryoshka tier dimensions.
        sidecar_mode : SidecarMode or str, default=SidecarMode.FP8
            Precision sidecar format ('none', 'fp16', 'fp8', 'fp4').
        q_mode : QuantizationMode, default=QuantizationMode.ONE_BIT
            1-bit or 2-bit quantization mode.
        pad_to_multiple : int, default=64
            SIMD alignment multiple.
        user_metadata : dict, optional
            Custom user metadata.
        lib_path : str, optional
            Path to native Pithos shared library.

        Returns
        -------
        MipsIndex
        """
        transformer = SphericalLiftingTransformer(pad_to_multiple=pad_to_multiple)
        lifted_vecs = transformer.fit_transform(vectors)

        meta = user_metadata.copy() if user_metadata is not None else {}
        meta["mips_transformer"] = transformer.to_dict()

        if path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".pithos", delete=False)
            path = tmp.name
            tmp.close()

        dim = lifted_vecs.shape[1]
        if tiers is None:
            tier_list = [dim]
        else:
            tier_list = [t for t in tiers if t <= dim]
            if not tier_list or tier_list[-1] != dim:
                tier_list.append(dim)

        VectorDb.compile_container(
            path=path,
            records=lifted_vecs,
            ids=ids,
            tiers=tier_list,
            metric="cosine",
            q_mode=q_mode,
            sidecar_mode=sidecar_mode,
            user_metadata=meta,
            lib_path=lib_path,
        )

        db = VectorDb(lib_path=lib_path)
        idx_name = f"mips_{os.path.basename(path).replace('.', '_')}"
        idx = db.load_index(idx_name, path)
        return cls(index=idx, transformer=transformer)

    @classmethod
    def from_file(cls, path: str, lib_path: Optional[str] = None) -> MipsIndex:
        """Loads a MipsIndex from an existing compiled .pithos container."""
        db = VectorDb(lib_path=lib_path)
        idx_name = f"mips_{os.path.basename(path).replace('.', '_')}"
        idx = db.load_index(idx_name, path)

        user_meta = idx.get_user_metadata()
        if user_meta and "mips_transformer" in user_meta:
            transformer = SphericalLiftingTransformer.from_dict(user_meta["mips_transformer"])
        else:
            # Fallback transformer inferred from container properties
            transformer = SphericalLiftingTransformer()
            transformer.input_dim = idx.dimension - 1
            transformer.lifted_dim = idx.dimension
            transformer.padded_dim = idx.dimension
            transformer.max_norm = 1.0
            transformer._is_fitted = True

        return cls(index=idx, transformer=transformer)

    def search(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False,
        return_numpy: bool = False,
    ) -> Union[List[SearchResult], List[List[SearchResult]], Tuple[np.ndarray, np.ndarray]]:
        """Finds Top-K nearest neighbors under Maximum Inner Product Search (MIPS).

        Parameters
        ----------
        queries : ndarray of shape (Q, D) or (D,)
            Raw unnormalized query vectors.
        k : int, default=10
            Number of closest neighbors.
        cuda : bool, default=False
            Whether to use CUDA acceleration.
        return_numpy : bool, default=False
            If True, returns tuple (I, raw_dot_products).

        Returns
        -------
        results : list of SearchResult, or tuple (I, raw_scores)
            Search results with true unnormalized inner products.
        """
        lifted_q, q_norms = self.transformer.transform_queries(queries)
        is_single = np.asarray(queries).ndim == 1

        if self.has_sidecar:
            # Fast precision sidecar MIPS reranking
            ranked_ids, ranked_sims = self.index.rerank(lifted_q, k=k, metric="cosine")
            raw_scores = self.transformer.untransform_scores(ranked_sims, q_norms)

            if return_numpy:
                return (ranked_ids, raw_scores)

            if is_single:
                res: List[SearchResult] = []
                for cid, score in zip(ranked_ids, raw_scores):
                    res.append(SearchResult(id=int(cid), score=float(score)))
                return res
            else:
                batch_res: List[List[SearchResult]] = []
                for q_idx in range(len(q_norms)):
                    q_res: List[SearchResult] = []
                    for cid, score in zip(ranked_ids[q_idx], raw_scores[q_idx]):
                        q_res.append(SearchResult(id=int(cid), score=float(score)))
                    batch_res.append(q_res)
                return batch_res
        else:
            # Approximate search via Hamming bit-slices
            raw_res = self.index.search(lifted_q, k=k, cuda=cuda, return_numpy=return_numpy)
            if return_numpy:
                out_ids, out_dists = raw_res
                # Approximate cosine sim = 1 - 2 * (d_H / B)
                B = self.transformer.padded_dim
                sims = 1.0 - 2.0 * (out_dists.astype(np.float32) / float(B))
                raw_scores = self.transformer.untransform_scores(sims, q_norms)
                return (out_ids, raw_scores)

            if is_single:
                single_res: List[SearchResult] = []
                B = self.transformer.padded_dim
                for item in raw_res:
                    approx_sim = 1.0 - 2.0 * (float(item.score) / float(B))
                    raw_score = float(self.transformer.untransform_scores(approx_sim, float(q_norms)))
                    single_res.append(SearchResult(id=item.id, score=raw_score))
                return single_res
            else:
                batch_res = []
                B = self.transformer.padded_dim
                for q_idx, q_list in enumerate(raw_res):
                    qn = float(q_norms[q_idx])
                    q_res = []
                    for item in q_list:
                        approx_sim = 1.0 - 2.0 * (float(item.score) / float(B))
                        raw_score = float(self.transformer.untransform_scores(approx_sim, qn))
                        q_res.append(SearchResult(id=item.id, score=raw_score))
                    batch_res.append(q_res)
                return batch_res

    def query_planetary_grid(
        self,
        queries: Union[np.ndarray, Sequence[Sequence[float]]],
        families: Union[np.ndarray, Sequence[int]],
        thresholds: Union[np.ndarray, Sequence[int]],
        min_votes: int = 5,
        rerank: bool = True,
        cuda: bool = False,
    ) -> PlanetaryGridResult:
        """Executes multi-family resonant gating and precision MIPS reranking on unnormalized queries."""
        lifted_q, q_norms = self.transformer.transform_queries(queries)
        res = self.index.query_planetary_grid(
            lifted_q,
            families=families,
            thresholds=thresholds,
            min_votes=min_votes,
            rerank=rerank,
            cuda=cuda,
        )

        if rerank and res.scores is not None and len(res.scores) > 0:
            # Transform maximum cosine similarity scores back to true inner product magnitudes
            # Mean or max query norm across active query consensus
            max_q_norm = float(np.max(q_norms))
            raw_scores = res.scores * self.transformer.max_norm * max_q_norm
            return PlanetaryGridResult(
                resonant_count=res.resonant_count,
                voting_mask=res.voting_mask,
                candidate_ids=res.candidate_ids,
                scores=raw_scores.astype(np.float32),
                votes=res.votes,
                masks=res.masks,
            )
        return res


class ConcentricShellIndex:
    """Magnitude-bucketing partitioner for heavy-tailed / power-law norm distributions.

    Partitions dataset X into K concentric shells based on L2 norm quantiles:
        Shell_k = { x in X | ||x||_2 in [r_{k-1}, r_k) }

    Each shell is indexed as an independent Pithos container. Queries search all shells
    in parallel and merge candidate matches via a global Top-K heap.

    Parameters
    ----------
    shells : list of MipsIndex
        Sub-indices for each concentric magnitude shell.
    shell_radii : list of tuple (float, float)
        Inner and outer radius [r_min, r_max) for each shell.
    """

    def __init__(self, shells: List[MipsIndex], shell_radii: List[Tuple[float, float]]):
        self.shells = shells
        self.shell_radii = shell_radii

    @property
    def num_shells(self) -> int:
        return len(self.shells)

    @property
    def dimension(self) -> int:
        return self.shells[0].dimension if self.shells else 0

    def size(self) -> int:
        return sum(s.size() for s in self.shells)

    def __len__(self) -> int:
        return self.size()

    @classmethod
    def from_vectors(
        cls,
        vectors: Union[np.ndarray, Sequence[Sequence[float]]],
        base_dir: Optional[str] = None,
        num_shells: int = 4,
        ids: Optional[Union[np.ndarray, Sequence[int]]] = None,
        sidecar_mode: Union[SidecarMode, str, int] = SidecarMode.FP8,
        pad_to_multiple: int = 64,
        lib_path: Optional[str] = None,
    ) -> ConcentricShellIndex:
        """Partitions vectors into concentric magnitude shells and builds sub-indices.

        Parameters
        ----------
        vectors : ndarray of shape (N, D)
            Input unnormalized vectors.
        base_dir : str, optional
            Storage directory for shell containers.
        num_shells : int, default=4
            Number of concentric shells.
        ids : ndarray of shape (N,), optional
            Record IDs.
        sidecar_mode : SidecarMode or str, default=SidecarMode.FP8
            Precision sidecar format.
        pad_to_multiple : int, default=64
            SIMD alignment multiple.
        lib_path : str, optional
            Path to native library.
        """
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        num_records, dim = arr.shape

        if ids is None:
            ids_arr = np.arange(num_records, dtype=np.int64)
        else:
            ids_arr = np.ascontiguousarray(ids, dtype=np.int64)

        if base_dir is None:
            base_dir = tempfile.mkdtemp(prefix="pithos_shells_")
        os.makedirs(base_dir, exist_ok=True)

        norms = np.linalg.norm(arr, axis=1)
        quantiles = np.linspace(0.0, 100.0, num_shells + 1)
        bin_edges = np.percentile(norms, quantiles)
        # Ensure distinct monotonically increasing edges
        bin_edges[0] = 0.0
        bin_edges[-1] = float(np.max(norms)) * 1.0001

        shells: List[MipsIndex] = []
        shell_radii: List[Tuple[float, float]] = []

        for s in range(num_shells):
            r_min = float(bin_edges[s])
            r_max = float(bin_edges[s + 1])
            if s == num_shells - 1:
                mask = (norms >= r_min) & (norms <= r_max)
            else:
                mask = (norms >= r_min) & (norms < r_max)

            shell_indices = np.where(mask)[0]
            if len(shell_indices) == 0:
                continue

            shell_vecs = arr[shell_indices]
            shell_ids = ids_arr[shell_indices]
            shell_path = os.path.join(base_dir, f"shell_{s}.pithos")

            shell_idx = MipsIndex.from_vectors(
                vectors=shell_vecs,
                path=shell_path,
                ids=shell_ids,
                sidecar_mode=sidecar_mode,
                pad_to_multiple=pad_to_multiple,
                lib_path=lib_path,
            )
            shells.append(shell_idx)
            shell_radii.append((r_min, r_max))

        return cls(shells=shells, shell_radii=shell_radii)

    def search(
        self,
        queries: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        k: int = 10,
        cuda: bool = False,
    ) -> Union[List[SearchResult], List[List[SearchResult]]]:
        """Searches across all concentric shells and merges Top-K results by true inner product.

        Parameters
        ----------
        queries : ndarray of shape (Q, D) or (D,)
            Raw unnormalized query vectors.
        k : int, default=10
            Number of nearest neighbors to retrieve.
        cuda : bool, default=False
            Whether to use CUDA GPU acceleration.

        Returns
        -------
        results : list of SearchResult, or list of list of SearchResult
        """
        is_single = np.asarray(queries).ndim == 1
        q_arr = np.ascontiguousarray(queries, dtype=np.float32)
        if is_single:
            q_arr = q_arr.reshape(1, -1)
        num_queries = q_arr.shape[0]

        all_shell_results: List[List[List[SearchResult]]] = []
        for shell in self.shells:
            s_res = shell.search(q_arr, k=k, cuda=cuda)
            if is_single:
                s_res = [s_res]
            all_shell_results.append(s_res)

        final_batch: List[List[SearchResult]] = []
        for q_idx in range(num_queries):
            merged_candidates: List[SearchResult] = []
            for s_idx in range(len(self.shells)):
                merged_candidates.extend(all_shell_results[s_idx][q_idx])

            # Sort descending by inner product score
            merged_candidates.sort(key=lambda r: r.score, reverse=True)
            final_batch.append(merged_candidates[:k])

        return final_batch[0] if is_single else final_batch
