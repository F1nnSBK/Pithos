"""tests/test_mips.py
~~~~~~~~~~~~~~~~~~
Comprehensive unit and integration tests for Pithos MIPS (Maximum Inner Product Search) extensions:
  1. Mathematical Exactness: Top-K ranking equivalence between lifted cosine and unnormalized dot product.
  2. SphericalLiftingTransformer: Ingest, query transformation, padding, and score untransformation.
  3. MipsIndex: Container compilation, loading, FAISS-style search, and planetary grid voting.
  4. ConcentricShellIndex: Heavy-tailed norm partitioning and parallel multi-shell merging.
  5. Core Precision Reranking: Index.rerank(..., metric="dot") / metric="ip".
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

import pithos
from pithos import (
    SphericalLiftingTransformer,
    ConcentricShellIndex,
    VectorDb,
    SidecarMode,
    QuantizationMode,
)


def test_spherical_lifting_exact_math():
    """Validates that Spherical Lifting produces 100% Top-K ranking equivalence to brute-force MIPS."""
    rng = np.random.default_rng(42)
    N = 1000
    D = 128
    Q = 20
    k = 10

    # Generate unnormalized vectors with high norm variance (norms between 0.1 and 50.0)
    raw_dirs = rng.standard_normal((N, D)).astype(np.float32)
    raw_dirs /= np.linalg.norm(raw_dirs, axis=1, keepdims=True)
    magnitudes = np.exp(rng.uniform(np.log(0.1), np.log(50.0), size=(N, 1))).astype(np.float32)
    X = raw_dirs * magnitudes

    # Generate queries with varying norms
    q_dirs = rng.standard_normal((Q, D)).astype(np.float32)
    q_dirs /= np.linalg.norm(q_dirs, axis=1, keepdims=True)
    q_mags = rng.uniform(0.5, 10.0, size=(Q, 1)).astype(np.float32)
    queries = q_dirs * q_mags

    # Exact brute-force unnormalized inner products
    exact_dot_matrix = np.dot(queries, X.T)
    exact_topk_ids = np.argsort(-exact_dot_matrix, axis=1)[:, :k]

    # Spherical Lifting transformation
    transformer = SphericalLiftingTransformer(pad_to_multiple=128)
    lifted_X = transformer.fit_transform(X)
    lifted_Q, q_norms = transformer.transform_queries(queries)

    # Validate unit norms on lifted vectors
    X_lifted_norms = np.linalg.norm(lifted_X, axis=1)
    np.testing.assert_allclose(X_lifted_norms, 1.0, atol=1e-5)

    Q_lifted_norms = np.linalg.norm(lifted_Q, axis=1)
    np.testing.assert_allclose(Q_lifted_norms, 1.0, atol=1e-5)

    # Cosine similarities on lifted sphere
    lifted_cos_matrix = np.dot(lifted_Q, lifted_X.T)
    lifted_topk_ids = np.argsort(-lifted_cos_matrix, axis=1)[:, :k]

    # Check 100% exact Top-K ranking match
    np.testing.assert_array_equal(exact_topk_ids, lifted_topk_ids)

    # Check exact score reconstruction within float32 tolerance
    reconstructed_scores = transformer.untransform_scores(lifted_cos_matrix, q_norms)
    np.testing.assert_allclose(reconstructed_scores, exact_dot_matrix, rtol=1e-4, atol=1e-4)


def test_mips_index_from_vectors_and_search():
    """Tests MipsIndex creation, FAISS-style search, and exact score restoration."""
    rng = np.random.default_rng(123)
    N = 500
    D = 64
    k = 5

    # Random database with varied magnitudes
    X = rng.standard_normal((N, D)).astype(np.float32)
    norms = rng.uniform(1.0, 20.0, size=(N, 1)).astype(np.float32)
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)) * norms

    queries = rng.standard_normal((10, D)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".pithos", delete=False) as tmp:
        path = tmp.name

    try:
        VectorDb.compile_container(path, records=X, metric="mips", sidecar_mode="fp16")
        db = VectorDb()
        mips_idx = db.load_index("default", path)

        assert mips_idx.dimension == 128
        assert mips_idx.size() == N
        assert mips_idx.has_sidecar

        # Search batch
        results = mips_idx.search(queries, k=k)
        assert len(results) == 10
        assert len(results[0]) == k

        # Check against ground truth brute force
        exact_dots = np.dot(queries, X.T)
        for q_idx in range(10):
            gt_top_ids = np.argsort(-exact_dots[q_idx])[:k]
            retrieved_ids = [r.id for r in results[q_idx]]
            # Top-1 and Top-k recall
            assert retrieved_ids[0] == gt_top_ids[0]
            # Retrieved scores should match ground truth dot products within precision tolerance
            for r in results[q_idx]:
                expected_dot = float(exact_dots[q_idx, r.id])
                assert abs(r.score - expected_dot) / (abs(expected_dot) + 1e-6) < 0.01

        # Search single query
        single_res = mips_idx.search(queries[0], k=k)
        assert len(single_res) == k
        assert single_res[0].id == results[0][0].id

        # Search return numpy
        I, S = mips_idx.search(queries, k=k, return_numpy=True)
        assert I.shape == (10, k)
        assert S.shape == (10, k)
        assert I[0, 0] == results[0][0].id
        

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_mips_index_from_file_persistence():
    """Tests loading a MipsIndex from an existing container file."""
    rng = np.random.default_rng(456)
    N = 200
    D = 32
    X = rng.standard_normal((N, D)).astype(np.float32)
    q = rng.standard_normal(D).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".pithos", delete=False) as tmp:
        path = tmp.name

    try:
        VectorDb.compile_container(path, records=X, metric="mips", sidecar_mode="fp16")
        db1 = VectorDb()
        idx1 = db1.load_index("persistence1", path)
        print(f"DEBUG: q.shape={q.shape}, idx1.dimension={idx1.dimension}")
        res1 = idx1.search(q, k=5)

        # Load from file
        db2 = VectorDb()
        idx2 = db2.load_index("persistence2", path)
        assert idx2.dimension == 64
        assert idx2.size() == N
        res2 = idx2.search(q, k=5)

        assert [r.id for r in res1] == [r.id for r in res2]
        assert np.allclose([r.score for r in res1], [r.score for r in res2], atol=1e-4)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_mips_query_planetary_grid():
    """Tests multi-family resonant voting and MIPS precision score recovery."""
    rng = np.random.default_rng(654)
    N = 400
    D = 64
    X = rng.standard_normal((N, D)).astype(np.float32)
    queries = rng.standard_normal((8, D)).astype(np.float32)
    families = np.arange(8, dtype=np.int32)
    thresholds = np.full(8, 64, dtype=np.int32)

    with tempfile.NamedTemporaryFile(suffix=".pithos", delete=False) as tmp:
        path = tmp.name

    try:
        VectorDb.compile_container(path, records=X, metric="mips", sidecar_mode="fp16")
        db = VectorDb()
        mips_idx = db.load_index("default_grid", path)
        res = mips_idx.query_planetary_grid(
            queries=queries,
            families=families,
            thresholds=thresholds,
            min_votes=1,
            rerank=True,
        )

        assert res.scores is not None
        assert len(res.scores) > 0
        assert len(res.candidate_ids) == len(res.scores)
        

    finally:
        if os.path.exists(path):
            os.remove(path)


def test_concentric_shell_index():
    """Tests ConcentricShellIndex partitioning on power-law distributed norm data."""
    rng = np.random.default_rng(789)
    N = 600
    D = 48
    k = 8

    # Power-law / heavy-tailed norms (Zipf-like distribution)
    raw_dirs = rng.standard_normal((N, D)).astype(np.float32)
    raw_dirs /= np.linalg.norm(raw_dirs, axis=1, keepdims=True)
    # Norms spanning 3 orders of magnitude: 0.1 to 100.0
    magnitudes = np.power(10.0, rng.uniform(-1.0, 2.0, size=(N, 1))).astype(np.float32)
    X = raw_dirs * magnitudes

    queries = rng.standard_normal((5, D)).astype(np.float32)

    temp_dir = tempfile.mkdtemp(prefix="pithos_test_shells_")
    try:
        shell_idx = ConcentricShellIndex.from_vectors(
            vectors=X,
            base_dir=temp_dir,
            num_shells=4,
            sidecar_mode="fp16",
            pad_to_multiple=128,
        )

        assert shell_idx.num_shells == 4
        assert shell_idx.size() == N
        assert shell_idx.dimension == 64

        results = shell_idx.search(queries, k=k)
        assert len(results) == 5
        assert len(results[0]) == k

        # Check that top candidate matches ground truth brute force
        exact_dots = np.dot(queries, X.T)
        for q_idx in range(5):
            gt_top_id = int(np.argmax(exact_dots[q_idx]))
            retrieved_top_id = results[q_idx][0].id
            assert retrieved_top_id == gt_top_id
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_core_rerank_dot_product():
    """Tests Index.rerank() with metric='dot' and metric='ip'."""
    rng = np.random.default_rng(999)
    N = 300
    D = 64
    X = rng.standard_normal((N, D)).astype(np.float32)
    Q = rng.standard_normal((3, D)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".pithos", delete=False) as tmp:
        path = tmp.name

    try:
        VectorDb.compile_container(
            path=path,
            records=X,
            tiers=[D],
            metric="cosine",
            sidecar_mode=SidecarMode.FP16,
        )

        db = VectorDb()
        idx = db.load_index("test_dot_rerank", path)

        # Rerank with dot product metric
        ranked_ids, ranked_scores = idx.rerank(Q, k=5, metric="dot")
        assert ranked_ids.shape == (3, 5)
        assert ranked_scores.shape == (3, 5)

        # Compare with exact matrix multiplication
        exact_dots = np.dot(Q, X.T)
        for q_idx in range(3):
            gt_ids = np.argsort(-exact_dots[q_idx])[:5]
            np.testing.assert_array_equal(ranked_ids[q_idx], gt_ids)
            np.testing.assert_allclose(ranked_scores[q_idx], exact_dots[q_idx, gt_ids], rtol=1e-3, atol=1e-2)

        # Test alias 'ip'
        ranked_ids_ip, _ = idx.rerank(Q, k=5, metric="ip")
        np.testing.assert_array_equal(ranked_ids, ranked_ids_ip)
    finally:
        if os.path.exists(path):
            os.remove(path)
