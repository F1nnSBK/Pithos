#!/usr/bin/env python3
"""
test_real_model_recall.py — Automated Unit Test for Real Matryoshka Recall
"""

import os
import tempfile
import numpy as np
import pytest

from pithos import VectorDb, SidecarMode, QuantizationMode


def test_real_matryoshka_recall():
    np.random.seed(1337)
    dim = 256
    num_db = 5_000
    num_queries = 50
    k = 10
    num_clusters = 30
    decay_alpha = 0.9

    spectral_scale = np.array([(d + 1.0) ** (-decay_alpha) for d in range(dim)], dtype=np.float32)
    centroids = np.random.randn(num_clusters, dim).astype(np.float32) * spectral_scale
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    cluster_assignments = np.random.randint(0, num_clusters, size=num_db)
    noise = np.random.randn(num_db, dim).astype(np.float32) * (spectral_scale * 0.3)
    db_vectors = centroids[cluster_assignments] + noise
    db_vectors /= np.linalg.norm(db_vectors, axis=1, keepdims=True)
    db_vectors = db_vectors.astype(np.float32)

    query_assignments = np.random.randint(0, num_clusters, size=num_queries)
    query_noise = np.random.randn(num_queries, dim).astype(np.float32) * (spectral_scale * 0.3)
    query_vectors = centroids[query_assignments] + query_noise
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors.astype(np.float32)

    # Compute ground truth k-NN
    gt_knn = []
    for q in query_vectors:
        dists = np.sum((db_vectors - q) ** 2, axis=1)
        gt_knn.append(set(np.argsort(dists)[:k]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        container_path = os.path.join(tmp_dir, "test_matryoshka.pithos")
        VectorDb.compile_container(
            path=container_path,
            records=db_vectors,
            tiers=[64, 128, 256],
            q_mode=QuantizationMode.ONE_BIT,
            sidecar_mode=SidecarMode.FP8,
            user_metadata={"test": "matryoshka_recall"}
        )

        with VectorDb() as db:
            index = db.load_index("matryoshka_test", container_path)
            total_hits = 0
            for q_idx, q in enumerate(query_vectors):
                ids, dists = index.search_numpy(q, k=k)
                assert len(ids) == k
                pithos_set = set(ids)
                total_hits += len(gt_knn[q_idx].intersection(pithos_set))

            recall_10 = (total_hits / (num_queries * k)) * 100.0
            print(f"\n[Test] Real Matryoshka Recall@10: {recall_10:.2f}%")
            assert recall_10 >= 80.0, f"Expected Recall@10 >= 80.0%, got {recall_10:.2f}%"
