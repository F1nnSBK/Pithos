import os
import tempfile
import unittest
import numpy as np
from pithos import VectorDb, SidecarMode, QuantizationMode

class TestSidecarVectorAccess(unittest.TestCase):
    def test_fp8_container_vector_access_and_rerank(self):
        """Test get_sidecar_buffer, get_vectors and rerank on FP8 .pithos container."""
        np.random.seed(42)
        dim = 128
        num_vecs = 200
        vecs = np.random.randn(num_vecs, dim).astype(np.float32)
        # Normalize for cosine testing
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        tiers = [64, 128]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_fp8.pithos")
            VectorDb.compile_container(
                path=path,
                records=vecs,
                tiers=tiers,
                sidecar_mode=SidecarMode.FP8
            )

            with VectorDb() as db:
                idx = db.load_index("test_fp8_idx", path)
                self.assertEqual(idx.sidecar_mode, SidecarMode.FP8)
                self.assertTrue(idx.has_sidecar)

                # 1. Test get_sidecar_buffer
                buf = idx.get_sidecar_buffer()
                self.assertIsNotNone(buf)
                self.assertEqual(len(buf), num_vecs * dim)

                # 2. Test get_vectors (all vectors)
                all_decoded = idx.get_vectors()
                self.assertEqual(all_decoded.shape, (num_vecs, dim))
                # FP8 reconstruction error is very small (< 0.05 absolute max error)
                np.testing.assert_allclose(all_decoded, vecs, atol=0.06)

                # 3. Test get_vectors (subset of indices)
                sub_indices = np.array([5, 10, 42, 99], dtype=np.int64)
                sub_decoded = idx.get_vectors(sub_indices)
                self.assertEqual(sub_decoded.shape, (4, dim))
                np.testing.assert_allclose(sub_decoded, vecs[sub_indices], atol=0.06)

                # 4. Test get_vectors (scalar index)
                single_vec = idx.get_vectors(42)
                self.assertEqual(single_vec.shape, (dim,))
                np.testing.assert_allclose(single_vec, vecs[42], atol=0.06)

                # 5. Test rerank (single query)
                query = vecs[42]
                ranked_ids, scores = idx.rerank(query, candidate_indices=sub_indices, k=2, metric="cosine")
                self.assertEqual(len(ranked_ids), 2)
                self.assertEqual(ranked_ids[0], 42)
                self.assertAlmostEqual(scores[0], 1.0, places=2)

                # 6. Test rerank (batch queries)
                queries = np.array([vecs[5], vecs[10]])
                b_ranked_ids, b_scores = idx.rerank(queries, candidate_indices=sub_indices, k=3, metric="cosine")
                self.assertEqual(b_ranked_ids.shape, (2, 3))
                self.assertEqual(b_ranked_ids[0, 0], 5)
                self.assertEqual(b_ranked_ids[1, 0], 10)

                # 7. Test rerank with L2 metric
                l2_ids, l2_dists = idx.rerank(query, candidate_indices=sub_indices, k=2, metric="l2")
                self.assertEqual(l2_ids[0], 42)
                self.assertLess(l2_dists[0], 0.05)

    def test_fp4_container_vector_access_and_rerank(self):
        """Test get_sidecar_buffer, get_vectors and rerank on NVFP4 .pithos container."""
        np.random.seed(123)
        dim = 64
        num_vecs = 100
        vecs = np.random.randn(num_vecs, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        tiers = [32, 64]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_fp4.pithos")
            VectorDb.compile_container(
                path=path,
                records=vecs,
                tiers=tiers,
                sidecar_mode=SidecarMode.FP4
            )

            with VectorDb() as db:
                idx = db.load_index("test_fp4_idx", path)
                self.assertEqual(idx.sidecar_mode, SidecarMode.FP4)
                self.assertTrue(idx.has_sidecar)

                # 1. Test get_sidecar_buffer
                buf = idx.get_sidecar_buffer()
                self.assertIsNotNone(buf)
                num_blocks = (dim + 15) // 16
                self.assertEqual(len(buf), num_vecs * num_blocks * 9)

                # 2. Test get_vectors (all vectors)
                all_decoded = idx.get_vectors()
                self.assertEqual(all_decoded.shape, (num_vecs, dim))
                # Cosine similarity between reconstructed and original should be > 0.90
                sims = np.sum(all_decoded * vecs, axis=1) / (np.linalg.norm(all_decoded, axis=1) * np.linalg.norm(vecs, axis=1))
                self.assertGreater(np.mean(sims), 0.90)

                # 3. Test rerank
                query = vecs[17]
                ranked_ids, scores = idx.rerank(query, candidate_indices=[0, 10, 17, 30], k=2, metric="cosine")
                self.assertEqual(ranked_ids[0], 17)
                self.assertGreater(scores[0], 0.90)

    def test_no_sidecar_fallback(self):
        """Test get_vectors and rerank on an index without sidecar (sign reconstruction fallback)."""
        np.random.seed(999)
        dim = 64
        num_vecs = 50
        vecs = np.random.randn(num_vecs, dim).astype(np.float32)
        tiers = [32, 64]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_nosidecar.pithos")
            VectorDb.compile_container(
                path=path,
                records=vecs,
                tiers=tiers,
                sidecar_mode=SidecarMode.NONE
            )

            with VectorDb() as db:
                idx = db.load_index("test_nosidecar_idx", path)
                self.assertEqual(idx.sidecar_mode, SidecarMode.NONE)
                self.assertFalse(idx.has_sidecar)

                # Buffer should be None
                self.assertIsNone(idx.get_sidecar_buffer())

                # Reconstructed vectors should have shape (N, dim)
                recon = idx.get_vectors()
                self.assertEqual(recon.shape, (num_vecs, dim))

                # Rerank should still work (via sign reconstruction)
                query = vecs[3]
                ranked_ids, scores = idx.rerank(query, candidate_indices=[1, 2, 3, 4], k=2)
                self.assertEqual(len(ranked_ids), 2)

    def test_query_planetary_grid_auto_rerank(self):
        """Test query_planetary_grid with automatic sidecar reranking and tuple unpacking."""
        np.random.seed(777)
        dim = 128
        num_vecs = 300
        vecs = np.random.randn(num_vecs, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        tiers = [64, 128]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_grid_auto.pithos")
            VectorDb.compile_container(
                path=path,
                records=vecs,
                tiers=tiers,
                sidecar_mode=SidecarMode.FP8
            )

            with VectorDb() as db:
                idx = db.load_index("test_grid_idx", path)

                # Create 8 queries (one per family 0..7) using existing vectors with small noise
                num_q = 8
                queries = np.array([vecs[i * 10] for i in range(num_q)], dtype=np.float32)
                families = np.arange(num_q, dtype=np.int32)
                # Permissive threshold
                thresholds = np.full(num_q, 60, dtype=np.int32)

                # 1. Test backwards-compatible tuple unpacking
                resonant_count, mask = idx.query_planetary_grid(
                    queries=queries,
                    families=families,
                    thresholds=thresholds,
                    min_votes=1,
                )
                self.assertIsInstance(resonant_count, int)
                self.assertEqual(len(mask), num_vecs)

                # 2. Test object access to pre-ranked candidates
                res = idx.query_planetary_grid(
                    queries=queries,
                    families=families,
                    thresholds=thresholds,
                    min_votes=1,
                )
                self.assertTrue(res.has_reranked)
                self.assertIsNotNone(res.candidate_ids)
                self.assertIsNotNone(res.scores)
                self.assertIsNotNone(res.votes)

                # Since min_votes=1, any record with at least 1 vote should be in candidate_ids
                if len(res.candidate_ids) > 0:
                    # Scores should be sorted descending
                    for i in range(len(res.scores) - 1):
                        self.assertGreaterEqual(res.scores[i], res.scores[i+1])
                    # Query 0 was vecs[0], so vecs[0] should have score ~1.0 and be among the top matches
                    self.assertIn(0, res.candidate_ids[:10])
                    self.assertAlmostEqual(res.scores[0], 1.0, places=2)

    def test_fp16_container_and_multifile_sidecar(self):
        """Test FP16 sidecar across single-file container, multi-file index, vector decoding, and reranking."""
        np.random.seed(888)
        dim = 128
        num_vecs = 150
        vecs = np.random.randn(num_vecs, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        tiers = [64, 128]

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Single-File FP16 .pithos container
            c_path = os.path.join(tmpdir, "test_fp16.pithos")
            VectorDb.compile_container(
                path=c_path,
                records=vecs,
                tiers=tiers,
                sidecar_mode=SidecarMode.FP16
            )

            with VectorDb() as db:
                idx = db.load_index("test_fp16_c_idx", c_path)
                self.assertEqual(idx.sidecar_mode, SidecarMode.FP16)
                self.assertTrue(idx.has_sidecar)

                # Buffer check (2 bytes per dimension)
                buf = idx.get_sidecar_buffer()
                self.assertIsNotNone(buf)
                self.assertEqual(len(buf), num_vecs * dim * 2)

                # Vector decoding check
                decoded = idx.get_vectors()
                self.assertEqual(decoded.shape, (num_vecs, dim))
                # FP16 reconstruction is exact within float16 epsilon (< 0.001)
                np.testing.assert_allclose(decoded, vecs, atol=1e-3)

                # Re-ranking check
                query = vecs[25]
                ranked_ids, scores = idx.rerank(query, candidate_indices=[10, 25, 50, 75], k=3)
                self.assertEqual(ranked_ids[0], 25)
                self.assertAlmostEqual(scores[0], 1.0, places=3)

                # Auto rerank via query_planetary_grid
                num_q = 8
                queries = np.array([vecs[i * 5] for i in range(num_q)], dtype=np.float32)
                families = np.arange(num_q, dtype=np.int32)
                thresholds = np.full(num_q, 60, dtype=np.int32)

                res = idx.query_planetary_grid(queries, families, thresholds, min_votes=1)
                self.assertTrue(res.has_reranked)
                self.assertGreater(len(res.candidate_ids), 0)

            # 2. Multi-File FP16 index (_fp16.bin)
            m_path = os.path.join(tmpdir, "idx_multifile_fp16")
            VectorDb.compile_index(
                base_path=m_path,
                records=vecs,
                tiers=tiers,
                sidecar_mode=SidecarMode.FP16
            )
            self.assertTrue(os.path.exists(f"{m_path}_fp16.bin"))

            with VectorDb() as db:
                idx_m = db.load_index("test_fp16_m_idx", m_path)
                self.assertEqual(idx_m.sidecar_mode, SidecarMode.FP16)
                self.assertTrue(idx_m.has_sidecar)

                decoded_m = idx_m.get_vectors(np.array([5, 25, 99], dtype=np.int64))
                self.assertEqual(decoded_m.shape, (3, dim))
                np.testing.assert_allclose(decoded_m, vecs[[5, 25, 99]], atol=1e-3)

if __name__ == "__main__":
    unittest.main()
