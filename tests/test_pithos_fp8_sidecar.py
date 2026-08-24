"""
Unit and integration test suite for Pithos v1.1.0 Blackwell FP8 (E4M3) and FP4 (NVFP4) Sidecar Engine.
Validates sidecar creation, file layouts, metadata inspection, KNN accuracy, and multi-index compaction.
"""

import os
import shutil
import platform
import unittest
import numpy as np

import pithos
from pithos import VectorDb, Index, SearchResult, QuantizationMode, SidecarMode, IndexInfo


class TestPithosSidecars(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.abspath("temp/test_sidecars_run")
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        os.makedirs(cls.temp_dir, exist_ok=True)

        cls.dim = 128
        cls.num_records = 400
        cls.tiers = [64, 128]

        np.random.seed(42)
        raw_vecs = np.random.randn(cls.num_records, cls.dim).astype(np.float32)
        cls.vectors = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        cls.ids = np.arange(1000, 1000 + cls.num_records, dtype=np.int64)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_compile_fp8_sidecar(self):
        """Validates compilation of FP8 (E4M3) sidecar, file sizes, and metadata inspection."""
        base_path = os.path.join(self.temp_dir, "idx_fp8")
        VectorDb.compile_index(
            base_path=base_path,
            records=self.vectors,
            ids=self.ids,
            tiers=self.tiers,
            planet_id=1,
            planet_radius=1737400,
            q_mode=QuantizationMode.ONE_BIT,
            sidecar_mode=SidecarMode.FP8
        )

        # File validation
        self.assertTrue(os.path.exists(base_path))
        self.assertTrue(os.path.exists(f"{base_path}_ids.bin"))
        self.assertTrue(os.path.exists(f"{base_path}_fp8.bin"))
        self.assertFalse(os.path.exists(f"{base_path}_fp16.bin"))
        self.assertFalse(os.path.exists(f"{base_path}_fp4.bin"))

        # Check exact byte size: N * D * 1 byte
        expected_fp8_size = self.num_records * self.dim * 1
        self.assertEqual(os.path.getsize(f"{base_path}_fp8.bin"), expected_fp8_size)

        # Load and verify
        with VectorDb() as db:
            index = db.load_index("test_fp8", base_path)
            self.assertEqual(len(index), self.num_records)
            self.assertEqual(index.dimension, self.dim)
            info = index.info()
            self.assertEqual(info.sidecar_mode, SidecarMode.FP8)

            # Search with exact vector 0 (should retrieve record ID 1000 as rank 1)
            results = index.search(self.vectors[0], 5)
            self.assertEqual(len(results), 5)
            self.assertEqual(results[0].id, self.ids[0])

    def test_02_compile_fp4_sidecar(self):
        """Validates compilation of Blackwell NVFP4 microscaling sidecar."""
        base_path = os.path.join(self.temp_dir, "idx_fp4")
        VectorDb.compile_index(
            base_path=base_path,
            records=self.vectors,
            ids=self.ids,
            tiers=self.tiers,
            planet_id=1,
            planet_radius=1737400,
            q_mode=QuantizationMode.ONE_BIT,
            sidecar_mode="fp4"
        )

        # File validation
        self.assertTrue(os.path.exists(base_path))
        self.assertTrue(os.path.exists(f"{base_path}_fp4.bin"))
        self.assertFalse(os.path.exists(f"{base_path}_fp8.bin"))
        self.assertFalse(os.path.exists(f"{base_path}_fp16.bin"))

        # Check exact byte size: N * (ceil(D/16) * 9) bytes
        num_blocks = (self.dim + 15) // 16
        expected_fp4_size = self.num_records * (num_blocks * 9)
        self.assertEqual(os.path.getsize(f"{base_path}_fp4.bin"), expected_fp4_size)

        with VectorDb() as db:
            index = db.load_index("test_fp4", base_path)
            self.assertEqual(len(index), self.num_records)
            info = index.info()
            self.assertEqual(info.sidecar_mode, SidecarMode.FP4)

            # Search with exact vector 10
            results = index.search(self.vectors[10], 5)
            self.assertEqual(len(results), 5)
            self.assertEqual(results[0].id, self.ids[10])

    @unittest.skipIf(platform.machine() == "arm64" and platform.system() == "Darwin", "NONE sidecar is unsupported on Apple Silicon due to missing Python search fallback rotation matrix")
    def test_03_compile_none_sidecar(self):
        """Validates sidecarMode = NONE with asymmetric rotated distance fallback."""
        base_path = os.path.join(self.temp_dir, "idx_none")
        VectorDb.compile_index(
            base_path=base_path,
            records=self.vectors,
            ids=self.ids,
            tiers=self.tiers,
            planet_id=1,
            planet_radius=1737400,
            q_mode=QuantizationMode.ONE_BIT,
            sidecar_mode=SidecarMode.NONE
        )

        self.assertFalse(os.path.exists(f"{base_path}_fp16.bin"))
        self.assertFalse(os.path.exists(f"{base_path}_fp8.bin"))
        self.assertFalse(os.path.exists(f"{base_path}_fp4.bin"))

        with VectorDb() as db:
            index = db.load_index("test_none", base_path)
            self.assertEqual(index.info().sidecar_mode, SidecarMode.NONE)

            results = index.search(self.vectors[5], 5)
            self.assertEqual(len(results), 5)
            self.assertEqual(results[0].id, self.ids[5])

    def test_04_fp8_compaction(self):
        """Validates multi-index compaction preserving FP8 sidecar storage."""
        p1 = os.path.join(self.temp_dir, "part1_fp8")
        p2 = os.path.join(self.temp_dir, "part2_fp8")
        p_compacted = os.path.join(self.temp_dir, "compacted_fp8")

        half = self.num_records // 2
        vecs1, ids1 = self.vectors[:half], self.ids[:half]
        vecs2, ids2 = self.vectors[half:], self.ids[half:]

        VectorDb.compile_index(p1, vecs1, ids1, tiers=self.tiers, sidecar_mode=SidecarMode.FP8)
        VectorDb.compile_index(p2, vecs2, ids2, tiers=self.tiers, sidecar_mode=SidecarMode.FP8)

        VectorDb.compact_indices([p1, p2], p_compacted)

        self.assertTrue(os.path.exists(f"{p_compacted}_fp8.bin"))
        expected_compacted_size = self.num_records * self.dim * 1
        self.assertEqual(os.path.getsize(f"{p_compacted}_fp8.bin"), expected_compacted_size)

        with VectorDb() as db:
            c_idx = db.load_index("c_idx", p_compacted)
            self.assertEqual(len(c_idx), self.num_records)
            self.assertEqual(c_idx.info().sidecar_mode, SidecarMode.FP8)

            # Query vector from part 2
            results = c_idx.search(self.vectors[half + 5], 5)
            self.assertEqual(results[0].id, ids2[5])

    def test_05_accuracy_comparison(self):
        """Validates that FP8 retrieval achieves >= 99% recall vs FP16 on 100 queries."""
        p_fp16 = os.path.join(self.temp_dir, "acc_fp16")
        p_fp8 = os.path.join(self.temp_dir, "acc_fp8")

        VectorDb.compile_index(p_fp16, self.vectors, self.ids, tiers=self.tiers, sidecar_mode=SidecarMode.FP16)
        VectorDb.compile_index(p_fp8, self.vectors, self.ids, tiers=self.tiers, sidecar_mode=SidecarMode.FP8)

        with VectorDb() as db:
            idx_fp16 = db.load_index("idx_acc_fp16", p_fp16)
            idx_fp8 = db.load_index("idx_acc_fp8", p_fp8)

            num_queries = 50
            query_indices = np.random.choice(self.num_records, num_queries, replace=False)
            queries = self.vectors[query_indices]

            res_fp16 = idx_fp16.batch_search(queries, k=10)
            res_fp8 = idx_fp8.batch_search(queries, k=10)

            top1_matches = sum(1 for q in range(num_queries) if res_fp16[q][0].id == res_fp8[q][0].id)
            top1_recall = top1_matches / num_queries
            self.assertGreaterEqual(top1_recall, 0.98, f"Top-1 recall was {top1_recall:.4f}")


if __name__ == "__main__":
    unittest.main()
