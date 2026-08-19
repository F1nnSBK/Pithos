"""
Unit and integration test suite for Pithos isolate lifecycle management,
ephemeral isolate compilation, memory reclamation hooks, and chunked stream compilation.
"""

import os
import shutil
import unittest
import numpy as np

import pithos
from pithos import (
    VectorDb,
    Index,
    QuantizationMode,
    SidecarMode,
    reset_isolate,
    shrink_to_fit,
)


class TestIsolateLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.abspath("temp/test_lifecycle_run")
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        os.makedirs(cls.temp_dir, exist_ok=True)

        cls.dim = 128
        cls.num_records = 300
        cls.tiers = [64, 128]

        np.random.seed(42)
        raw_vecs = np.random.randn(cls.num_records, cls.dim).astype(np.float32)
        cls.vectors = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        cls.ids = np.arange(5000, 5000 + cls.num_records, dtype=np.int64)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_context_manager_and_close(self):
        """Validates VectorDb context manager (__enter__ / __exit__) and explicit close."""
        container_path = os.path.join(self.temp_dir, "test_cm.pithos")
        VectorDb.compile_container(
            path=container_path,
            records=self.vectors,
            ids=self.ids,
            tiers=self.tiers,
            sidecar_mode=SidecarMode.FP8
        )

        with VectorDb() as db:
            idx = db.load_index("idx_cm", container_path)
            self.assertEqual(len(idx), self.num_records)
            res = idx.search(self.vectors[0], k=3)
            self.assertEqual(len(res), 3)
            self.assertEqual(res[0].id, self.ids[0])

        # Verify db._closed is True and close is idempotent
        self.assertTrue(db._closed)
        db.close()

    def test_02_shrink_to_fit(self):
        """Validates shrink_to_fit execution on VectorDb and module-level API."""
        with VectorDb() as db:
            db.shrink_to_fit()

        shrink_to_fit()
        pithos.shrink_to_fit()

    def test_03_reset_isolate(self):
        """Validates isolate teardown and fresh re-initialization via reset_isolate."""
        container_path = os.path.join(self.temp_dir, "test_reset.pithos")
        VectorDb.compile_container(
            path=container_path,
            records=self.vectors,
            ids=self.ids,
            tiers=self.tiers,
            sidecar_mode=SidecarMode.FP8
        )

        # 1. Load and search before reset
        db1 = VectorDb()
        idx1 = db1.load_index("idx_reset1", container_path)
        res1 = idx1.search(self.vectors[5], k=3)
        self.assertEqual(res1[0].id, self.ids[5])
        db1.close()

        # 2. Reset isolate
        pithos.reset_isolate()

        # 3. Load and search after reset on fresh isolate
        with VectorDb() as db2:
            idx2 = db2.load_index("idx_reset2", container_path)
            res2 = idx2.search(self.vectors[5], k=3)
            self.assertEqual(res2[0].id, self.ids[5])

    def test_04_ephemeral_compilation_isolation(self):
        """Validates that successive compilations run in ephemeral isolates without leaking state."""
        for i in range(5):
            c_path = os.path.join(self.temp_dir, f"test_ephemeral_{i}.pithos")
            user_meta = {"iteration": i, "pipeline": "luna_stream"}
            VectorDb.compile_container(
                path=c_path,
                records=self.vectors,
                ids=self.ids,
                tiers=self.tiers,
                sidecar_mode=SidecarMode.FP8,
                user_metadata=user_meta
            )
            self.assertTrue(os.path.exists(c_path))

            with VectorDb() as db:
                idx = db.load_index(f"idx_eph_{i}", c_path)
                self.assertEqual(len(idx), self.num_records)
                self.assertEqual(idx.user_metadata.get("iteration"), i)
                res = idx.search(self.vectors[10], k=1)
                self.assertEqual(res[0].id, self.ids[10])

    def test_05_stream_compilation_batches(self):
        """Validates compile_container_stream with batch generator yielding numpy arrays."""
        stream_path = os.path.join(self.temp_dir, "test_stream_batches.pithos")
        batch_size = 50
        total = self.num_records

        def batch_generator():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                yield self.ids[start:end], self.vectors[start:end]

        user_meta = {
            "source": "stream_generator",
            "batch_size": batch_size,
            "total_records": total
        }

        VectorDb.compile_container_stream(
            path=stream_path,
            record_stream=batch_generator(),
            total_records=total,
            dimension=self.dim,
            tiers=self.tiers,
            metric="cosine",
            q_mode=QuantizationMode.ONE_BIT,
            sidecar_mode=SidecarMode.FP8,
            user_metadata=user_meta,
            chunk_size=batch_size
        )

        self.assertTrue(os.path.exists(stream_path))

        with VectorDb() as db:
            idx = db.load_index("idx_stream", stream_path)
            self.assertEqual(len(idx), total)
            self.assertEqual(idx.dimension, self.dim)
            self.assertEqual(idx.user_metadata.get("source"), "stream_generator")
            self.assertEqual(idx.info().sidecar_mode, SidecarMode.FP8)

            # Query all vectors in dataset; top-1 recall must be 100%
            queries = self.vectors[:20]
            results = idx.batch_search(queries, k=5)
            for q_idx in range(len(queries)):
                self.assertEqual(results[q_idx][0].id, self.ids[q_idx])

    def test_06_stream_compilation_single_records(self):
        """Validates compile_container_stream with generator yielding individual vector records."""
        stream_path = os.path.join(self.temp_dir, "test_stream_records.pithos")
        total = 100

        def single_record_generator():
            for i in range(total):
                yield self.vectors[i]

        VectorDb.compile_container_stream(
            path=stream_path,
            record_stream=single_record_generator(),
            total_records=total,
            dimension=self.dim,
            tiers=self.tiers,
            metric="cosine",
            q_mode=QuantizationMode.ONE_BIT,
            sidecar_mode=SidecarMode.FP8,
            chunk_size=25
        )

        self.assertTrue(os.path.exists(stream_path))

        with VectorDb() as db:
            idx = db.load_index("idx_stream_singles", stream_path)
            self.assertEqual(len(idx), total)
            res = idx.search(self.vectors[12], k=3)
            self.assertEqual(res[0].id, 12)


if __name__ == "__main__":
    unittest.main()
