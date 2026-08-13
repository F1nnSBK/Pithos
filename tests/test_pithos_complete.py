"""
Comprehensive end-to-end verification suite for Pithos / PithosDB Python library.
Validates every single class, method, FFI binding, and edge case against a temporary dataset.
"""

import os
import shutil
import unittest
import numpy as np

# Verify dual imports
import pithos
import pithosdb
from pithos import VectorDb, Index, DeltaBuffer, SearchResult, QuantizationMode, IndexInfo


class TestPithosComplete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.abspath("temp/test_suite_run")
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        os.makedirs(cls.temp_dir, exist_ok=True)

        cls.dim = 128
        cls.num_records = 500
        cls.tiers = [32, 64, 96, 128]

        np.random.seed(1337)
        raw_vecs = np.random.randn(cls.num_records, cls.dim).astype(np.float32)
        cls.vectors = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        cls.ids = np.arange(1000, 1000 + cls.num_records, dtype=np.int64)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_dual_imports_and_version(self):
        """Validates that both 'import pithos' and 'import pithosdb' provide identical APIs and versions."""
        self.assertEqual(pithos.__version__, pithosdb.__version__)
        self.assertIs(pithos.VectorDb, pithosdb.VectorDb)
        self.assertIs(pithos.Index, pithosdb.Index)
        self.assertIs(pithos.DeltaBuffer, pithosdb.DeltaBuffer)
        self.assertIs(pithos.QuantizationMode, pithosdb.QuantizationMode)

    def test_02_compile_all_quantization_modes(self):
        """Tests compilation across 1-bit, 2-bit ternary, float32 bypass, and FP16 sidecar flag."""
        modes = [
            ("1bit", QuantizationMode.ONE_BIT, True),
            ("2bit", QuantizationMode.TWO_BIT, True),
            ("float32", QuantizationMode.FLOAT32, False),
            ("1bit_nofp16", QuantizationMode.ONE_BIT, False),
        ]
        
        for name, q_mode, write_fp16 in modes:
            base_path = os.path.join(self.temp_dir, f"idx_{name}")
            VectorDb.compile_index(
                base_path=base_path,
                records=self.vectors,
                ids=self.ids,
                tiers=self.tiers,
                planet_id=2,
                planet_radius=3389500,
                q_mode=q_mode,
                write_fp16=write_fp16
            )
            # Verify created files
            self.assertTrue(os.path.exists(base_path), f"Header missing for {name}")
            self.assertTrue(os.path.exists(f"{base_path}_ids.bin"))
            self.assertTrue(os.path.exists(f"{base_path}_metadata.bin"))
            self.assertTrue(os.path.exists(f"{base_path}_tier_0.bin"))
            self.assertTrue(os.path.exists(f"{base_path}_tier_3.bin"))
            if write_fp16:
                self.assertTrue(os.path.exists(f"{base_path}_fp16.bin"))
            else:
                self.assertFalse(os.path.exists(f"{base_path}_fp16.bin"))

    def test_03_load_and_info(self):
        """Validates index loading, metadata retrieval, dimension, size, and tier memory mapping."""
        base_path = os.path.join(self.temp_dir, "idx_1bit")
        with VectorDb() as db:
            index = db.load_index("test_info_idx", base_path)
            self.assertIsNotNone(index)
            self.assertEqual(len(index), self.num_records)
            self.assertEqual(index.size(), self.num_records)
            self.assertEqual(index.dimension, self.dim)
            self.assertEqual(index.planet_id, 2)
            self.assertEqual(index.planet_radius, 3389500)
            self.assertEqual(index.tier_count, len(self.tiers))

            # Test IndexInfo struct & dict access
            info = index.info()
            self.assertEqual(info.size, self.num_records)
            self.assertEqual(info["dimension"], self.dim)
            self.assertEqual(info["planet_id"], 2)
            
            # Test direct off-heap memory addresses
            addr, length = index.get_tier_address(0)
            self.assertGreater(addr, 0)
            self.assertGreater(length, 0)

            m_addr, m_len = index.get_metadata_address()
            self.assertGreater(m_addr, 0)
            self.assertGreater(m_len, 0)

            id_addr, id_len = index.get_ids_address()
            self.assertGreater(id_addr, 0)
            self.assertGreater(id_len, 0)

            # Test direct off-heap zero-copy buffer views (FPGA / DMA integration)
            tier_buf = index.get_tier_buffer(0)
            self.assertIsInstance(tier_buf, np.ndarray)
            self.assertEqual(tier_buf.nbytes, length)

            meta_buf = index.get_metadata_buffer()
            self.assertIsInstance(meta_buf, np.ndarray)
            self.assertEqual(len(meta_buf), self.num_records)

            ids_buf = index.get_ids_buffer()
            self.assertIsInstance(ids_buf, np.ndarray)
            self.assertEqual(len(ids_buf), self.num_records)
            self.assertEqual(ids_buf[0], self.ids[0])

            # Test FPGA hardware descriptor
            fpga_desc = index.get_fpga_descriptor(0)
            self.assertEqual(fpga_desc.record_count, self.num_records)
            self.assertEqual(fpga_desc.tier_dimension, self.dim)
            self.assertEqual(fpga_desc.tier_base_address, addr)
            self.assertEqual(fpga_desc.tier_byte_length, length)
            self.assertEqual(fpga_desc.metadata_base_address, m_addr)
            self.assertEqual(fpga_desc.ids_base_address, id_addr)

            # Test vector transformation & binarization
            packed = index.transform_and_quantize(self.vectors[0])
            self.assertEqual(len(packed), (self.dim + 63) // 64)
            self.assertIsInstance(packed, np.ndarray)

            # Test chunk size & energy budget
            index.set_chunk_size(64)
            index.set_energy_budget(0.90)



    def test_04_load_with_svd_weights(self):
        """Tests loading an index with SVD / LoRA projection weights for dynamic spectral truncation."""
        base_path = os.path.join(self.temp_dir, "idx_1bit")
        lora_dim = 32
        weights = np.random.randn(self.dim, lora_dim).astype(np.float32)

        with VectorDb() as db:
            index = db.load_index("svd_idx", base_path, weights=weights, lora_dim=lora_dim)
            self.assertEqual(len(index), self.num_records)

            # Query should succeed seamlessly with SVD spectral energy scaling
            queries = self.vectors[:5]
            results = index.search(queries, k=3)
            self.assertEqual(len(results), 5)
            for res_list in results:
                self.assertEqual(len(res_list), 3)
                for r in res_list:
                    self.assertIsInstance(r, SearchResult)
                    self.assertGreaterEqual(r.score, 0)
                    self.assertGreaterEqual(r.distance, 0.0)

    def test_05_single_and_batch_search(self):
        """Tests single-vector search and batch matrix search correctness."""
        base_path = os.path.join(self.temp_dir, "idx_1bit")
        with VectorDb() as db:
            index = db.load_index("search_idx", base_path)

            # 1. Single 1D vector query
            single_q = self.vectors[0]
            top_k_single = index.search(single_q, k=5)
            self.assertIsInstance(top_k_single, list)
            self.assertEqual(len(top_k_single), 5)
            self.assertIsInstance(top_k_single[0], SearchResult)
            # The query itself is record ID self.ids[0] (1000) so it must match with minimal score
            self.assertEqual(top_k_single[0].id, self.ids[0])

            # 2. Batch 2D matrix query
            batch_q = self.vectors[:10]
            top_k_batch = index.search(batch_q, k=5)
            self.assertEqual(len(top_k_batch), 10)
            for i, res_list in enumerate(top_k_batch):
                self.assertEqual(len(res_list), 5)
                self.assertEqual(res_list[0].id, self.ids[i])

    def test_06_resonant_voting(self):
        """Tests query_planetary_grid multi-family voting."""
        base_path = os.path.join(self.temp_dir, "idx_1bit")
        with VectorDb() as db:
            index = db.load_index("vote_idx", base_path)
            num_q = 8
            queries = self.vectors[:num_q]
            families = np.arange(num_q, dtype=np.int32) % 8
            thresholds = np.full(num_q, 50, dtype=np.int32)
            voting_mask = np.zeros(self.num_records, dtype=np.uint8)

            resonant_count, mask_out = index.query_planetary_grid(
                queries=queries,
                families=families,
                thresholds=thresholds,
                voting_mask=voting_mask
            )
            self.assertGreaterEqual(resonant_count, 0)
            self.assertEqual(len(mask_out), self.num_records)

    def test_07_lsm_delta_buffer_inserts_deletes_search_merged(self):
        """Tests LSM DeltaBuffer inserts, tombstones, and search_merged deduplication."""
        base_path = os.path.join(self.temp_dir, "idx_1bit")
        with VectorDb() as db:
            index = db.load_index("lsm_idx", base_path)
            delta = db.create_delta_buffer("lsm_idx", flush_threshold=10)
            self.assertEqual(delta.size(), 0)
            self.assertFalse(delta.needs_flush())

            # Insert new vector
            new_vec = np.ones(self.dim, dtype=np.float32)
            new_vec /= np.linalg.norm(new_vec)
            delta.insert(record_id=99999, vector=new_vec)
            self.assertEqual(delta.size(), 1)

            # Search merged: querying new_vec should find 99999 as top neighbor
            merged_res = index.search_merged(new_vec, k=5)
            self.assertGreaterEqual(len(merged_res), 1)
            self.assertEqual(merged_res[0].id, 99999)

            # Soft delete 99999
            deleted = delta.delete(99999)
            self.assertTrue(deleted)
            self.assertEqual(delta.size(), 0)

            # Search merged after tombstone: 99999 must not appear
            merged_res_after = index.search_merged(new_vec, k=5)
            returned_ids = [r.id for r in merged_res_after]
            self.assertNotIn(99999, returned_ids)

    def test_08_delta_backup_and_restore(self):
        """Tests DeltaBuffer disk backup and restore."""
        base_path = os.path.join(self.temp_dir, "idx_1bit")
        backup_file = os.path.join(self.temp_dir, "delta_test.bin")

        with VectorDb() as db:
            db.load_index("backup_idx", base_path)
            delta = db.create_delta_buffer("backup_idx", flush_threshold=100)

            vec = self.vectors[0]
            delta.insert(8888, vec)
            delta.insert(8889, vec)
            self.assertEqual(delta.size(), 2)

            # Backup
            delta.backup(backup_file)
            self.assertTrue(os.path.exists(backup_file))

            # Drop and recreate
            db.drop_index("backup_idx")

        # Open fresh db and restore
        with VectorDb() as db2:
            db2.load_index("backup_idx", base_path)
            delta2 = db2.create_delta_buffer("backup_idx", flush_threshold=100)
            delta2.restore(backup_file, flush_threshold=100)
            self.assertEqual(delta2.size(), 2)

    def test_09_index_compaction(self):
        """Tests zero-copy sidecar compaction of multiple partitioned indices into a single index."""
        p1 = os.path.join(self.temp_dir, "part_1")
        p2 = os.path.join(self.temp_dir, "part_2")
        compacted = os.path.join(self.temp_dir, "compacted_all")

        vecs1 = self.vectors[:200]
        ids1 = self.ids[:200]
        vecs2 = self.vectors[200:400]
        ids2 = self.ids[200:400]

        VectorDb.compile_index(p1, vecs1, ids1, tiers=self.tiers)
        VectorDb.compile_index(p2, vecs2, ids2, tiers=self.tiers)

        VectorDb.compact_indices([p1, p2], compacted)

        # Verify compacted index
        with VectorDb() as db:
            c_idx = db.load_index("compacted", compacted)
            self.assertEqual(len(c_idx), 400)
            self.assertEqual(c_idx.dimension, self.dim)

            # Search on compacted index
            res = c_idx.search(vecs1[0], k=3)
            self.assertEqual(res[0].id, ids1[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
