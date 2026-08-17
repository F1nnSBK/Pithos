"""
Unit and integration test suite for Pithos Universal Single-File Container Format (.pithos).
Validates Superblock ("DIOGENES"), Trailer ("PITHOSDB"), TOC JSON parsing, embedded metadata,
quantization tiers, and zero-copy mmap execution across 1-bit, 2-bit QJL, and FP8 sidecar modes.
"""

import os
import shutil
import unittest
import json
import numpy as np

import pithos
from pithos import VectorDb, Index, SearchResult, QuantizationMode, SidecarMode


class TestPithosContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.abspath("temp/test_container_run")
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        os.makedirs(cls.temp_dir, exist_ok=True)

        cls.dim = 128
        cls.num_records = 300
        cls.tiers = [64, 128]

        np.random.seed(42)
        raw_vecs = np.random.randn(cls.num_records, cls.dim).astype(np.float32)
        cls.vectors = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        cls.ids = np.arange(2000, 2000 + cls.num_records, dtype=np.int64)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_01_compile_and_validate_single_file_container(self):
        """Validates single-file .pithos container creation, magic bytes, and TOC structure."""
        container_path = os.path.join(self.temp_dir, "dataset_diogenes.pithos")
        user_meta = {
            "dataset": "astronomy_spectral_v1",
            "curator": "Diogenes of Sinope",
            "schema": "schema-agnostic autarkic container"
        }
        meta_payload = b"payload_spectral_lines_434nm_486nm_656nm"

        VectorDb.compile_container(
            path=container_path,
            records=self.vectors,
            ids=self.ids,
            tiers=self.tiers,
            metric="cosine",
            q_mode=QuantizationMode.ONE_BIT,
            sidecar_mode=SidecarMode.FP8,
            metadata_payload=meta_payload,
            metadata_format="raw",
            user_metadata=user_meta
        )

        # File validation
        self.assertTrue(os.path.exists(container_path))
        file_size = os.path.getsize(container_path)

        with open(container_path, "rb") as f:
            # 1. Check Superblock Magic (first 8 bytes)
            header_magic = f.read(8)
            self.assertEqual(header_magic, b"DIOGENES", "Superblock magic must be DIOGENES")

            # 2. Check Trailer Magic (last 8 bytes)
            f.seek(file_size - 8)
            trailer_magic = f.read(8)
            self.assertEqual(trailer_magic, b"PITHOSDB", "Trailer signature must be PITHOSDB")

            # 3. Read Trailer metadata (last 20 bytes: uint64 toc_offset, uint32 toc_length, 8 bytes magic)
            f.seek(file_size - 20)
            trailer_data = f.read(20)
            toc_offset = int.from_bytes(trailer_data[0:8], byteorder="little")
            toc_len = int.from_bytes(trailer_data[8:12], byteorder="little")
            self.assertEqual(trailer_data[12:20], b"PITHOSDB")

            # 4. Read and parse TOC JSON
            f.seek(toc_offset)
            toc_json_bytes = f.read(toc_len)
            toc_dict = json.loads(toc_json_bytes.decode("utf-8"))
            self.assertEqual(toc_dict.get("format"), "pithos_v2")
            self.assertIn("sections", toc_dict)
            self.assertIn("user_metadata", toc_dict)
            self.assertEqual(toc_dict["user_metadata"]["curator"], "Diogenes of Sinope")

    def test_02_load_container_and_search(self):
        """Loads .pithos container via VectorDb, validates search accuracy and user metadata access."""
        container_path = os.path.join(self.temp_dir, "dataset_diogenes.pithos")
        db = VectorDb()
        db.load_index("astronomy", container_path)

        idx = db.get_index("astronomy")
        self.assertEqual(idx.size(), self.num_records)
        self.assertEqual(idx.dimension, self.dim)

        # Validate user metadata retrieval
        user_meta = idx.user_metadata
        self.assertIsInstance(user_meta, dict)
        self.assertEqual(user_meta.get("curator"), "Diogenes of Sinope")

        # Query index: Query vector 0 must retrieve itself as top 1
        query_vec = self.vectors[0]
        results = idx.search(query_vec, k=5)
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0].id, self.ids[0])
        db.drop_index("astronomy")

    def test_03_monolithic_master_container_with_arrow_partitions(self):
        """Tests 1-file master container with embedded Apache Arrow partition table (1 Inode architecture)."""
        import pyarrow as pa
        
        # 1. Build partition table for sub-products/slices
        n_partitions = 10
        records_per_part = 100
        total_recs = n_partitions * records_per_part
        
        partitions_data = []
        for i in range(n_partitions):
            partitions_data.append({
                "partition_id": f"PART_{1000 + i}",
                "start_idx": i * records_per_part,
                "count": records_per_part,
                "domain_tag": "spectral_atlas"
            })
        arrow_table = pa.Table.from_pylist(partitions_data)
        
        master_path = os.path.join(self.temp_dir, "monolithic_master.pithos")
        vecs = np.random.randn(total_recs, self.dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        ids = np.arange(total_recs, dtype=np.int64)
        
        # 2. Compile monolithic container with embedded Arrow IPC table
        VectorDb.compile_container(
            path=master_path,
            records=vecs,
            ids=ids,
            tiers=[64, 128],
            metric="cosine",
            q_mode=QuantizationMode.ONE_BIT,
            sidecar_mode=SidecarMode.FP8,
            arrow_table=arrow_table,
            user_metadata={
                "dataset_name": "monolithic_global_master",
                "total_partitions": n_partitions,
                "architecture": "1-file master (1 Inode for dataset)"
            }
        )
        
        # 3. Load zero-copy and verify Arrow metadata and partitions
        with VectorDb() as db:
            idx = db.load_index("master_idx", master_path)
            self.assertEqual(idx.size(), total_recs)
            self.assertEqual(idx.dimension, self.dim)
            
            # Read embedded Arrow table
            retrieved_table = idx.arrow_table
            self.assertIsNotNone(retrieved_table)
            self.assertEqual(len(retrieved_table), n_partitions)
            self.assertEqual(retrieved_table.column("partition_id")[0].as_py(), "PART_1000")
            
            # Verify search on master container
            query = vecs[250] # Vector inside partition 2
            results = idx.search(query, k=5)
            self.assertEqual(len(results), 5)
            self.assertEqual(results[0].id, 250)
            
            db.drop_index("master_idx")


if __name__ == "__main__":
    unittest.main()
