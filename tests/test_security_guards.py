#!/usr/bin/env python3
"""
test_security_guards.py — Automated Security & Defensive Bounds Validation Test Suite for Pithos
"""

import os
import struct
import tempfile
import unittest
import numpy as np

from pithos import VectorDb, SidecarMode, QuantizationMode


class TestSecurityGuards(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.dim = 64
        cls.num_db = 100
        cls.vectors = np.random.randn(cls.num_db, cls.dim).astype(np.float32)

    def test_corrupted_magic_rejection(self):
        """Verify container loader rejects files with corrupted magic header."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = os.path.join(tmp_dir, "corrupt_magic.pithos")
            # Write 200 bytes of garbage
            with open(corrupt_path, "wb") as f:
                f.write(b"CORRUPT_MAGIC_HEADER_GARBAGE" * 10)

            with VectorDb() as db:
                with self.assertRaises(Exception):
                    db.load_index("corrupt_idx", corrupt_path)

    def test_out_of_bounds_toc_offset_rejection(self):
        """Verify container loader rejects forged TOC offset pointing beyond physical file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid_path = os.path.join(tmp_dir, "valid.pithos")
            VectorDb.compile_container(
                path=valid_path,
                records=self.vectors,
                tiers=[32, 64],
                q_mode=QuantizationMode.ONE_BIT,
                sidecar_mode=SidecarMode.FP8
            )

            # Read valid bytes and tamper with TOC offset in superblock (bytes 36..43)
            with open(valid_path, "rb") as f:
                data = bytearray(f.read())

            # Superblock TOC offset is at byte offset 46 (uint64_t)
            struct.pack_into("<Q", data, 46, 999_999_999)

            tampered_path = os.path.join(tmp_dir, "tampered_toc.pithos")
            with open(tampered_path, "wb") as f:
                f.write(data)

            with VectorDb() as db:
                with self.assertRaises(Exception):
                    db.load_index("tampered_idx", tampered_path)

    def test_excessive_toc_length_rejection(self):
        """Verify container loader rejects TOC length exceeding 10MB limit (JSON bomb protection)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid_path = os.path.join(tmp_dir, "valid.pithos")
            VectorDb.compile_container(
                path=valid_path,
                records=self.vectors,
                tiers=[32, 64],
                q_mode=QuantizationMode.ONE_BIT,
                sidecar_mode=SidecarMode.FP8
            )

            with open(valid_path, "rb") as f:
                data = bytearray(f.read())

            # Superblock TOC length is at byte offset 54 (uint32_t)
            struct.pack_into("<I", data, 54, 25 * 1024 * 1024)  # 25 MB

            tampered_path = os.path.join(tmp_dir, "large_toc.pithos")
            with open(tampered_path, "wb") as f:
                f.write(data)

            with VectorDb() as db:
                with self.assertRaises(Exception):
                    db.load_index("large_toc_idx", tampered_path)

    def test_invalid_dimension_rejection(self):
        """Verify container loader rejects invalid dimension <= 0 or > 65536."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            valid_path = os.path.join(tmp_dir, "valid.pithos")
            VectorDb.compile_container(
                path=valid_path,
                records=self.vectors,
                tiers=[32, 64],
                q_mode=QuantizationMode.ONE_BIT,
                sidecar_mode=SidecarMode.FP8
            )

            with open(valid_path, "rb") as f:
                data = bytearray(f.read())

            # Superblock dimension is at byte offset 20 (int32)
            struct.pack_into("<i", data, 20, -10)  # Negative dimension

            tampered_path = os.path.join(tmp_dir, "neg_dim.pithos")
            with open(tampered_path, "wb") as f:
                f.write(data)

            with VectorDb() as db:
                with self.assertRaises(Exception):
                    db.load_index("neg_dim_idx", tampered_path)

    def test_path_null_byte_rejection(self):
        """Verify path and name sanitization rejects null bytes to prevent injection attacks."""
        with VectorDb() as db:
            with self.assertRaises(Exception):
                db.load_index("test\0injection", "nonexistent.pithos")

    def test_zero_residue_query_isolation(self):
        """Verify multi-query execution leaves zero residue and returns deterministic results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            container_path = os.path.join(tmp_dir, "isolation_test.pithos")
            VectorDb.compile_container(
                path=container_path,
                records=self.vectors,
                tiers=[32, 64],
                q_mode=QuantizationMode.ONE_BIT,
                sidecar_mode=SidecarMode.FP8
            )

            with VectorDb() as db:
                index = db.load_index("iso_idx", container_path)

                # Query 1
                q1 = self.vectors[0]
                ids1_a, dists1_a = index.search_numpy(q1, k=5)

                # Query 2 (different vector)
                q2 = self.vectors[50]
                ids2, dists2 = index.search_numpy(q2, k=5)

                # Re-run Query 1 (must be bit-exact match, zero cross-query interference)
                ids1_b, dists1_b = index.search_numpy(q1, k=5)

                np.testing.assert_array_equal(ids1_a, ids1_b)
                np.testing.assert_array_equal(dists1_a, dists1_b)


if __name__ == "__main__":
    unittest.main()
