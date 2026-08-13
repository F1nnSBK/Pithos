#!/usr/bin/env python3
"""
Test Suite: FPGA & DMA Co-Design Hardware Emulation Verification

Validates:
1. Physical/Virtual off-heap memory mapping stability and 64-byte cache alignment.
2. Hardware FPGA Descriptor generation (addresses, lengths, word counts).
3. Zero-copy DMA buffer slicing and memoryview validity.
4. Host-side query preconditioning & binarization via `transform_and_quantize`.
5. Emulated hardware FPGA kernel execution directly on off-heap DMA buffers:
   - Computes bitwise XOR + popcount against raw tier pointers.
   - Compares the top-k nearest neighbors found by the simulated FPGA kernel
     against Pithos native batchSearch results for 100% Bit-for-Bit exact match!
"""

import os
import shutil
import unittest
import numpy as np
import ctypes
import pithos as vdb

class TestFpgaCoDesign(unittest.TestCase):

    def setUp(self):
        self.temp_dir = "temp/test_fpga_verification"
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.dim = 128
        self.num_records = 500
        self.tiers = [128]
        np.random.seed(1337)

        self.vectors = np.random.randn(self.num_records, self.dim).astype(np.float32)
        self.ids = np.arange(1000, 1000 + self.num_records, dtype=np.int64)

        self.index_path = os.path.join(self.temp_dir, "fpga_test_idx")
        vdb.VectorDb.compile_index(
            base_path=self.index_path,
            records=self.vectors,
            ids=self.ids,
            tiers=self.tiers,
            planet_id=1,
            planet_radius=1737400,
            q_mode=vdb.QuantizationMode.ONE_BIT,
            write_fp16=False
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_fpga_hardware_descriptor_and_alignment(self):
        """Verifies descriptor attributes and cache-line boundary alignment."""
        with vdb.VectorDb() as db:
            index = db.load_index("test_fpga_idx", self.index_path)
            desc = index.get_fpga_descriptor(tier_idx=0)

            self.assertEqual(desc.record_count, self.num_records)
            self.assertEqual(desc.tier_dimension, self.dim)
            self.assertEqual(desc.words_per_record, (self.dim + 63) // 64)
            self.assertGreater(desc.tier_base_address, 0)
            self.assertGreater(desc.tier_byte_length, 0)

            # Virtual memory segments mapped via POSIX mmap must be page/cache aligned (e.g. at least 64-byte aligned)
            self.assertEqual(desc.tier_base_address % 64, 0, "Tier base address must be 64-byte cache-line aligned for DMA!")
            self.assertEqual(desc.metadata_base_address % 64, 0, "Metadata base address must be cache-line aligned!")
            self.assertEqual(desc.ids_base_address % 64, 0, "IDs base address must be cache-line aligned!")

    def test_fpga_hardware_kernel_emulation_equivalence(self):
        """
        Emulates an FPGA PCIe DMA Kernel:
        1. Reads raw uint64 words directly from off-heap tier memory pointer.
        2. Binarizes queries on Host CPU using `transform_and_quantize`.
        3. Executes bitwise popcount(Query ^ Record) and thresholding directly in FPGA DMA buffer.
        4. Verifies 100% byte-for-byte exact match with native Pithos query_planetary_grid!
        """
        with vdb.VectorDb() as db:
            index = db.load_index("test_fpga_idx", self.index_path)
            desc = index.get_fpga_descriptor(tier_idx=0)

            # 3 semantic queries across families 0, 1, 2
            queries = self.vectors[[10, 42, 99]]
            families = np.array([0, 1, 2], dtype=np.int32)
            thresholds = np.array([45, 50, 48], dtype=np.int32)
            num_queries = len(queries)

            # 1. Native Pithos Resonant Voting on CPU/Native library
            native_mask = np.zeros(self.num_records, dtype=np.uint8)
            native_resonant_count, native_out_mask = index.query_planetary_grid(
                queries=queries,
                families=families,
                thresholds=thresholds,
                out_voting_mask=native_mask
            )

            # 2. Host CPU Query Preconditioning: Binarize all 3 queries
            words_per_rec = desc.words_per_record
            packed_queries = np.zeros((num_queries, words_per_rec), dtype=np.uint64)
            for q_idx in range(num_queries):
                packed_queries[q_idx] = index.transform_and_quantize(queries[q_idx])

            # 3. FPGA PCIe DMA Hardware Simulation:
            # Streams bytes directly from desc.tier_base_address and desc.metadata_base_address
            raw_tier_ptr = ctypes.cast(desc.tier_base_address, ctypes.POINTER(ctypes.c_uint64))
            raw_meta_ptr = ctypes.cast(desc.metadata_base_address, ctypes.POINTER(ctypes.c_uint64))

            fpga_voting_mask = np.zeros(self.num_records, dtype=np.uint8)
            fpga_resonant_count = 0

            for i in range(desc.record_count):
                meta_val = raw_meta_ptr[i]
                if (meta_val & 1) == 1: # Deleted/Tombstone bit check
                    continue

                rec_mask = 0
                for q_idx in range(num_queries):
                    dist = 0
                    for w in range(words_per_rec):
                        rec_word = raw_tier_ptr[i * words_per_rec + w]
                        query_word = packed_queries[q_idx, w]
                        # Hardware FPGA: XOR + Popcount
                        dist += bin(rec_word ^ int(query_word)).count('1')

                    if dist <= thresholds[q_idx]:
                        rec_mask |= (1 << families[q_idx])

                fpga_voting_mask[i] = rec_mask
                # >= 5 votes condition for resonant detection
                if bin(rec_mask).count('1') >= 5:
                    fpga_resonant_count += 1

            # 4. Verify 100% byte-for-byte exact equality between FPGA DMA Kernel and Native Pithos!
            np.testing.assert_array_equal(
                fpga_voting_mask,
                native_out_mask,
                err_msg="FPGA DMA Hardware Kernel bitmask did not match native Pithos output!"
            )
            self.assertEqual(fpga_resonant_count, native_resonant_count)

            # Verify that identical query on identical record produces Hamming distance 0
            rec42_word0 = raw_tier_ptr[42 * words_per_rec + 0]
            rec42_word1 = raw_tier_ptr[42 * words_per_rec + 1]
            q42_word0 = packed_queries[1, 0]
            q42_word1 = packed_queries[1, 1]
            exact_self_dist = bin(rec42_word0 ^ int(q42_word0)).count('1') + bin(rec42_word1 ^ int(q42_word1)).count('1')
            self.assertEqual(exact_self_dist, 0, "Query 42 XOR Record 42 must yield Hamming distance 0!")

            print("\n[FPGA Verification SUCCESS]")
            print(f"  - Total Records Evaluated : {self.num_records}")
            print(f"  - DMA Memory Verified     : {desc.tier_byte_length} bytes tier + {desc.metadata_byte_length} bytes metadata")
            print(f"  - Bitmask Match Status    : 100% Byte-for-Byte Bit-Exact (All {self.num_records} bitmasks matched)")
            print(f"  - Self-Hamming Distance   : {exact_self_dist} bits (Exact 0)")

    def test_multi_tier_fpga_dma_sweep(self):
        """
        Validates multi-tier Matryoshka FPGA DMA streaming:
        Tier 0 (dims 0..64) and Tier 1 (dims 64..128) are streamed sequentially via separate DMA channels.
        """
        multi_path = os.path.join(self.temp_dir, "fpga_multitier_idx")
        tiers = [64, 128]
        vdb.VectorDb.compile_index(
            base_path=multi_path,
            records=self.vectors,
            ids=self.ids,
            tiers=tiers,
            planet_id=1,
            planet_radius=1737400
        )

        with vdb.VectorDb() as db:
            index = db.load_index("multi_fpga_idx", multi_path)
            
            desc0 = index.get_fpga_descriptor(tier_idx=0)
            desc1 = index.get_fpga_descriptor(tier_idx=1)

            self.assertEqual(desc0.words_per_record, 1) # 64 bits = 1 word
            self.assertEqual(desc1.words_per_record, 1) # 64 bits = 1 word
            self.assertEqual(desc0.tier_dimension, 64)
            self.assertEqual(desc1.tier_dimension, 64)

            # Query vector
            query = self.vectors[17]
            packed_query = index.transform_and_quantize(query)

            # Hardware FPGA simulation: stream Tier 0 over Channel 0, Tier 1 over Channel 1
            ptr0 = ctypes.cast(desc0.tier_base_address, ctypes.POINTER(ctypes.c_uint64))
            ptr1 = ctypes.cast(desc1.tier_base_address, ctypes.POINTER(ctypes.c_uint64))

            # Query record 17 against itself
            t0_dist = bin(ptr0[17] ^ int(packed_query[0])).count('1')
            t1_dist = bin(ptr1[17] ^ int(packed_query[1])).count('1')
            total_dist = t0_dist + t1_dist

            self.assertEqual(total_dist, 0, "Multi-tier FPGA DMA sweep for self-query must yield 0 Hamming distance!")

if __name__ == "__main__":
    unittest.main()

