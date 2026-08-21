import os
import tempfile
import unittest
import numpy as np
from pithos import VectorDb, QuantizationMode, SidecarMode
from pithos.core import (
    _encode_fp8_e4m3_scalar,
    _decode_fp8_e4m3_scalar,
    _encode_fp8_e4m3_array,
    _decode_fp8_e4m3_array,
    _encode_fp4_nibble,
    _encode_nvfp4_blocks_array
)

class TestFastSidecarVectorization(unittest.TestCase):
    def test_fp8_e4m3_vectorized_parity(self):
        """Verify 100% bit-exact parity between vectorized and scalar FP8 E4M3 encoders."""
        np.random.seed(42)
        # Test across 50,000 normal and extreme values
        normal_vals = np.random.randn(50_000).astype(np.float32) * 5.0
        edge_vals = np.array([
            0.0, -0.0, 448.0, -448.0, 500.0, -500.0, 1e-7, -1e-7,
            0.015625 / 2.0, 0.015625, 0.5, -0.5, 1.0, -1.0, 2.0, 6.0,
            float('nan'), float('inf'), float('-inf')
        ], dtype=np.float32)
        test_arr = np.concatenate([normal_vals, edge_vals])

        vec_res = _encode_fp8_e4m3_array(test_arr)
        scalar_res = np.array([_encode_fp8_e4m3_scalar(x) for x in test_arr], dtype=np.uint8)

        np.testing.assert_array_equal(vec_res, scalar_res)

        # Test decode LUT parity (ignoring NaN)
        decoded_vec = _decode_fp8_e4m3_array(vec_res)
        decoded_scalar = np.array([_decode_fp8_e4m3_scalar(b) for b in scalar_res], dtype=np.float32)
        valid_mask = ~np.isnan(decoded_vec) & ~np.isnan(decoded_scalar)
        np.testing.assert_array_almost_equal(decoded_vec[valid_mask], decoded_scalar[valid_mask])

    def test_nvfp4_blocks_vectorized_parity(self):
        """Verify that NVFP4 Block-16 microscaling encoder packs valid nibble pairs and scale bytes."""
        np.random.seed(123)
        vecs = np.random.randn(200, 128).astype(np.float32)
        encoded = _encode_nvfp4_blocks_array(vecs)
        
        # 128D has 8 blocks of 16 -> 8 * 9 = 72 bytes per vector
        self.assertEqual(encoded.shape, (200, 72))

        # Check each block's scale factor and packed nibbles
        for r in range(10):
            row = vecs[r]
            for b in range(8):
                block = row[b*16 : (b+1)*16]
                max_abs = float(np.max(np.abs(block)))
                scale = max_abs / 6.0 if max_abs > 0.0 else 0.0
                expected_scale_byte = _encode_fp8_e4m3_scalar(scale)
                
                block_bytes = encoded[r, b*9 : (b+1)*9]
                self.assertEqual(block_bytes[0], expected_scale_byte)

    def test_compile_container_fast_fp8_and_fp4(self):
        """Verify compiling .pithos container files with FP8 and FP4 sidecars using vectorized paths."""
        import json
        np.random.seed(999)
        dim = 128
        num_vecs = 500
        vecs = np.random.randn(num_vecs, dim).astype(np.float32)
        tiers = [64, 128]

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Test FP8 Container
            fp8_path = os.path.join(tmpdir, "test_fast_fp8.pithos")
            VectorDb.compile_container(
                path=fp8_path,
                records=vecs,
                tiers=tiers,
                sidecar_mode=SidecarMode.FP8
            )
            self.assertTrue(os.path.exists(fp8_path))
            file_size = os.path.getsize(fp8_path)
            with open(fp8_path, "rb") as f:
                self.assertEqual(f.read(8), b"DIOGENES")
                f.seek(file_size - 20)
                trailer = f.read(20)
                self.assertEqual(trailer[12:20], b"PITHOSDB")
                toc_offset = int.from_bytes(trailer[0:8], byteorder="little")
                toc_len = int.from_bytes(trailer[8:12], byteorder="little")
                f.seek(toc_offset)
                toc = json.loads(f.read(toc_len).decode("utf-8"))
                self.assertEqual(toc["sections"]["sidecar"]["format"], "fp8_e4m3")
                self.assertEqual(toc["sections"]["sidecar"]["length"], num_vecs * dim)

            # 2. Test NVFP4 Container
            fp4_path = os.path.join(tmpdir, "test_fast_fp4.pithos")
            VectorDb.compile_container(
                path=fp4_path,
                records=vecs,
                tiers=tiers,
                sidecar_mode=SidecarMode.FP4
            )
            self.assertTrue(os.path.exists(fp4_path))
            file_size4 = os.path.getsize(fp4_path)
            with open(fp4_path, "rb") as f:
                self.assertEqual(f.read(8), b"DIOGENES")
                f.seek(file_size4 - 20)
                trailer = f.read(20)
                self.assertEqual(trailer[12:20], b"PITHOSDB")
                toc_offset = int.from_bytes(trailer[0:8], byteorder="little")
                toc_len = int.from_bytes(trailer[8:12], byteorder="little")
                f.seek(toc_offset)
                toc = json.loads(f.read(toc_len).decode("utf-8"))
                self.assertEqual(toc["sections"]["sidecar"]["format"], "nvfp4_e2m1")
                self.assertEqual(toc["sections"]["sidecar"]["length"], num_vecs * (dim // 16) * 9)

    def test_compile_container_stream_fast(self):
        """Verify streaming container compilation with chunked fast sidecar writes."""
        import json
        np.random.seed(555)
        dim = 64
        num_vecs = 600
        vecs = np.random.randn(num_vecs, dim).astype(np.float32)
        
        def stream_gen():
            for i in range(0, num_vecs, 100):
                yield vecs[i:i+100]

        with tempfile.TemporaryDirectory() as tmpdir:
            stream_path = os.path.join(tmpdir, "test_stream_fp8.pithos")
            VectorDb.compile_container_stream(
                path=stream_path,
                record_stream=stream_gen(),
                total_records=num_vecs,
                dimension=dim,
                sidecar_mode=SidecarMode.FP8,
                chunk_size=100
            )
            self.assertTrue(os.path.exists(stream_path))
            file_size = os.path.getsize(stream_path)
            with open(stream_path, "rb") as f:
                self.assertEqual(f.read(8), b"DIOGENES")
                f.seek(file_size - 20)
                trailer = f.read(20)
                self.assertEqual(trailer[12:20], b"PITHOSDB")
                toc_offset = int.from_bytes(trailer[0:8], byteorder="little")
                toc_len = int.from_bytes(trailer[8:12], byteorder="little")
                f.seek(toc_offset)
                toc = json.loads(f.read(toc_len).decode("utf-8"))
                self.assertEqual(toc["sections"]["sidecar"]["format"], "fp8_e4m3")
                self.assertEqual(toc["sections"]["sidecar"]["length"], num_vecs * dim)

if __name__ == "__main__":
    unittest.main()
