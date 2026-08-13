#!/usr/bin/env python3
"""
Pithos FPGA & Hardware Co-Design Python Example

Demonstrates:
1. Extracting zero-copy direct NumPy memoryview buffers of off-heap tier bit vectors
2. Accessing 64-bit metadata bitmasks and record IDs without copying
3. Generating complete FpgaDescriptor structs for PCIe DMA transfer engines
4. Host-side query preconditioning and binarization (transform_and_quantize)
"""

import os
import shutil
import numpy as np
import pithos as vdb

def main():
    print("================================================================")
    print("      PITHOS FPGA & DIRECT DMA HARDWARE CO-DESIGN (PYTHON)      ")
    print("================================================================\n")

    temp_dir = "temp/fpga_py_demo"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    index_path = os.path.join(temp_dir, "fpga_embeddings")
    dimension = 256
    num_records = 2000
    tiers = [128, 256]

    # Generate synthetic embeddings
    print(f"[1/4] Preparing dataset: {num_records} vectors, D={dimension}, 2 tiers {tiers}...")
    vectors = np.random.randn(num_records, dimension).astype(np.float32)
    ids = np.arange(50000, 50000 + num_records, dtype=np.int64)

    # Compile index
    vdb.VectorDb.compile_index(
        base_path=index_path,
        records=vectors,
        ids=ids,
        tiers=tiers,
        planet_id=2,        # Mars ID
        planet_radius=3389500
    )

    with vdb.VectorDb() as db:
        index = db.load_index("mars_index", index_path)

        # 1. Hardware Descriptor for FPGA DMA Engine / PCIe Driver
        print("\n[2/4] Generating FPGA Hardware Descriptor:")
        desc = index.get_fpga_descriptor(tier_idx=0)
        print(f"  - Record Count           : {desc.record_count}")
        print(f"  - Tier 0 Dimension       : {desc.tier_dimension}")
        print(f"  - Tier 0 Base Address    : 0x{desc.tier_base_address:016x}")
        print(f"  - Tier 0 Byte Length     : {desc.tier_byte_length} bytes")
        print(f"  - Metadata Base Address  : 0x{desc.metadata_base_address:016x}")
        print(f"  - Record IDs Address     : 0x{desc.ids_base_address:016x}")
        print(f"  - Words per Record       : {desc.words_per_record} x 64-bit uint64 words")

        # 2. Zero-Copy NumPy Direct Array Views
        print("\n[3/4] Creating Zero-Copy Direct NumPy Buffers (No Heap Copying):")
        tier_buf = index.get_tier_buffer(tier_idx=0)
        meta_buf = index.get_metadata_buffer()
        ids_buf = index.get_ids_buffer()

        print(f"  - Tier 0 Buffer View     : shape={tier_buf.shape}, dtype={tier_buf.dtype}, nbytes={tier_buf.nbytes}")
        print(f"  - Metadata Buffer View   : shape={meta_buf.shape}, dtype={meta_buf.dtype}, first flag=0x{meta_buf[0]:x}")
        print(f"  - Record IDs Buffer View : shape={ids_buf.shape}, dtype={ids_buf.dtype}, first ID={ids_buf[0]}")

        # 3. Host Query Preconditioning & Binarization
        print("\n[4/4] Host CPU Query Binarization (Rademacher + Fast Walsh-Hadamard):")
        query = np.ones(dimension, dtype=np.float32)
        packed_bits = index.transform_and_quantize(query)

        print(f"  - Transformed query to {len(packed_bits)} uint64 words:")
        for w_idx, word in enumerate(packed_bits):
            print(f"    Word [{w_idx}]: 0x{word:016x}")

        print("\nReady to stream packed query and off-heap tier addresses into custom FPGA kernels!")

if __name__ == "__main__":
    main()
