#!/usr/bin/env python3
"""
Pithos Quickstart: Model-Isomorphic Vector Database (MIDB) Python Example

Demonstrates:
1. Compiling a multi-tier binary columnar index from continuous float embeddings
2. Memory-mapping the index off-heap with zero GC overhead
3. Dynamic Matryoshka spectral energy budget targeting (tau early-exit)
4. Real-time LSM DeltaBuffer ingestion (inserts & soft deletes)
5. Unified merged search (base memory-mapped index + live delta buffer)
"""

import os
import shutil
import numpy as np
import pithos as vdb

def main():
    print("================================================================")
    print("         PITHOS MODEL-ISOMORPHIC VECTOR DATABASE (PYTHON)        ")
    print("================================================================\n")

    temp_dir = "temp/python_quickstart"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    index_path = os.path.join(temp_dir, "space_embeddings")
    dimension = 128
    num_records = 5000
    tiers = [32, 64, 128]

    # Generate synthetic embeddings (e.g. from CLIP / Sentence-Transformers)
    print(f"[1/5] Generating {num_records} vectors (D={dimension}) across 3 Matryoshka tiers {tiers}...")
    np.random.seed(42)
    vectors = np.random.randn(num_records, dimension).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(1000, 1000 + num_records, dtype=np.int64)

    # 1. Compile multi-tier binary columnar index with FP16 sidecar
    print(f"[2/5] Compiling binary columnar index at '{index_path}'...")
    vdb.VectorDb.compile_index(
        base_path=index_path,
        records=vectors,
        ids=ids,
        tiers=tiers,
        planet_id=1,        # Domain tag
        planet_radius=1737400,
        q_mode=vdb.QuantizationMode.ONE_BIT,
        write_fp16=True
    )

    # 2. Open database context and map index off-heap
    with vdb.VectorDb() as db:
        print("[3/5] Memory-mapping index off-heap...")
        index = db.load_index("space_index", index_path)
        info = index.info()
        print(f"  - Loaded Index '{index.name}': {info.size} records, {info.dimension} dims, {info.tiers_count} tiers")

        # Set early-exit cumulative spectral energy budget tau in (0, 1]
        index.set_energy_budget(0.85)

        # 3. High-throughput k-NN Search
        query_vector = vectors[0]
        results = index.search(query_vector, k=5)
        print("\n[4/5] Top-5 Nearest Neighbors (Base Index):")
        for rank, res in enumerate(results, start=1):
            print(f"  Rank {rank}: Record ID = {res.id:5d} | Distance = {res.score / 1_000_000.0:.4f}")

        # 4. Real-time Ingestion via LSM DeltaBuffer
        print("\n[5/5] Attaching LSM DeltaBuffer for Real-Time Streaming Ingestion...")
        delta = db.create_delta_buffer("space_index", flush_threshold=1000)

        new_id = 99999
        new_vector = query_vector + np.random.normal(0, 0.01, size=dimension).astype(np.float32)
        delta.insert(new_id, new_vector)
        print(f"  - Inserted new record ID {new_id} into live memory buffer (delta size: {delta.size()})")

        # Merged Search across immutable base index + active delta buffer
        merged_results = index.search_merged(query_vector, k=5)
        print("\nTop-5 Merged Search Results (Base + Delta):")
        for rank, res in enumerate(merged_results, start=1):
            origin = "LSM Delta" if res.id == new_id else "Base Index"
            print(f"  Rank {rank}: Record ID = {res.id:5d} | Distance = {res.score / 1_000_000.0:.4f} | Origin = {origin}")

    print("\nDatabase closed and memory unmapped cleanly.")

if __name__ == "__main__":
    main()
