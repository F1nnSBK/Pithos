import os
import sys
import time
import ctypes
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark import PithosMIDB, generate_hypersphere_vectors

def get_dir_size(base_path):
    size = 0
    # List of all files related to the index
    for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin", "_fp16.bin"]:
        p = base_path + ext
        if os.path.exists(p):
            size += os.path.getsize(p)
    return size

def main():
    print("[Python Verify Optional FP16] Starting evaluation...")
    
    engine = PithosMIDB()
    
    N = 20000
    D = 128
    tiers = np.array([64, 128], dtype=np.int32)
    
    np.random.seed(42)
    vectors = generate_hypersphere_vectors(N, D)
    ids = np.arange(N, dtype=np.int64)
    
    # Generate queries
    num_queries = 100
    queries = generate_hypersphere_vectors(num_queries, D)
    
    # Exact Ground Truth (L2 Distance in NumPy)
    print("[Python Verify Optional FP16] Computing Ground Truth...")
    gt_ids = []
    for q in queries:
        dists = np.sum((vectors - q) ** 2, axis=1)
        gt_ids.append(np.argsort(dists)[:10])
        
    db_with_path = "temp/db_with_fp16"
    db_without_path = "temp/db_without_fp16"
    
    # 1. Compile WITH FP16 Sidecar
    print("\n[1/4] Compiling Index WITH FP16 sidecar...")
    t0 = time.perf_counter()
    status = engine.compile_index_file(db_with_path, 1, 1000, D, tiers, ids, vectors, q_mode=1, write_fp16=True)
    t_comp_with = time.perf_counter() - t0
    if status != 0:
        raise RuntimeError("Compilation WITH FP16 failed")
        
    # 2. Compile WITHOUT FP16 Sidecar
    print("[2/4] Compiling Index WITHOUT FP16 sidecar...")
    t0 = time.perf_counter()
    status = engine.compile_index_file(db_without_path, 1, 1000, D, tiers, ids, vectors, q_mode=1, write_fp16=False)
    t_comp_without = time.perf_counter() - t0
    if status != 0:
        raise RuntimeError("Compilation WITHOUT FP16 failed")
        
    size_with = get_dir_size(db_with_path) / (1024.0 * 1024.0)
    size_without = get_dir_size(db_without_path) / (1024.0 * 1024.0)
    
    # Load indexes
    engine.load_index("idx_with", db_with_path)
    engine.load_index("idx_without", db_without_path)
    
    # 3. Test Search Latency and Recall
    # WITH FP16
    print("\n[3/4] Benchmarking Search WITH FP16...")
    t0 = time.perf_counter()
    res_ids_with, _ = engine.batch_search("idx_with", queries, k=10)
    t_search_with = (time.perf_counter() - t0) * 1000.0
    
    recall_with = 0.0
    for q_idx in range(num_queries):
        intersect = np.intersect1d(res_ids_with[q_idx], gt_ids[q_idx])
        recall_with += len(intersect) / 10.0
    recall_with /= num_queries
    
    import platform
    if platform.machine().lower() in ("arm64", "aarch64"):
        print("[4/4] SKIPPING Search WITHOUT FP16 on ARM64 (SidecarMode.NONE is unsupported)")
        recall_without = 0.0
        t_search_without = 0.0
    else:
        # WITHOUT FP16
        print("[4/4] Benchmarking Search WITHOUT FP16...")
        t0 = time.perf_counter()
        res_ids_without, _ = engine.batch_search("idx_without", queries, k=10)
        t_search_without = (time.perf_counter() - t0) * 1000.0
        
        recall_without = 0.0
        for q_idx in range(num_queries):
            intersect = np.intersect1d(res_ids_without[q_idx], gt_ids[q_idx])
            recall_without += len(intersect) / 10.0
        recall_without /= num_queries
    
    # 4. Test Resonant Voting (Family Voting should be identical)
    print("\n[Voting] Testing Resonant Voting consistency...")
    families = np.random.randint(0, 8, size=num_queries, dtype=np.int32)
    thresholds = np.full(num_queries, 40, dtype=np.int32)
    
    mask_with = bytearray(N)
    t0 = time.perf_counter()
    votes_with = engine.lib.vdb_query_planetary_grid(
        engine.thread, "idx_with".encode("utf-8"),
        queries.ctypes.data_as(ctypes.c_void_p),
        families.ctypes.data_as(ctypes.c_void_p),
        thresholds.ctypes.data_as(ctypes.c_void_p),
        num_queries, ctypes.c_char_p(bytes(mask_with))
    )
    t_vote_with = (time.perf_counter() - t0) * 1000.0
    
    mask_without = bytearray(N)
    t0 = time.perf_counter()
    votes_without = engine.lib.vdb_query_planetary_grid(
        engine.thread, "idx_without".encode("utf-8"),
        queries.ctypes.data_as(ctypes.c_void_p),
        families.ctypes.data_as(ctypes.c_void_p),
        thresholds.ctypes.data_as(ctypes.c_void_p),
        num_queries, ctypes.c_char_p(bytes(mask_without))
    )
    t_vote_without = (time.perf_counter() - t0) * 1000.0
    
    # Clean up
    engine.close()
    for base in [db_with_path, db_without_path]:
        for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin", "_fp16.bin"]:
            p = base + ext
            if os.path.exists(p):
                os.remove(p)
                
    # Print comparison results table
    print("\n" + "="*80)
    print("           PITHOS COMPARISON: WITH vs. WITHOUT FP16 SIDECAR")
    print("========================================================================")
    print(f" Metric                     |  With FP16 Sidecar  |  Without FP16 Sidecar")
    print(f"----------------------------|---------------------|----------------------")
    print(f" Disk Size (MB)             |  {size_with:17.3f}  |  {size_without:20.3f}")
    print(f" Compile Time (s)           |  {t_comp_with:17.4f}  |  {t_comp_without:20.4f}")
    print(f" KNN Search Latency (ms)    |  {t_search_with:17.2f}  |  {t_search_without:20.2f}")
    print(f" KNN Recall@10              |  {recall_with*100.0:16.2f}% |  {recall_without*100.0:19.2f}%")
    print(f" Voting Match Count         |  {votes_with:17,d}  |  {votes_without:20,d}")
    print(f" Voting Latency (ms)        |  {t_vote_with:17.2f}  |  {t_vote_without:20.2f}")
    print("========================================================================")
    print("[Python Verify Optional FP16] Completed successfully!")

if __name__ == "__main__":
    main()
