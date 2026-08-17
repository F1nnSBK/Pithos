"""
Pithos v1.1.0 Blackwell FP8 / FP4 Sidecar Verification & Benchmark Script
Evaluates storage footprint reduction, recall@1/10 accuracy, and retrieval latency.
"""

import os
import sys
import time
import shutil
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pithos import VectorDb, QuantizationMode, SidecarMode


def get_index_files_size(base_path: str) -> int:
    size = 0
    extensions = ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin", "_tier_2.bin", "_tier_3.bin", "_fp16.bin", "_fp8.bin", "_fp4.bin"]
    for ext in extensions:
        p = base_path + ext
        if os.path.exists(p):
            size += os.path.getsize(p)
    return size


def main():
    print("=" * 80)
    print("  PITHOS v1.1.0 BLACKWELL FP8 / FP4 SIDECAR ENGINE VERIFICATION")
    print("=" * 80)

    temp_dir = os.path.abspath("temp/benchmark_fp8_sidecar")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    N = 25000
    D = 128
    tiers = [64, 128]
    k = 10
    num_queries = 200

    print(f"\n[Configuration] Dataset size N={N:,}, Dimension D={D}, Tiers={tiers}, Queries={num_queries}")

    # Generate synthetic embeddings on unit hypersphere (like DINOv3)
    np.random.seed(42)
    raw = np.random.randn(N, D).astype(np.float32)
    vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    ids = np.arange(1, N + 1, dtype=np.int64)

    raw_q = np.random.randn(num_queries, D).astype(np.float32)
    queries = raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True)

    # Compute Exact Ground Truth
    print("[Ground Truth] Computing exact float32 brute-force KNN...")
    gt_neighbors = []
    for q in queries:
        dists = np.sum((vectors - q) ** 2, axis=1)
        topk = np.argsort(dists)[:k]
        gt_neighbors.append(ids[topk])

    modes = [
        ("FP16 Sidecar (2.0 B/dim)", SidecarMode.FP16, os.path.join(temp_dir, "idx_fp16")),
        ("FP8 Sidecar (1.0 B/dim)",  SidecarMode.FP8,  os.path.join(temp_dir, "idx_fp8")),
        ("NVFP4 Sidecar (0.56 B/dim)", SidecarMode.FP4, os.path.join(temp_dir, "idx_fp4")),
        ("Bit-Only (No Sidecar)",     SidecarMode.NONE, os.path.join(temp_dir, "idx_none")),
    ]

    results_table = []

    for label, sidecar_mode, path in modes:
        print(f"\n--- Compiling & Evaluating: {label} ---")
        t0 = time.perf_counter()
        VectorDb.compile_index(
            base_path=path,
            records=vectors,
            ids=ids,
            tiers=tiers,
            sidecar_mode=sidecar_mode
        )
        comp_time_ms = (time.perf_counter() - t0) * 1000.0
        total_bytes = get_index_files_size(path)
        mb_size = total_bytes / (1024 * 1024)
        bytes_per_vec = total_bytes / N

        with VectorDb() as db:
            index = db.load_index(f"idx_{label}", path)

            # Warmup
            _ = index.search(queries[:5], k=k)

            # Latency benchmark
            t0 = time.perf_counter()
            retrieved = index.search(queries, k=k)
            query_time_total = time.perf_counter() - t0
            lat_per_query_us = (query_time_total / num_queries) * 1_000_000.0
            qps = num_queries / query_time_total

            # Accuracy evaluation
            recall1_count = 0
            recall10_count = 0
            for q_idx in range(num_queries):
                res = retrieved[q_idx]
                pred_ids = [r.id for r in res]
                gt_ids = set(gt_neighbors[q_idx])

                if pred_ids[0] == gt_neighbors[q_idx][0]:
                    recall1_count += 1
                overlap = len(set(pred_ids) & gt_ids)
                recall10_count += (overlap / k)

            r1 = (recall1_count / num_queries) * 100.0
            r10 = (recall10_count / num_queries) * 100.0

            results_table.append({
                "label": label,
                "size_mb": mb_size,
                "b_per_vec": bytes_per_vec,
                "latency_us": lat_per_query_us,
                "qps": qps,
                "r1": r1,
                "r10": r10
            })

            print(f"  Index Size     : {mb_size:.2f} MB ({bytes_per_vec:.1f} B/vec)")
            print(f"  Compile Time   : {comp_time_ms:.1f} ms")
            print(f"  Query Latency  : {lat_per_query_us:.1f} µs ({qps:,.0f} QPS)")
            print(f"  Recall@1       : {r1:.2f}%")
            print(f"  Recall@10      : {r10:.2f}%")

    print("\n" + "=" * 80)
    print("  SUMMARY BENCHMARK MATRIX (N=25,000, D=128)")
    print("=" * 80)
    print(f"{'Format Mode':<28} | {'Size (MB)':<10} | {'B/Vec':<8} | {'Recall@1':<10} | {'Recall@10':<10} | {'Latency (µs)':<12}")
    print("-" * 88)
    for r in results_table:
        print(f"{r['label']:<28} | {r['size_mb']:<10.2f} | {r['b_per_vec']:<8.1f} | {r['r1']:<9.2f}% | {r['r10']:<9.2f}% | {r['latency_us']:<12.1f}")
    print("=" * 80)

    # Clean up temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\n[Verification Complete] All Blackwell FP8 / FP4 sidecar benchmarks completed successfully.")


if __name__ == "__main__":
    main()
