import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import argparse
import numpy as np
from benchmark import PithosMIDB

DB_FILE = "temp/benchmark_data/lunar_real_data"
DIMENSION = 384
TIERS = np.array([64, 128, 256, 384], dtype=np.int32)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pithos Real Data Verification")
    parser.add_argument("--trace", action="store_true", help="Enable deep execution profiling with step-by-step timestamps")
    return parser.parse_args()

def format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds * 1e6:.2f} µs"

def print_performance_table(duration_p1: float, duration_p2: float, trace_data: dict = None):
    total_time = duration_p1 + duration_p2
    
    col1_title = "Pipeline Stage / Operational Step"
    col2_title = "Duration"
    col3_title = "Budget %"
    
    w1 = 54
    w2 = 13
    w3 = 11
    
    print("\n" + " " * 20 + "Pipeline Performance Summary")
    print(f"┌{'─' * w1}┬{'─' * w2}┬{'─' * w3}┐")
    print(f"│ {col1_title:<{w1-2}} │ {col2_title:>{w2-2}} │ {col3_title:>{w3-2}} │")
    print(f"├{'─' * w1}┼{'─' * w2}┼{'─' * w3}┤")
    
    def print_row(label, val, pct):
        print(f"│ {label:<{w1-2}} │ {val:>{w2-2}} │ {pct:>{w3-2}} │")
        
    print_row("Phase 1: Ingestion & Distance Analysis (Total)", format_duration(duration_p1), f"{(duration_p1/total_time)*100:5.1f}%")
    if trace_data:
        for k, v in trace_data.items():
            if k.startswith("p1_"):
                name = "  └─ " + k[3:].replace("_", " ").title()
                print_row(name, format_duration(v), f"{(v/total_time)*100:5.1f}%")
                
    print_row("Phase 2: Native FFI Scan & Verification (Total)", format_duration(duration_p2), f"{(duration_p2/total_time)*100:5.1f}%")
    if trace_data:
        for k, v in trace_data.items():
            if k.startswith("p2_"):
                name = "  └─ " + k[3:].replace("_", " ").title()
                print_row(name, format_duration(v), f"{(v/total_time)*100:5.1f}%")
                
    print(f"├{'─' * w1}┼{'─' * w2}┼{'─' * w3}┤")
    print_row("Total Pipeline Execution Time", format_duration(total_time), "100.0%")
    print(f"└{'─' * w1}┴{'─' * w2}┴{'─' * w3}┘\n")

def get_java_random_signs(dimension, seed=42):
    current_seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    signs = []
    for _ in range(dimension):
        current_seed = (current_seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        val = current_seed >> 47
        signs.append(1.0 if val != 0 else -1.0)
    return np.array(signs, dtype=np.float32)

def get_hadamard_matrix(n):
    if n == 1:
        return np.array([[1.0]], dtype=np.float32)
    H_prev = get_hadamard_matrix(n // 2)
    return np.block([[H_prev, H_prev], [H_prev, -H_prev]]) / np.sqrt(2.0)

def transform_and_binarize(vectors, dimension=384):
    signs = get_java_random_signs(dimension)
    z = vectors * signs
    
    start = 0
    for limit in [64, 128, 256, 384]:
        width = limit - start
        H = get_hadamard_matrix(width)
        z[:, start:limit] = z[:, start:limit] @ H.T
        start = limit
        
    bits = (z >= 0.0).astype(np.uint8)
    return bits

def pack_bits(bits_array):
    n = bits_array.shape[0]
    packed = np.zeros((n, 48), dtype=np.uint8)
    for i in range(48):
        byte_slice = bits_array[:, i*8 : (i+1)*8]
        val = np.zeros(n, dtype=np.uint8)
        for b in range(8):
            val = (val << 1) | byte_slice[:, b]
        packed[:, i] = val
    return packed

def popcount_matrix(queries_packed, db_packed):
    q_u64 = queries_packed.view(np.uint64)
    db_u64 = db_packed.view(np.uint64)
    
    dists = np.zeros((q_u64.shape[0], db_u64.shape[0]), dtype=np.int32)
    for q_idx in range(q_u64.shape[0]):
        xor_result = q_u64[q_idx, :, np.newaxis] ^ db_u64.T # shape (6, N)
        popcounts = np.zeros(db_u64.shape[0], dtype=np.int32)
        for i in range(6):
            word_xor = np.ascontiguousarray(xor_result[i])
            bytes_arr = word_xor.view(np.uint8).reshape(-1, 8)
            bit_counts = np.array([bin(b).count("1") for b in range(256)], dtype=np.int32)
            popcounts += np.sum(bit_counts[bytes_arr], axis=1)
        dists[q_idx] = popcounts
    return dists

def main():
    args = parse_arguments()
    trace_data = {} if args.trace else None
    
    t_start_p1 = time.perf_counter()
    
    # 1. Load queries, labels, and database for analysis
    print("Loading queries, labels, and database vectors...")
    if not os.path.exists("temp/benchmark_data/queries.npy") or not os.path.exists("temp/benchmark_data/db_labels.npy") or not os.path.exists("temp/benchmark_data/db_vectors_subset.npy"):
        print("[Error] Files missing. Please run ingest_pipeline.py and query_generator.py.")
        sys.exit(1)
        
    t_load_start = time.perf_counter()
    queries = np.load("temp/benchmark_data/queries.npy")               # shape (278, 384)
    db_labels = np.load("temp/benchmark_data/db_labels.npy")           # shape (1000000,)
    db_vectors_subset = np.load("temp/benchmark_data/db_vectors_subset.npy") # shape (10000, 384)
    weights = np.load("temp/benchmark_data/weights.npy")               # shape (384, 384)
    
    db_labels_subset = db_labels[:10000]
    
    if trace_data is not None:
        trace_data["p1_io_loading"] = time.perf_counter() - t_load_start
        
    # Compute transformed binarized vectors in Python to analyze distance distribution
    print("Analyzing Hamming distance distribution...")
    t_dist_start = time.perf_counter()
    
    q_bits = transform_and_binarize(queries[:50], DIMENSION)
    q_packed = pack_bits(q_bits)
    
    db_bits = transform_and_binarize(db_vectors_subset, DIMENSION)
    db_packed = pack_bits(db_bits)
    
    dists = popcount_matrix(q_packed, db_packed) # shape (50, 10000)
    
    is_pit = (db_labels_subset == 1)
    
    dists_to_pits = dists[:, is_pit]
    dists_to_negatives = dists[:, ~is_pit]
    
    mean_pits = np.mean(dists_to_pits)
    std_pits = np.std(dists_to_pits)
    min_pits = np.min(dists_to_pits)
    max_pits = np.max(dists_to_pits)
    
    mean_others = np.mean(dists_to_negatives)
    std_others = np.std(dists_to_negatives)
    min_others = np.min(dists_to_negatives)
    max_others = np.max(dists_to_negatives)
    
    if trace_data is not None:
        trace_data["p1_cpu_distance_analysis"] = time.perf_counter() - t_dist_start
        
    print("\n" + "="*80)
    print("                 PITHOS HAMMING DISTANCE DISTRIBUTION REPORT            ")
    print("========================================================================")
    print(f" Target Class (Lunar Pit/Cave Entrance):")
    print(f"  - Mean Distance           : {mean_pits:.2f} bits")
    print(f"  - Std Dev                 : {std_pits:.2f} bits")
    print(f"  - Range (Min / Max)       : {min_pits} / {max_pits} bits")
    print("------------------------------------------------------------------------")
    print(f" Background Class (Flat Mondgelände/Terrain):")
    print(f"  - Mean Distance           : {mean_others:.2f} bits")
    print(f"  - Std Dev                 : {std_others:.2f} bits")
    print(f"  - Range (Min / Max)       : {min_others} / {max_others} bits")
    print("========================================================================\n")
    
    # Find the optimal threshold
    t_opt_start = time.perf_counter()
    best_f1 = 0
    best_threshold = 0
    for T in range(int(min_pits), int(max_others)):
        is_resonant_subset = (np.sum(dists <= T, axis=0) >= 5)
        tp = np.sum(is_resonant_subset & is_pit)
        fp = np.sum(is_resonant_subset & ~is_pit)
        fn = np.sum(~is_resonant_subset & is_pit)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = T
            
    if trace_data is not None:
        trace_data["p1_threshold_optimization"] = time.perf_counter() - t_opt_start
        
    print(f"Suggested optimal threshold: {best_threshold} (F1-score on subset: {best_f1*100.0:.2f}%)")
    
    duration_p1 = time.perf_counter() - t_start_p1
    
    t_start_p2 = time.perf_counter()
    
    # 2. Run the actual native verification with the suggested threshold
    print(f"\nRe-running native verification with threshold = {best_threshold}...")
    
    t_engine_start = time.perf_counter()
    engine = PithosMIDB()
    
    # Load index supplying the weights matrix for SVD energy calculation
    status = engine.load_index("lunar_real", DB_FILE, weights, DIMENSION)
    if status != 0:
        print(f"[Error] Failed to load index. Code: {status}")
        sys.exit(1)
        
    if trace_data is not None:
        trace_data["p2_native_load_index"] = time.perf_counter() - t_engine_start
        
    total_records = engine.size("lunar_real")
    voting_mask = np.zeros(total_records, dtype=np.uint8)
    
    # Update thresholds to the suggested value
    families = np.load("temp/benchmark_data/families.npy")
    thresholds = np.full(queries.shape[0], best_threshold, dtype=np.int32)
    
    t_vote_start = time.perf_counter()
    resonant_count = engine.query_planetary_grid("lunar_real", queries, families, thresholds, voting_mask)
    t_vote_sec = time.perf_counter() - t_vote_start
    t_vote_ms = t_vote_sec * 1000.0
    
    if trace_data is not None:
        trace_data["p2_native_resonant_voting"] = t_vote_sec
        
    throughput_mvps = (total_records * queries.shape[0]) / t_vote_sec / 1e6
    print(f"Scan completed in {t_vote_ms:.3f} ms ({throughput_mvps:.2f} MVPS).")
    
    # Compute full dataset metrics
    t_metrics_start = time.perf_counter()
    bit_counts = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    mask_popcounts = bit_counts[voting_mask]
    
    is_resonant = (mask_popcounts >= 5)
    is_pit = (db_labels == 1)
    
    tp = np.sum(is_resonant & is_pit)
    fp = np.sum(is_resonant & ~is_pit)
    fn = np.sum(~is_resonant & is_pit)
    tn = np.sum(~is_resonant & ~is_pit)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    if trace_data is not None:
        trace_data["p2_metrics_computation"] = time.perf_counter() - t_metrics_start
        
    print("\n" + "="*80)
    print("                 PITHOS REAL-DATA CLASSIFICATION METRICS                ")
    print("========================================================================")
    print(f" Target Class               : Lunar Pit/Cave Entrance Anchor")
    print(f" Total Database Records     : {total_records:,}")
    print(f" Actual Target Count (Pits) : {np.sum(is_pit):,}")
    print(f" Resonant Matches (Found)   : {resonant_count:,}")
    print("------------------------------------------------------------------------")
    print(f" Confusion Matrix")
    print(f"  - True Positives (TP)     : {tp:,}")
    print(f"  - False Positives (FP)    : {fp:,}")
    print(f"  - False Negatives (FN)    : {fn:,}")
    print(f"  - True Negatives (TN)     : {tn:,}")
    print("------------------------------------------------------------------------")
    print(f" Performance Metrics")
    print(f"  - Precision               : {precision * 100.0:6.2f}%")
    print(f"  - Recall                  : {recall * 100.0:6.2f}%")
    print(f"  - F1-Score                : {f1 * 100.0:6.2f}%")
    print("========================================================================\n")
    
    duration_p2 = time.perf_counter() - t_start_p2
    
    engine.close()
    
    # Save values to pithos_metrics.json
    import json
    metrics_path = "temp/benchmark_data/pithos_metrics.json"
    metrics_data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            pass
            
    metrics_data.update({
        "mu_target": float(mean_pits),
        "std_target": float(std_pits),
        "mu_other": float(mean_others),
        "std_other": float(std_others),
        "threshold": int(best_threshold),
        "best_mvps": float(throughput_mvps)
    })
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    try:
        from generate_graphics import create_distribution_plot
        create_distribution_plot()
        print("Updated Hamming distance distribution plot successfully.")
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not regenerate distribution plot ({e})")
        
    print_performance_table(duration_p1, duration_p2, trace_data)


if __name__ == "__main__":
    main()
