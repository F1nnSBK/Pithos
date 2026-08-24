#!/usr/bin/env python3
"""scripts/benchmark_mips_tufte.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates a Tufte-style scientific benchmark SVG plot evaluating MIPS
(Maximum Inner Product Search) speed-recall tradeoffs on anisotropic data.
"""

import os
import time
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import faiss
import pithos
from pithos import ConcentricShellIndex

def generate_anisotropic_data(n_samples, n_dim, n_clusters=100, dynamic_range=1000.0, seed=42):
    """Generates clustered (anisotropic) vectors with heavy-tailed norm distributions."""
    rng = np.random.default_rng(seed)
    
    # Generate cluster centers on the unit sphere
    centers = rng.standard_normal((n_clusters, n_dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Assign samples to clusters and add Gaussian noise
    assignments = rng.integers(0, n_clusters, size=n_samples)
    X = centers[assignments]
    noise = rng.standard_normal(X.shape).astype(np.float32) * 0.25
    X += noise
    
    # Project back to sphere
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    # Assign log-uniform magnitudes across the dynamic range
    magnitudes = np.exp(rng.uniform(0.0, np.log(dynamic_range), size=(n_samples, 1))).astype(np.float32)
    X *= magnitudes
    return X

def compute_recall_at_k(retrieved_ids, ground_truth_ids, k):
    """Computes exact Recall@K."""
    recalls = []
    for ret, gt in zip(retrieved_ids, ground_truth_ids):
        top_k_ret = set(ret[:k])
        top_k_gt = set(gt[:k])
        recalls.append(len(top_k_ret.intersection(top_k_gt)) / float(k))
    return np.mean(recalls)

def evaluate_mips_pareto():
    N = 50000
    D = 64
    Q = 200
    K_MAX = 100
    RUNS = 5
    
    print(f"Generating anisotropic dataset (N={N}, D={D}, Clusters=100)...")
    X = generate_anisotropic_data(N, D, n_clusters=100, dynamic_range=1000.0)
    
    # Generate queries (anisotropic, slightly out of distribution)
    queries = generate_anisotropic_data(Q, D, n_clusters=20, dynamic_range=10.0, seed=999)
    
    print("Computing exact ground truth (Brute-Force Dot Product)...")
    exact_dots = np.dot(queries, X.T)
    gt_topk = np.argsort(-exact_dots, axis=1)[:, :K_MAX]
    
    results = defaultdict(lambda: {"qps": [], "recall_1": [], "recall_10": [], "recall_100": []})
    
    # --- 1. FAISS IndexFlatIP (Exact Baseline) ---
    print("Benchmarking FAISS IndexFlatIP...")
    faiss_flat = faiss.IndexFlatIP(D)
    faiss_flat.add(X)
    for run in range(RUNS):
        start = time.perf_counter()
        _, I_flat = faiss_flat.search(queries, K_MAX)
        qps = Q / (time.perf_counter() - start)
        results["FAISS FlatIP"]["qps"].append(qps)
        results["FAISS FlatIP"]["recall_1"].append(compute_recall_at_k(I_flat, gt_topk, 1))
        results["FAISS FlatIP"]["recall_10"].append(compute_recall_at_k(I_flat, gt_topk, 10))
        results["FAISS FlatIP"]["recall_100"].append(compute_recall_at_k(I_flat, gt_topk, 100))
    del faiss_flat

    # --- 2. FAISS IndexIVFFlat (ANN Baseline Pareto) ---
    print("Benchmarking FAISS IndexIVFFlat (ANN Pareto)...")
    nlist = int(np.sqrt(N))
    faiss_ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_ivf.train(X)
    faiss_ivf.add(X)
    for nprobe in [1, 2, 5, 10, 20]:
        faiss_ivf.nprobe = nprobe
        name = f"FAISS IVF (np={nprobe})"
        for run in range(RUNS):
            start = time.perf_counter()
            _, I_ivf = faiss_ivf.search(queries, K_MAX)
            qps = Q / (time.perf_counter() - start)
            results[name]["qps"].append(qps)
            results[name]["recall_1"].append(compute_recall_at_k(I_ivf, gt_topk, 1))
            results[name]["recall_10"].append(compute_recall_at_k(I_ivf, gt_topk, 10))
            results[name]["recall_100"].append(compute_recall_at_k(I_ivf, gt_topk, 100))
    del faiss_ivf

    import gc
    gc.collect()

    # --- 3. Pithos Native MIPS (FP16 and FP8) ---
    from pithos import VectorDb
    for mode in ["fp16", "fp8"]:
        print(f"Benchmarking Pithos Native MIPS ({mode.upper()})...")
        with tempfile.NamedTemporaryFile(suffix=".pithos", delete=False) as tmp:
            path = tmp.name
        VectorDb.compile_container(path, records=X, metric="mips", sidecar_mode=mode)
        
        db = VectorDb()
        idx = db.load_index("default", path)
        
        # Warmup
        idx.search(queries, k=K_MAX, return_numpy=True)
        
        name = f"Pithos MIPS ({mode.upper()})"
        for run in range(RUNS):
            start = time.perf_counter()
            I_pithos, _ = idx.search(queries, k=K_MAX, return_numpy=True)
            qps = Q / (time.perf_counter() - start)
            results[name]["qps"].append(qps)
            results[name]["recall_1"].append(compute_recall_at_k(I_pithos, gt_topk, 1))
            results[name]["recall_10"].append(compute_recall_at_k(I_pithos, gt_topk, 10))
            results[name]["recall_100"].append(compute_recall_at_k(I_pithos, gt_topk, 100))
        
        db.close()
        os.remove(path)
        gc.collect()

    return results

def plot_tufte_pareto(results, output_path: str):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'text.color': '#222222',
        'axes.labelcolor': '#222222',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'figure.autolayout': True,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), dpi=300)
    metrics = [("recall_1", "Recall@1 (%)"), ("recall_10", "Recall@10 (%)"), ("recall_100", "Recall@100 (%)")]

    # Group colors
    colors = {
        "FAISS FlatIP": "#7f7f7f",      # Neutral gray
        "FAISS IVF": "#1f77b4",         # Deep blue
        "Pithos MIPS (FP16)": "#2ca02c", # Forest green
        "Pithos MIPS (FP8)": "#ff7f0e",  # Safety orange
    }

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        ax.set_xlabel('Queries Per Second (QPS)')
        ax.set_ylabel(metric_label)
        ax.set_xscale('log')
        ax.set_ylim(-2, 105)

        # Plot FAISS IVF Pareto curve
        ivf_keys = sorted([k for k in results.keys() if "FAISS IVF" in k], key=lambda x: np.mean(results[x]["qps"]))
        if ivf_keys:
            ivf_qps = [np.mean(results[k]["qps"]) for k in ivf_keys]
            ivf_rec = [np.mean(results[k][metric_key]) * 100.0 for k in ivf_keys]
            ax.plot(ivf_qps, ivf_rec, color=colors["FAISS IVF"], linestyle='--', alpha=0.6, zorder=1)

        # Plot points with error bars (std dev)
        for name, data in results.items():
            base_name = "FAISS IVF" if "FAISS IVF" in name else name
            c = colors.get(base_name, "#000000")
            marker = 'o' if "Pithos" in name else 's'
            
            mean_qps = np.mean(data["qps"])
            std_qps = np.std(data["qps"])
            mean_rec = np.mean(data[metric_key]) * 100.0
            std_rec = np.std(data[metric_key]) * 100.0

            # Only label the first IVF point to avoid legend clutter
            label = base_name if ("FAISS IVF" not in name or name == ivf_keys[0]) else None
            
            ax.errorbar(
                mean_qps, mean_rec, 
                xerr=std_qps, yerr=std_rec,
                fmt=marker, color=c, ecolor=c, elinewidth=1.0, 
                capsize=2, markersize=5, label=label, zorder=2
            )

        if ax == axes[0]:
            ax.legend(frameon=False, loc='lower right', fontsize=9)

    fig.suptitle('Speed-Recall Pareto Frontier: Pithos MIPS vs FAISS (Anisotropic N=100K, D=64)', fontsize=13, color='#111111')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='svg')
    plt.close()
    print(f"Scientific Tufte MIPS benchmark written to: {output_path}")

if __name__ == '__main__':
    results = evaluate_mips_pareto()
    out_svg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "assets", "mips_recall_benchmark.svg")
    plot_tufte_pareto(results, out_svg)
