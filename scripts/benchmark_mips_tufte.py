#!/usr/bin/env python3
"""scripts/benchmark_mips_tufte.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates a Tufte-style scientific benchmark SVG plot evaluating MIPS
(Maximum Inner Product Search) recall across varying vector norm variances.
"""

import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pithos
from pithos import SphericalLiftingTransformer, MipsIndex, ConcentricShellIndex, VectorDb, SidecarMode


def evaluate_mips_recall():
    rng = np.random.default_rng(42)
    N = 1000
    D = 64
    Q = 50
    k = 10

    norm_ratios = [1, 5, 25, 100, 500, 1000]
    
    recalls_lifting_fp16 = []
    recalls_lifting_fp8 = []
    recalls_concentric = []
    recalls_naive_cosine = []

    for ratio in norm_ratios:
        # Generate unnormalized vectors with specified dynamic range
        raw_dirs = rng.standard_normal((N, D)).astype(np.float32)
        raw_dirs /= np.linalg.norm(raw_dirs, axis=1, keepdims=True)
        magnitudes = np.exp(rng.uniform(0.0, np.log(float(ratio)), size=(N, 1))).astype(np.float32)
        X = raw_dirs * magnitudes

        # Generate unnormalized queries
        q_dirs = rng.standard_normal((Q, D)).astype(np.float32)
        q_dirs /= np.linalg.norm(q_dirs, axis=1, keepdims=True)
        q_mags = rng.uniform(1.0, 10.0, size=(Q, 1)).astype(np.float32)
        queries = q_dirs * q_mags

        # Ground truth brute-force Top-K
        exact_dots = np.dot(queries, X.T)
        gt_topk = np.argsort(-exact_dots, axis=1)[:, :k]

        # 1. Spherical Lifting + FP16
        with tempfile.NamedTemporaryFile(suffix=".pithos", delete=False) as tmp:
            p16_path = tmp.name
        idx_fp16 = MipsIndex.from_vectors(X, path=p16_path, sidecar_mode="fp16", pad_to_multiple=64)
        res_fp16 = idx_fp16.search(queries, k=k, return_numpy=True)
        ids_fp16, _ = res_fp16
        r_fp16 = np.mean([len(set(ids_fp16[q]).intersection(set(gt_topk[q]))) / float(k) for q in range(Q)])
        recalls_lifting_fp16.append(r_fp16 * 100.0)
        os.remove(p16_path)

        # 2. Spherical Lifting + FP8
        with tempfile.NamedTemporaryFile(suffix=".pithos", delete=False) as tmp:
            p8_path = tmp.name
        idx_fp8 = MipsIndex.from_vectors(X, path=p8_path, sidecar_mode="fp8", pad_to_multiple=64)
        res_fp8 = idx_fp8.search(queries, k=k, return_numpy=True)
        ids_fp8, _ = res_fp8
        r_fp8 = np.mean([len(set(ids_fp8[q]).intersection(set(gt_topk[q]))) / float(k) for q in range(Q)])
        recalls_lifting_fp8.append(r_fp8 * 100.0)
        os.remove(p8_path)

        # 3. Concentric Shell Index (4 shells)
        shell_dir = tempfile.mkdtemp(prefix="pithos_shell_bench_")
        c_shell = ConcentricShellIndex.from_vectors(X, base_dir=shell_dir, num_shells=4, sidecar_mode="fp16", pad_to_multiple=64)
        res_shell = c_shell.search(queries, k=k)
        ids_shell = np.array([[r.id for r in q_list] for q_list in res_shell])
        r_shell = np.mean([len(set(ids_shell[q]).intersection(set(gt_topk[q]))) / float(k) for q in range(Q)])
        recalls_concentric.append(r_shell * 100.0)
        shutil.rmtree(shell_dir)

        # 4. Naive Cosine Search (baseline without MIPS lifting)
        with tempfile.NamedTemporaryFile(suffix=".pithos", delete=False) as tmp:
            p_naive = tmp.name
        VectorDb.compile_container(p_naive, records=X, tiers=[D], metric="cosine", sidecar_mode=SidecarMode.FP16)
        db = VectorDb()
        naive_idx = db.load_index(f"naive_{ratio}", p_naive)
        ranked_naive, _ = naive_idx.rerank(queries, k=k, metric="cosine")
        r_naive = np.mean([len(set(ranked_naive[q]).intersection(set(gt_topk[q]))) / float(k) for q in range(Q)])
        recalls_naive_cosine.append(r_naive * 100.0)
        os.remove(p_naive)

    return norm_ratios, recalls_lifting_fp16, recalls_lifting_fp8, recalls_concentric, recalls_naive_cosine


def plot_tufte_mips_recall(ratios, r_fp16, r_fp8, r_shell, r_naive, output_path: str):
    # Set up Tufte minimalist styling
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

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)

    # Color palette
    c_fp16 = '#1f77b4'       # Primary deep blue
    c_fp8 = '#2ca02c'        # Forest green
    c_shell = '#9467bd'      # Muted purple
    c_naive = '#d62728'      # Subdued red

    x_indices = np.arange(len(ratios))
    x_labels = [f'{r}x' for r in ratios]

    # Plot lines with clean markers
    ax.plot(x_indices, r_fp16, marker='o', markersize=5, color=c_fp16, linewidth=1.8, label='Spherical Lifting (FP16 Sidecar)')
    ax.plot(x_indices, r_shell, marker='s', markersize=5, color=c_shell, linewidth=1.6, linestyle='--', label='Concentric Shell Partitioning (4 Shells)')
    ax.plot(x_indices, r_fp8, marker='^', markersize=5, color=c_fp8, linewidth=1.5, linestyle='-.', label='Spherical Lifting (FP8 Sidecar)')
    ax.plot(x_indices, r_naive, marker='x', markersize=6, color=c_naive, linewidth=1.4, linestyle=':', label='Naive Cosine Similarity (No MIPS Lifting)')

    # Tufte layout adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 6))
    ax.spines['bottom'].set_position(('outward', 6))

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Vector Magnitude Dynamic Range (Max Norm / Min Norm)')
    ax.set_ylabel('Top-10 Recall (%)')
    ax.set_ylim(-2, 105)

    ax.set_title('Pithos MIPS Recall vs Vector Norm Variance', pad=14, fontsize=12, loc='left', color='#111111')
    ax.legend(frameon=False, loc='lower left', fontsize=9.5)

    # Save as pure SVG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='svg')
    plt.close()
    print(f"Tufte MIPS SVG plot successfully written to: {output_path}")


if __name__ == '__main__':
    ratios, r_fp16, r_fp8, r_shell, r_naive = evaluate_mips_recall()
    out_svg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "assets", "mips_recall_benchmark.svg")
    plot_tufte_mips_recall(ratios, r_fp16, r_fp8, r_shell, r_naive, out_svg)
