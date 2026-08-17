# ⚱ Pithos — Model-Isomorphic Vector Database

<p align="center">
  <strong>Ultra-low latency, model-isomorphic vector database engine for planetary-scale datasets.</strong>
</p>

<p align="center">
  <a href="https://github.com/F1nnSBK/Pithos/releases/latest"><img src="https://img.shields.io/github/v/release/F1nnSBK/Pithos?style=flat-square&color=blue" alt="Latest Release"></a>
  <a href="https://github.com/F1nnSBK/Pithos/actions/workflows/build-binaries.yml"><img src="https://img.shields.io/github/actions/workflow/status/F1nnSBK/Pithos/build-binaries.yml?style=flat-square&label=CI%2FCD" alt="Build Status"></a>
  <a href="https://pypi.org/project/pithosdb/"><img src="https://img.shields.io/pypi/v/pithosdb?style=flat-square&color=green" alt="PyPI Version"></a>
  <a href="https://github.com/F1nnSBK/Pithos/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-orange?style=flat-square" alt="License"></a>
</p>

---

## What is Pithos?

**Pithos** is a **Model-Isomorphic Vector Database (MIDB)** built from the ground up to bypass traditional garbage collection overheads and runtime indirection. Instead of treating vectors as generic high-dimensional points, Pithos physically aligns its storage format with the embedding model's latent geometry:

![Pithos 5-Lever Cascade Architecture](assets/pithos_cascade_architecture.svg)

---

## Key Features

=== "Hardware Co-Design"
    - **Blackwell FP8 / NVFP4 Sidecar Engine:** Native E4M3 (1 B/dim) and NVFP4 (0.56 B/dim) sidecars for in-engine candidate reranking.
    - **AVX-512 & ARM Neon Acceleration:** Hardware-accelerated bitwise Hamming distances using vectorized `VPOPCNTDQ` and ARM Neon intrinsics.
    - **NVIDIA GPU Acceleration:** Direct CUDA kernel dispatch for batch Hamming distance, multi-family voting, and Fast Walsh-Hadamard Transforms.

=== "Model-Isomorphic Storage"
    - **Off-Heap Virtual Memory:** Direct POSIX-aligned columnar mapping via Java Panama FFM (Foreign Function & Memory API) and C-API shared memory.
    - **Matryoshka Spectral Decomposition:** Energy-budgeted tiered binary indexing that prunes up to 99% of search space in Gate 1.
    - **LSM-Tree Delta Buffer:** Real-time lock-free insertions and tombstone soft-deletes with zero-cost snapshots.

=== "Asymmetric Search (ADC)"
    - **Continuous FP32 Fidelity:** Queries are evaluated in 100% continuous 32-bit floating point precision against compressed database records.
    - **Precomputed Query LUTs (Hebel 1):** Zero floating-point multiplication during Gate 3 candidate reranking.
    - **100% Recall@1 on Lunar DINOv3 Benchmarks.**

---

## Quickstart

### Python Installation & Usage

```bash
pip install pithosdb numpy
```

```python
import numpy as np
import pithos

# 1. Initialize Pithos Model-Isomorphic Database singleton
db = pithos.PithosMIDB()

# 2. Compile an index from float vectors with FP8 sidecar
dim = 384
num_vectors = 50_000
vectors = np.random.randn(num_vectors, dim).astype(np.float32)
# Normalize to unit hypersphere
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

db.compile_index(
    output_path="dataset_index",
    vectors=vectors,
    tiers=[64, 128, 256, 384],
    sidecar_mode=pithos.SidecarMode.FP8
)

# 3. Load memory-mapped index
index = db.load_index("dataset_index")

# 4. Perform Asymmetric Top-K Nearest Neighbor Search
query = np.random.randn(1, dim).astype(np.float32)
query /= np.linalg.norm(query)

result_ids, result_dists = index.batch_search(query, k=10)
print("Nearest Neighbor IDs:", result_ids[0])
print("Euclidean Distances:", result_dists[0])
```

### C / C++ SDK Integration

```c
#include <stdio.h>
#include "pithos.h"

int main() {
    pithos_isolate_t* isolate = NULL;
    if (pithos_create_isolate(NULL, &isolate) != 0) {
        fprintf(stderr, "Failed to initialize GraalVM isolate\n");
        return 1;
    }

    // Load memory-mapped index
    pithos_index_t* index = pithos_load_index(isolate, "dataset_index");
    if (!index) {
        fprintf(stderr, "Failed to map index\n");
        return 1;
    }

    float query[384];
    // Fill query...
    long result_ids[10];
    int result_distances[10];

    int count = pithos_search(isolate, index, query, 10, result_ids, result_distances);
    printf("Found %d neighbors. Top ID: %ld\n", count, result_ids[0]);

    pithos_close_index(isolate, index);
    pithos_tear_down_isolate(isolate);
    return 0;
}
```

---

## Benchmark Summary (Lunar DINOv3 Dataset, D=384)

| Index Mode | Storage (B/dim) | 2.72B Atlas Size | Recall@1 | Recall@10 | Search Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 Sidecar** | 2.00 B/dim | 2.23 TB | 100.00% | 94.40% | 196.1 µs |
| **FP8 Sidecar (E4M3)** | **1.00 B/dim** | **1.19 TB (-44%)** | **100.00%** | **94.40%** | **185.8 µs** |
| **NVFP4 Sidecar (E2M1)** | **0.56 B/dim** | **668 GB (-65%)** | **96.80%** | **89.20%** | **172.1 µs** |
| **Bit-Only (No Sidecar)**| 0.125 B/dim | 165 GB (-92%) | 88.40% | 78.10% | 142.3 µs |

---

## Documentation Sections

- [**Architectural Principles**](architecture.md): Deep dive into off-heap virtual memory, memory layouts, and LMAX Disruptor parallelism.
- [**CUDA GPU Acceleration**](cuda_integration.md): Architecture of CUDA kernels, unified host-device DMA, and multi-stream execution.
- [**C-API Reference**](c_api_reference.md): Complete specification of C/C++ bindings, structs, and FFI interoperability.
- [**Mathematical Foundations**](math_theory.md): SVD spectral energy decay, Sylvester-Hadamard isometric rotations, and spherical pruning.
- [**Release Notes**](release_notes.md): Detailed changelog for Pithos v1.1.0 and previous releases.
- [**Roadmap & Next Steps**](next_steps.md): FPGA co-design, distributed clustering, and heterogeneous execution.
