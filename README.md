# Pithos - Model-Isomorphic Vector Database (MIDB)

[![PyPI Version](https://img.shields.io/pypi/v/pithosdb?color=blue&label=pithosdb)](https://pypi.org/project/pithosdb/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pithosdb)](https://pypi.org/project/pithosdb/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java 25](https://img.shields.io/badge/Java-25%20(Vector%20API%20%26%20FFM)-orange)](https://openjdk.org/)
[![Native Image](https://img.shields.io/badge/GraalVM-Native%20Image-red)](https://www.graalvm.org/)

**Pithos** is a high-performance **Model-Isomorphic Database (MIDB)** and Ahead-of-Time (AOT) compiled vector search engine designed for **Matryoshka-structured binary embeddings**, compiled into a standalone native shared library (`.dylib` / `.so`) via **GraalVM Native Image**.

### What is a Model-Isomorphic Database (MIDB)?
Traditional vector databases treat embeddings as generic high-dimensional arrays. A **Model-Isomorphic Database (MIDB)** structurally mirrors the latent geometry and spectral energy distribution of the neural embedding model:
- **Row-Energy Prefix Pruning (Φ(k)):** Cumulative row L2 energy (`||W_{i,:}||_2^2`) derived from projection/LoRA weights (`TransformOperator.computeCumulativeEnergy`), preserving Matryoshka coordinate alignment without SVD axis permutation.
- **Isometric Preconditioning & Rotation:** Eliminates spatial burstiness via Rademacher sign flipping and spreads embedding energy uniformly using block-diagonal Fast Walsh-Hadamard Transforms (`H_BD`) for power-of-two blocks, Kronecker-factorized orthogonal rotations (`H_u ⊗ Ω_v` with DCT-II) for arbitrary non-power-of-two dimensions, and exact sub-64-bit tier bit-masking.
- **4-Gate Read-Path Cascade:** Gate 0 (`4x8-Bit` MIH CSR prefix routing) → Gate 1 (64-bit liveliness & tombstone metadata bitmask) → Gate 2 (Matryoshka tiered SIMD Hamming scan via Java Vector API AVX-512 / ARM NEON) → Gate 3 (Asymmetric Distance Computation with **FP8 E4M3**, **NVFP4 E2M1**, **FP16**, or rotated binary/ternary L2 fallback + monotonic early cutoff).
- **Zero-GC Off-Heap Memory:** Bypasses garbage collection entirely using the **Java Foreign Function & Memory (FFM) API (Project Panama)** and POSIX memory-mapped I/O (`mmap`).
- **Hardware SIMD & CUDA:** Vectorized with Java Vector API (`jdk.incubator.vector`) and optional native NVIDIA CUDA kernels for batch distance computation, Walsh-Hadamard transforms, FP8 reranking, and multi-family resonant voting.
- **Pythonic Zero-Copy FFI:** Seamless integration with NumPy arrays via `pithosdb` (or `pithos`).

---

## Python Quickstart

Install the official Python package:

```bash
pip install pithosdb
# or with uv:
uv pip install pithosdb
```

```python
import numpy as np
from pithosdb import VectorDb, SidecarMode, QuantizationMode

# 1. Generate normalized embeddings
rng = np.random.default_rng(42)
records = rng.standard_normal((10_000, 384)).astype(np.float32)
records /= np.linalg.norm(records, axis=1, keepdims=True)

# 2. Compile into a universal self-contained .pithos container (FP8 E4M3 + 4x8 MIH)
VectorDb.compile_container(
    path="temp/sample_index.pithos",
    records=records,
    tiers=[64, 128, 256, 384],
    q_mode=QuantizationMode.ONE_BIT,
    sidecar_mode=SidecarMode.FP8,
)

# 3. Memory-map off-heap & run zero-copy k-NN search + real-time LSM writes
with VectorDb() as db:
    index = db.load_index("sample", "temp/sample_index.pithos")
    queries = records[:10]

    # Object search or zero-copy flat NumPy FFI views
    results = index.search(queries, k=5)
    out_ids, out_dists = index.search_numpy(queries[0], k=5)
    print(f"Top-5 IDs: {out_ids}, Scaled Distances: {out_dists}")

    # Real-time ingestion & tombstone-aware merged search via LSM DeltaBuffer + WAL
    delta = db.create_delta_buffer("sample", flush_threshold=1000)
    delta.insert(record_id=42, vector=records[0])
    merged_hits = index.search_merged(queries[0], k=5)
    print(f"Merged Top Matches: {merged_hits}")
```

---

## Precompiled Native Binaries

Precompiled native libraries are automatically published on GitHub Releases:

[Download Latest Release Assets](https://github.com/F1nnSBK/Pithos/releases/latest)

| Artifact | Platform | Acceleration |
| :--- | :--- | :--- |
| `libpithos-macos-aarch64.dylib` | macOS (Apple Silicon / ARM64) | NEON SIMD |
| `libpithos-linux-x86_64.so` | Linux (x86_64) | AVX2 / AVX-512 |
| `libpithos-linux-aarch64.so` | Linux (ARM64 / Graviton) | NEON SIMD |
| `libpithos-linux-cuda-x86_64.so` | Linux (x86_64) | NVIDIA CUDA GPU |
| `libpithos-linux-cuda-aarch64.so` | Linux (ARM64 / Grace) | NVIDIA CUDA GPU |
| `pithos.h` / `graal_isolate.h` | C/C++ Headers | Standalone C-ABI |

---

## System Architecture & Features

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Client Layer (Python / C / C++)                      │
│               pithosdb (ctypes Zero-Copy NumPy / C-FFI)                 │
└───────────────────┬─────────────────────────────────┬───────────────────┘
                    │ C-ABI Read Path (vdb_*)         │ Write Path (vdb_insert / WAL)
┌───────────────────▼────────────────────┐  ┌─────────▼───────────────────┐
│   Parallel Read Workers (IntStream)    │  │  LSM DeltaBuffer + _wal.bin │
│  ┌──────────────────────────────────┐  │  │  LMAX Disruptor 4.0.0       │
│  │  4-Gate Read-Path Cascade        │  │  │  Lock-Free Ring Buffer +    │
│  │  Gate 0: 4x8-Bit MIH CSR Routing │  │  │  Shadowed IDs & Tombstones  │
│  │  Gate 1: 64-Bit Liveliness Mask  │  │  └─────────┬───────────────────┘
│  │  Gate 2: Tiered SIMD Hamming Scan│  │            │
│  │  Gate 3: FP8 / NVFP4 / FP16 / L2 │  │            │ Unified
│  │          Reranking (Early Cutoff)│  │            │ search_merged
│  └────────────────┬─────────────────┘  │            │
└───────────────────┼────────────────────┘            │
                    ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             Project Panama Off-Heap Storage (POSIX mmap)                │
│  • Single-File Container (.pithos, DIOGENES v2):                        │
│    Superblock (128 B), IDs, Matryoshka Tiers, Sidecar (FP8/NVFP4/FP16), │
│    4x8 MIH CSR Prefix Table, Metadata (Arrow IPC / JSONL), TOC, Trailer │
│  • Or Multi-File Layout (<name>, <name>_tier_*.bin, <name>_fp{16,8,4})  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Documentation

Detailed architectural guides, mathematical specifications, and C-API references:

- **[Universal Single-File Container (.pithos)](docs/container_format.md):** Binary specification of the schema-agnostic `DIOGENES` v2 container format, 4x8 MIH CSR tables, and Arrow IPC metadata.
- **[Architectural Principles & Core Innovations](docs/architecture.md):** Mathematical foundations, Kronecker/Walsh-Hadamard rotations, row-energy prefix pruning, and the 4-gate cascaded read-path.
- **[Mathematical Foundations](docs/math_theory.md):** Isometric transforms, Kronecker composite-dimension proofs, sub-Gaussian energy flattening, MIH collision bounds, and Gate 3 early cutoff.
- **[C-API Reference & Runtime Configuration](docs/c_api_reference.md):** Complete C-ABI entry points (`pithos.h`), FFI mappings, sidecar configuration, and FPGA/DMA co-design interfaces.
- **[CUDA GPU Acceleration Guide](docs/cuda_integration.md):** CUDA kernels (`pithos_kernels.h`), host-device wrappers (`pithos_cuda.h`), and Docker build setup.
- **[Release Notes](docs/release_notes.md):** Detailed changelog through Pithos v2.2.6.

---

## Building from Source

### Prerequisites
- **GraalVM JDK 25** (with `native-image`)
- **Apache Maven 3.9+**
- **Python 3.9+** with `numpy` (and `pytest` for tests)
- *(Optional)* **NVIDIA CUDA Toolkit 12+** for GPU kernels

### 1. Compile Native Library (macOS & Linux)
```bash
export JAVA_HOME=/path/to/graalvm-jdk-25
export PATH=$JAVA_HOME/bin:$PATH

mvn clean package -DskipTests
```
The compiled shared library is generated in `target/pithos.dylib` (macOS) or `target/pithos.so` (Linux).

### 2. Run Test Suites
```bash
mvn clean test
pytest -v
```

### 3. Build with CUDA Support (Linux)
```bash
mvn clean package -Pcuda -Dcuda.enabled=true
```

---

## License

Licensed under the **[Apache License, Version 2.0](LICENSE)**.
