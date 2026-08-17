# ⚱ Pithos - Model-Isomorphic Vector Database (MIDB)

[![PyPI Version](https://img.shields.io/pypi/v/pithosdb?color=blue&label=pithosdb)](https://pypi.org/project/pithosdb/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pithosdb)](https://pypi.org/project/pithosdb/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java 25](https://img.shields.io/badge/Java-25%20(Vector%20API%20%26%20FFM)-orange)](https://openjdk.org/)
[![Native Image](https://img.shields.io/badge/GraalVM-Native%20Image-red)](https://www.graalvm.org/)

**Pithos** is a high-performance **Model-Isomorphic Database (MIDB)** and Ahead-of-Time (AOT) compiled vector search engine designed for **Matryoshka-structured binary embeddings** at planetary scale, compiled into a standalone native shared library (`.dylib` / `.so`) via **GraalVM Native Image**.

### What is a Model-Isomorphic Database (MIDB)?
Traditional vector databases treat embeddings as generic high-dimensional arrays. A **Model-Isomorphic Database (MIDB)** structurally mirrors the latent geometry and spectral energy distribution of the neural embedding model:
- **SVD-Driven Spectral Truncation:** Columnar tiers are allocated according to cumulative singular value energy (Φ(k)) derived from model projection/LoRA weights.
- **Isometric Preconditioning & Rotation:** Eliminates spatial burstiness via Rademacher sign flipping and spreads embedding energy uniformly with block-diagonal Fast Walsh-Hadamard Transforms (H_BD).
- **3-Gate Read-Path Cascade:** 1-cycle metadata filtering → Matryoshka early-exit Hamming scanning → exact FP16 in-engine reranking.
- **Zero-GC Off-Heap Memory:** Bypasses garbage collection entirely using the **Java Foreign Function & Memory (FFM) API (Project Panama)** and POSIX memory-mapped I/O (`mmap`).
- **Hardware SIMD & CUDA:** Vectorized with Java Vector API (AVX-512 / ARM NEON) and native NVIDIA CUDA kernels for batch distance computation and multi-family resonant voting.
- **Pythonic Zero-Copy FFI:** Seamless integration with NumPy arrays via `pithosdb`.

---

## Python Quickstart

Install the official Python package:

```bash
pip install pithosdb
# or with uv:
uv pip install pithosdb
```

```python
import pithosdb
import numpy as np

# 1. Open database off-heap (Zero JVM overhead)
with pithosdb.VectorDb() as db:
    # 2. Compile an index from float embeddings
    records = np.random.randn(10_000, 384).astype(np.float32)
    pithosdb.VectorDb.compile_index(
        base_path="temp/sample_index",
        records=records,
        tiers=[64, 128, 256, 384]
    )
    
    # 3. Memory-map index & run zero-copy batch k-NN search
    index = db.load_index("sample", "temp/sample_index")
    queries = np.random.randn(10, 384).astype(np.float32)
    results = index.search(queries, k=5)
    
    for q_idx, matches in enumerate(results):
        print(f"Query {q_idx} Top Matches: {matches}")

    # 4. Real-time Ingestion via LSM DeltaBuffer
    delta = db.create_delta_buffer("sample", flush_threshold=1000)
    delta.insert(record_id=42, vector=np.random.randn(384).astype(np.float32))
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
| `pithos.h` / `graal_isolate.h` | C/C++ Headers | Standalone C-ABI |

---

## System Architecture & Features

```
┌─────────────────────────────────────────────────────────────┐
│                 Client Layer (Python / C / C++)             │
│            pithosdb (ctypes Zero-Copy NumPy / FFI)          │
└──────────────────────────────┬──────────────────────────────┘
                               │ C-ABI (vdb_*)
┌──────────────────────────────▼──────────────────────────────┐
│                    Pithos Core Engine                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │   LMAX Disruptor Lock-Free Multi-Threaded Workers     │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              │                              │
│  ┌───────────────────────────▼───────────────────────────┐  │
│  │               3-Gate Read-Path Cascade                │  │
│  │  Gate 1: Metadata & Tombstone Filter (1 cycle)        │  │
│  │  Gate 2: Matryoshka Early-Exit Hamming Scan (SIMD)    │  │
│  │  Gate 3: In-Engine FP16 / Asymmetric Reranking        │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                              │
│  ┌───────────────────────────▼───────────────────────────┐  │
│  │       Project Panama Off-Heap Storage (POSIX mmap)    │  │
│  │       - <name> (64B Header)                           │  │
│  │       - <name>_tier_*.bin (Packed Columnar Bits)      │  │
│  │       - <name>_metadata.bin (Attributes & Flags)      │  │
│  │       - <name>_fp16.bin (Half-Precision Sidecar)      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Documentation

Detailed architectural guides, mathematical specifications, and C-API references:

- **[Architectural Principles & Core Innovations](docs/ARCHITECTURAL_PRINCIPLES.md):** Mathematical foundations, block-diagonal Walsh-Hadamard rotations, SVD-driven spectral truncation, and the 3-gate read-path cascade.
- **[C-API Reference & Runtime Configuration](docs/C_API_REFERENCE.md):** Complete declarations of entry points (`libpithos`), FFI mappings, CUDA wrappers, and hardware co-design guidelines (FPGA/DMA offloading).
- **[CUDA GPU Acceleration Guide](docs/cuda_integration.md):** Shared memory popcount kernels, asynchronous stream pipelines, and multi-family voting.

---

## Building from Source

### Prerequisites
- **GraalVM JDK 25** (with `native-image`)
- **Apache Maven 3.9+**
- *(Optional)* **NVIDIA CUDA Toolkit 12+** for GPU kernels

### 1. Compile Native Library (macOS & Linux)
```bash
export JAVA_HOME=/path/to/graalvm-jdk-25
export PATH=$JAVA_HOME/bin:$PATH

mvn clean package -DskipTests
```
The compiled shared library is generated in `target/pithos.dylib` (macOS) or `target/pithos.so` (Linux).

### 2. Run Test Suite
```bash
mvn test
```

### 3. Build with CUDA Support (Linux)
```bash
mvn clean package -Pcuda -Dcuda.enabled=true
```

---

## License

Licensed under the **[Apache License, Version 2.0](LICENSE)**.
