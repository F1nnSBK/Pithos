# Pithos Vector Search Engine

*(Note: This repository was formerly known as `lcvk`)*

A high-performance, Ahead-of-Time (AOT) compiled, dimension-agnostic vector search engine written in **Java 25**, optimized for **Matryoshka-structured binary embeddings** at planetary scale, and compiled into a native shared library (`.dylib` / `.so`) via **GraalVM Native Image**.

Pithos achieves its speed by collapsing abstraction boundaries between language runtimes, the operating system, and hardware execution models. It bypasses garbage collection entirely, mapping memory-bandwidth-bound datasets off-heap using the Java Foreign Function & Memory (FFM) API (Project Panama) and POSIX-aligned virtual memory mapping (`mmap`).

**Now with CUDA acceleration support** for GPU-accelerated Hamming distance computation and multi-family voting, enabling massive parallel search operations on NVIDIA GPUs.

---

##  Documentation Directory

To make the codebase easier to navigate, detailed guides and theory have been split into standalone documents:

- **[Architectural Principles & Core Innovations](docs/ARCHITECTURAL_PRINCIPLES.md):** Mathematical foundations, block-diagonal Walsh-Hadamard rotations, SVD-driven spectral truncation, and the 3-gate read-path cascade.
- **[C-API Reference & Runtime Configuration](docs/C_API_REFERENCE.md):** Complete declarations of entry points (`libpithos`), FFI mappings, CUDA wrappers, and hardware co-design guidelines (FPGA/DMA offloading).

---

##  Directory Structure

```
.
├── pom.xml                 # Maven configuration (dimension-agnostic pithos packaging, CUDA profile)
├── README.md               # This file
├── test_client.c           # C verification client calling Pithos float C-API
├── pithos.h                # C API header file
├── graal_isolate.h         # GraalVM Native Image header
├── docs/                   # Documentation resources
│   ├── ARCHITECTURAL_PRINCIPLES.md # Math, theory, and system architecture
│   ├── C_API_REFERENCE.md          # C-API declarations and tuning guidelines
│   └── archive/                    # Archived log history
├── benchmarks/             # Verification scripts
│   ├── run_real_verification.py    # Lunar Pit / adapter classification pipeline
│   ├── verify_compaction.py        # Index compaction verification script
│   ├── verify_wal.py               # Write-Ahead Log verification script
│   └── verify_optional_fp16.py     # FP16 vs. Non-FP16 verification script
├── examples/               # Developer integration demos
│   ├── cpp/demo.c                  # C integration demo linking libpithos
│   └── java/ZeroCostDemo.java      # FFM Panama off-heap GC bypass demo
└── src/                    # Core source tree (Java backend, CUDA kernels, JNI bindings)
```

---

## 📦 Python Quickstart

Pithos is installable via `pip` or `uv`:

```bash
pip install pithosdb
```

```python
import pithosdb
import numpy as np

# Initialize database off-heap
with pithosdb.VectorDb() as db:
    index = db.load_index("lunar", "path/to/lunar_index")
    
    # Zero-copy batch search with NumPy
    queries = np.random.randn(10, 384).astype(np.float32)
    results = index.search(queries, k=5)
    
    for q_idx, matches in enumerate(results):
        print(f"Query {q_idx} Top Matches: {matches}")
```

---

## ⚡ Precompiled Native Libraries

Precompiled native libraries are automatically published as GitHub Release assets:

🔗 [Download Latest Release Assets](https://github.com/F1nnSBK/Pithos/releases/latest)

Each release includes:
- `libpithos-linux-x86_64.so` — Linux (x86_64)
- `libpithos-macos-aarch64.dylib` — macOS (Apple Silicon)
- `libpithos-linux-x86_64-cuda.so` — Linux (x86_64) with CUDA support
- `pithos.h` — C API header
- `graal_isolate.h` — GraalVM Native Image header

---

## 🛠️ Building from Source

### 1. Compile & Build (Native macOS & Linux)
Ensure you have **GraalVM JDK 25** and **Maven** installed:
```bash
export JAVA_HOME=/path/to/graalvm-jdk-25
export PATH=$JAVA_HOME/bin:$PATH
mvn clean package
```
This executes all unit tests (including SVD, FWHT, compaction, and WAL recovery) and compiles `libpithos.dylib` / `libpithos.so` inside `target/`.

### 2. Verification
```bash
python benchmarks/run_real_verification.py
```

### 3. Building with CUDA Support (Linux)
```bash
export JAVA_HOME=/path/to/graalvm-jdk-25
export PATH=$JAVA_HOME/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
mvn clean package -Pcuda -Dcuda.enabled=true
```


---

## ️ Roadmap & Next Steps

### 1. Distribute Search Topologies
- **Objective**: Scale out to multi-node clusters.
- **Concept**: Add consistent hashing rings to shard the Matryoshka columnar indexes across multiple nodes, executing query routing and remote merging in parallel.

### 2. Dynamic Memory Re-alignment
- **Objective**: Avoid restart overhead during delta-buffer flushes.
- **Concept**: Implement dynamic pointer rotation in `vdb_load_index` to hot-swap mapped memory regions on the fly without closing active isolate threads.
