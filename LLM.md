# Pithos Vector Search Engine - LLM Context

This document is designed to provide Large Language Models (LLMs) with a comprehensive overview of the Pithos Vector Search Engine repository.

## 1. Project Overview

Pithos is a high-performance, Ahead-of-Time (AOT) compiled, dimension-agnostic vector search engine.
- Language: Java 25 (core logic), C/C++ (CUDA kernels, JNI), Python (wrappers).
- Focus: Matryoshka-structured binary embeddings at planetary scale.
- Output: Compiled into a native shared library (.dylib / .so) via GraalVM Native Image.

### Core Innovations
- Bypass Garbage Collection: Uses Java Foreign Function & Memory (FFM) API (Project Panama) and POSIX-aligned virtual memory mapping (mmap) to map memory-bandwidth-bound datasets off-heap.
- Dimensionality Reduction: Employs block-diagonal Walsh-Hadamard rotations and SVD-driven spectral truncation.
- Search Pipeline: 3-gate read-path cascade (Gate 1: Base Tier, Gate 2: Matryoshka Tiers, Gate 3: Resonant Voting / FP16 Reranking).
- Acceleration: CUDA acceleration for GPU Hamming distance computation and multi-family voting.

## 2. Directory Structure

- /src/main/java/org/pithos: Core Java source code (VectorDb, FlatIndex, CApi, DeltaBuffer).
- /docs/: Documentation resources (ARCHITECTURAL_PRINCIPLES.md, C_API_REFERENCE.md, MATH_THEORY.md, CUDA_INTEGRATION.md, NEXT_STEPS.md).
- /benchmarks/: Python scripts for performance testing (verify_wal.py, verify_compaction.py, verify_optional_fp16.py).
- /examples/: Integration demos (C and Java FFM).
- /graal_isolate.h & pithos.h: Native headers for the compiled library.
- /pom.xml: Maven configuration.

## 3. Key Concepts for LLMs

### Matryoshka Representation Learning (MRL)
Pithos leverages MRL to truncate high-dimensional vectors into smaller, cascading tiers (e.g., 64-dim, 128-dim, 384-dim). The search algorithm dynamically cascades through these tiers, pruning unlikely candidates early.

### Resonant Voting
Instead of computing full distances for every query, Pithos groups queries into families and uses threshold-based Hamming distance voting to identify resonant nodes across the dataset.

### Write-Ahead Log (WAL) & Compaction
Pithos supports real-time inserts/deletes using an LSM Delta-Buffer architecture. Delta buffer state is backed by a crash-resilient WAL. Multiple index segments can be zero-copy compacted into larger segments via stream-level transfers.

### Optional FP16 Sidecar
To optimize disk footprint, Pithos allows compiling indexes without the FP16 sidecar (which usually stores raw vectors for exact reranking). This provides an 84% reduction in disk size at the cost of some Recall@10 accuracy, while Resonant Voting remains completely unaffected.

## 4. Building and Execution

- Native Compilation: Uses GraalVM. Command: `mvn clean package`.
- Python API: The Python wrapper accesses the native library via ctypes (see `benchmark.py`).
- Reproducibility: `reproduce_all.sh` runs the entire verification suite.

## 5. Development Guidelines

- No Emojis: Markdown files must use standard ASCII characters only. No emojis are permitted in documentation.
- CAPS filenames: Core markdown files (e.g., README.md, LLM.md, NEXT_STEPS.md) must use UPPERCASE filenames to be automatically recognized by GitHub and agents.
- Zero-Cost Abstractions: When modifying core Java logic, prioritize FFM (MemorySegment) and avoid on-heap allocations during the critical search path.
