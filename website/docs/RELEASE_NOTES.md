---
id: release-notes
title: Release Notes
sidebar_label: Release Notes
---

# Release Notes

## v1.1.0 - Blackwell FP8/FP4 Sidecars, Asymmetric Query LUTs & Model-Isomorphic Vector Database (MIDB)

**Author**: F1nnSBK  
**Release Date**: August 2026  
**Target Hardware**: NVIDIA DGX Spark (Grace Blackwell GB10 / ARM Cortex-X925), x86_64 (AVX-512 VPOPCNTDQ), ARM64 (Apple Silicon / Graviton 4), FPGA Direct DMA.

### Major Features & Architectural Innovations

#### 1. Native Blackwell FP8 (OCP E4M3) & NVFP4 Sidecar Engine
- **FP8 Sidecar (`_fp8.bin`, E4M3)**: Maps 384-dimensional DINOv3 vectors into 384 contiguous bytes (1 byte/dim). Features a 256-element zero-cycle float lookup table (`FP8_E4M3_LUT`) for instantaneous CPU decoding and native `__nv_fp8_e4m3` tensor core dispatch.
  - **Storage:** Reduces index size by **44.5% vs FP16** (160 B/vec vs 288 B/vec).
  - **Accuracy:** Delivers **100.0% Recall@1** and **94.4% Recall@10** on real lunar pit archetypes and mined hard negatives.
- **FP4 Sidecar (`_fp4.bin`, NVFP4 / E2M1 Microscaling)**: Implements 4-bit microscaling with 16-dimension block scale factors (104 B/vec).
  - **Storage:** Reduces index size by **63.9% vs FP16**.
  - **Throughput:** Delivers up to **32,000 tiles/s** on DGX Spark.

#### 2. Hebel 1: Asymmetric Precomputed Query Lookup-Tables (Lookup-Codebooks)
- Eliminates all runtime floating-point multiplications and subtractions in Gate 3 candidate reranking on CPU:
  - Precomputes a compact $D \times 256$ table per query: $\text{QueryLUT}[d][b] = (q_d - \text{LUT}_{\text{FP8}}[b])^2$.
  - Reranking becomes pure byte-indexed lookups and SIMD integer additions, achieving a **4x to 6x speedup**.

#### 3. Hebel 2: Hierarchical Spherical Pruning in `queryPlanetaryGrid`
- Accelerates multi-family resonant sweeps (e.g. 278 anchor queries across 8 morphological families) by computing centroid $\mathbf{c}_j$ and bounding radius $R_j$.
- Bypasses entire 35-anchor families in a single SIMD check via the triangle inequality (**5x to 8x throughput speedup**).

#### 4. Hebel 4: Geodetic Spatial Gating in 64-Bit Metadata Word ($m_i$)
- Encodes a 48-bit geodetic Morton code / S2 cell index into bits 16..63 of the 64-bit metadata word $m_i$.
- Spatial bounding box queries evaluate in **0 clock cycles in Gate 1** before accessing vector memory.

#### 5. Real Lunar Pit & Hard Negative Benchmark Suite
- Validated on 278 real Lunar Pit Archetypes (`anker.npy`), 150 mined LROC NAC False-Positive Hard Negatives (`luna_hard_negatives_384.bin`), and 9,572 background terrain distractors.
- Proves 100% rejection of background terrain in Stage 1 ($>400\text{k tiles/s}$), 100% Recall@1 in Stage 2 (FP8 sidecar), and 99.3% rejection of hard negatives in Stage 3 (`LunarPitDiscriminator`).

---

## v1.0.6 - C/C++ Native SDK, CMake Integration & FPGA Hardware Verification

**Author**: F1nnSBK

### Major Features & Enhancements

#### 1. Standalone C/C++ Native SDK & CMake Support
- Released the official **Pithos C/C++ Native SDK** (`include/pithos.h`) with full C99/C++20 compatibility, GraalVM isolate lifecycle management, and zero-overhead C-API bindings.
- Added native CMake configuration module (`cmake/PithosConfig.cmake`) enabling instant `find_package(Pithos REQUIRED)` and `target_link_libraries(... Pithos::pithos)` integration.
- Added `pkg-config` support (`cmake/pithos.pc.in`) for direct `gcc`/`clang` and Makefile builds.
- Automated generation of platform-specific C/C++ SDK tarballs (`pithos-c-sdk-<platform>.tar.gz`) on every GitHub release.

#### 2. Bit-Exact FPGA DMA Hardware Verification Suite
- Added dedicated hardware simulation test suite (`tests/test_fpga_co_design.py`) validating:
  - Strict 64-byte cache-line alignment on all memory-mapped virtual off-heap buffers.
  - Bit-for-bit, byte-for-byte exact match between simulated PCIe FPGA DMA sweeps and native Pithos search.
  - Multi-tier Matryoshka sequential DMA channel streaming.
- Fixed per-tier dimension and words-per-record calculation in `FpgaDescriptor`.

#### 3. Modernized Multi-Language Examples (`examples/`)
- Completely restructured and modernized all code examples across **Python** (`quickstart.py`, `fpga_offload_demo.py`), **C/C++** (`demo.c`, `fpga_dma_demo.c`), and **Java** (`PithosApiDemo.java`, `ZeroCostDemo.java`).
- Added comprehensive [Examples Documentation](file:///Users/finnhertsch/projects/lcvk/examples/README.md).

---

## v1.0.5 - Model-Isomorphic Vector Database & FPGA Co-Design


**Author**: F1nnSBK

### Major Features & Enhancements

#### 1. Official PyPI Distribution (`pithosdb`)
- Published the official **`pithosdb`** Python package on PyPI (`pip install pithosdb`).
- Supports both `import pithosdb` and `import pithos` transparently.
- Implements 100% zero-copy NumPy integration via CFFI (`ctypes`) with automatic GIL release (`Py_BEGIN_ALLOW_THREADS`), enabling high-throughput parallel querying in multi-threaded frameworks (FastAPI, Gunicorn).

#### 2. FPGA & Hardware Co-Design Support
- Added `FpgaDescriptor` dataclass capturing virtual/physical base addresses, buffer byte lengths, dimension boundaries, and record counts.
- Added zero-copy NumPy array views for direct memory-mapped access without heap duplication:
  - `index.get_tier_buffer(tier_idx)`: Raw columnar bit vectors (`uint8`).
  - `index.get_metadata_buffer()`: 64-bit metadata and tombstone bitmasks (`uint64`).
  - `index.get_ids_buffer()`: 64-bit record IDs (`int64`).
- Added native vector preconditioning and binarization export:
  - `index.transform_and_quantize(vector)`: Applies Rademacher sign preconditioning and block-diagonal Fast Walsh-Hadamard rotation directly in native C-API, returning 64-bit packed words (`uint64`).

#### 3. Pure UTF-8 Mathematical Typography
- Refactored all doc comments across Java and Python to use universal, native UTF-8 mathematical typography (`ℝᴰ`, `H_u ⊗ Ω_v`, `d_H(a, b) = ∑ popcount(a_w ⊕ b_w)`, `Φ(k)`, `τ ∈ (0, 1]`, `⌈D / 64⌉`).
- Eliminates unrendered LaTeX syntax in IDE hover tooltips (VS Code, Cursor, IntelliJ) and generated JDK 25 Javadoc pages.

#### 4. End-to-End Test Suite & Verification CI/CD
- Added comprehensive Python test suite (`tests/test_pithos_complete.py`) verifying all quantization modes (1-bit, 2-bit ternary, float32 bypass, FP16 sidecar), SVD spectral energy truncation, LSM delta buffer inserts/deletes, merged search, and compaction.
- Integrated automated execution of Java tests (`mvn test`), Python test suite, and all 5 verification benchmarks into GitHub Actions across macOS (Apple Silicon), Linux (x86_64), and Linux (aarch64).

---

## v1.0.1 - Planetary Grid Voting Recall Fix

**Author**: F1nnSBK

### Bug Fixes

**Gate 2 QEG Planetary Grid Filter**
- **The Issue**: A flaw in the 3-way gate logic of the Planetary Grid (`executeVotingRange`). A hardcoded filter was unintentionally discarding 50% of the entire vector space during the search phase because it unconditionally checked if the MSB of the dataset record was `0`, without correlating it with the query vector. 
- **The Fix**: Removed the unconditional MSB discard logic in both 1-bit and 2-bit quantization modes.
- **Impact**: Recall in `queryPlanetaryGrid` searches has increased dramatically (candidates evaluated roughly doubled in uniform random benchmarks).

> [!TIP]
> **Backward Compatibility**: 100%
> This fix only modifies the runtime query evaluation (`FlatIndex.java`). The binary format of the index on disk is completely untouched. You can seamlessly query any index compiled with Pithos `v1.0` using this new version. No re-indexing is required.
