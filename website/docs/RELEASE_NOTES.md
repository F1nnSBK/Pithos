---
id: release-notes
title: Release Notes
sidebar_label: Release Notes
---

# Release Notes

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
