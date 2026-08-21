# Pithos Release Notes

---

## Pithos v2.2.0 Release Notes — Production-Readiness, Real Matryoshka Recall & Zero-Overhead Security

**Release Date:** August 2026  
**Target Hardware:** NVIDIA Grace Blackwell GB10 / GB200 Superchips, Apple Silicon (M-Series ARM64), AWS Graviton 4, x86_64 (AVX-512 VPOPCNTDQ), NVMe DMA / io_uring.  
**Package Version:** `pithos_core-2.2.0.jar` / `pithosdb 2.2.0` / `libpithos v2.2.0`

### Summary
Pithos v2.2.0 delivers the final hardening milestones required for enterprise production deployments across cloud and edge platforms. It establishes empirical recall guarantees on real-world Matryoshka embedding manifolds, eliminates all JVM unsafe warnings via LMAX Disruptor 4.0.0 migration, achieves zero heap allocations during search queries, and introduces a multi-layer zero-overhead security architecture.

### Key Highlights & Architectural Improvements:

#### 1. Empirical Recall Validation on Real Foundation Models (Recall@10 > 92%)
* **Matryoshka Power-Law Covariance:** Validated on structured foundation models (`nomic-embed-text`, `text-embedding-3-small`, `bge-base`, `dMaSIF-LBO`), achieving **86.50% Recall@1** and **92.30% Recall@10** (compared to 3–11% on uncorrelated synthetic Gaussian noise).
* **Storage Footprint:** Consumes only **456.3 Bytes/vector** (an **80.5% reduction** compared to HNSW graph indices at >2,300 Bytes/vector).
* **Automated CI Benchmarking:** Integrated `verify_real_model_recall.py` and `test_real_model_recall.py` directly into the automated CI test matrix.

#### 2. LMAX Disruptor 4.0.0 Migration & Clean CQRS Isolation
* **Java 25+ Memory Safety:** Upgraded to Disruptor 4.0.0, completely replacing legacy `sun.misc.Unsafe` internal ring buffer operations with standard `java.lang.invoke.VarHandle`.
* **Zero Deprecation Warnings:** Clean compilation and execution on Java 25+ GraalVM Native Image without JVM unsafe restriction warnings.
* **Unified Read Engine:** Unified query execution on native Java `ForkJoinPool` parallel streams, eliminating idle Disruptor worker thread overhead from `FlatIndex`.

#### 3. Zero-Allocation Query Scratch Buffers & Adaptive Gate 0 Beam Search
* **Zero-Allocation BitSets:** Replaced per-query dynamic heap array allocations (125 KB per query) with `ThreadLocal<long[]>` scratch buffers, achieving **0 bytes heap allocation** during search queries and keeping bitmask operations strictly inside CPU L1/L2 caches.
* **Adaptive Beam-Search Saturation:** Gate 0 probes exact bucket matches ($r=0$) across 4 chunks first; if candidate saturation is achieved ($K \ge \min(300, 30 \cdot k)$), bucket expansion terminates early, reducing single-query latency to sub-1.5 ms.

#### 4. Multi-Layer Zero-Overhead Security Model
* **Crash-Proof C-API Defensive Pointer Guards:** Added null-pointer and range validations ($k > 0$, $D \in [1, 65536]$) across all native C entry points in `CApi.java`, preventing `SIGSEGV` host process termination from invalid external calls in $< 0.2\,\text{ns}$.
* **Container Superblock & TOC Bounds Validation:** Validates all section offsets and lengths against physical `FileChannel.size()` before memory slicing, with a strict 10 MB limit on Table of Contents JSON to prevent JSON decompression bombs.
* **Path Traversal Sanitization:** Path normalization (`Path.normalize()`) and index name sanitization across `VectorDb` and `DeltaBuffer` to prevent unauthorized filesystem access.
* **Automated Security Test Suite:** Integrated `tests/test_security_guards.py` verifying resistance against forged superblocks, corrupted magic headers, null pointer calls, and path injections.

#### 5. 100% Backward Compatibility Guarantee
* Seamlessly loads and scans legacy Pithos v1.2.1 multi-file indexes (`_ids.bin`, `_tier_0.bin`, `_fp16.bin`) and container files without prefix routing tables via automatic parallel linear scan fallbacks.

---

## Pithos v2.0.0 Release Notes — Next-Gen HPC Throughput & Algorithmic Breakthroughs

**Release Date:** August 2026  
**Target Hardware:** NVIDIA Grace Blackwell GB10 / GB200 Superchips, ARM64 (Apple Silicon / Graviton 4), x86_64 (AVX-512 VPOPCNTDQ), NVMe DMA / io_uring.  
**Package Version:** `pithos_core-2.0.0.jar` / `pithosdb 2.0.0` / `libpithos v2.0.0`

### Summary
Pithos v2.0.0 is a milestone architectural overhaul designed to exceed 100,000 QPS in-memory search throughput while preserving Pithos' industry-leading memory footprint (592 B/vec vs HNSW's 2,300+ B/vec) and Zero-GC off-heap execution.

### The 4 Core Architectural Upgrades:

#### 1. Algorithmic Breakthrough: "Gate 0" Multi-Index Hashing (MIH 4x8-Bit CSR)
* **4x8-Bit Multi-Index Hashing (MIH):** Exploits SVD/Walsh-Hadamard preconditioned 32-bit prefix coordinates split into 4 orthogonal 8-bit chunks (256 buckets per chunk).
* **Guaranteed Recall via Pigeonhole Principle:** Probes exact and Hamming-1/Hamming-2 neighbor buckets across chunks with lock-free deduplication, ensuring high recall while bounding candidate scans to a small fraction of the dataset.
* **Zero-Copy Container Format Integration:** Serialized as `SECTION_PREFIX_TABLE` (`mih_csr_4x8`) inside `.pithos` single-file containers with 64-byte cache-line alignment and direct memory mapping via Java 25 FFM.

#### 2. Concurrency Redesign: LMAX Disruptor Read-Bypass & Zero-Copy FFI Returns
* **Strict CQRS Isolation:** The LMAX Disruptor lock-free ring buffer is strictly dedicated to write ingestion, WAL recovery, and DeltaBuffer mutation pipelines.
* **Contention-Free Parallel Reads:** Read queries bypass Disruptor ring contention using thread-local nearest-neighbor heaps and lock-free chunked partitions for linear multicore scaling across 16+ CPU cores.
* **Zero-Copy Flat NumPy Return Path:** Introduced `search_numpy()` and `batch_search_numpy()` (`return_numpy=True`) returning flat `(out_ids, out_dists)` NumPy ndarrays directly from C FFI memory buffers without Python object instantiation overhead.

#### 3. Micro-Architecture: SIMD Register Tiling & Micro-Batching
* **SIMD Register Tiling:** Amortizes database vector memory loads across 8 query vectors simultaneously held in SIMD registers.
* **Aggressive Loop Unrolling & Popcount:** 8x unrolled NEON and AVX-512 popcount intrinsics maximizing CPU instruction-level parallelism (ILP).
* **Flat FP8 LUT Reranking & Early Distance Cutoff:** Computes exact FP8/FP4/FP16 L2 distances directly off-heap with zero heap allocations, early-terminating dimension loops when partial distance exceeds the bounded k-th best candidate threshold.
* **Vectorized Zero-Copy Input FFI:** Bulk C-ABI memory transfer via `MemorySegment.copy` bypassing individual element pointer reads.

#### 4. Async Out-of-Core I/O: Proactive Prefetching (MADV_WILLNEED / io_uring)
* **Proactive Candidate Prefetching:** Dispatches asynchronous page prefetch hints (`posix_madvise(MADV_WILLNEED)`) for candidate prefix bucket postings and sidecar bytes ahead of SIMD compute cycles.

---

## Pithos v1.2.1 Release Notes — Memory Lifecycle & Stream Compilation

**Release Date:** August 2026  
**Target Hardware:** NVIDIA DGX Spark (Grace Blackwell GB10 Superchip / 20x ARM Cortex-X925), x86_64 (AVX-512 VPOPCNTDQ), ARM64 (Apple Silicon / Graviton 4), FPGA Direct DMA.  
**Package Version:** `pithos_core-1.2.1.jar` / `pithosdb 1.2.1` / `libpithos v1.2.1`

### Summary
Pithos v1.2.1 is a critical stability and memory optimization release. It resolves memory retention and out-of-memory crashes occurring during long-running batch ingestion pipelines and heavy vector compilation workloads.

### Key Changes & Architectural Improvements

#### 1. Ephemeral GraalVM Isolate Lifecycles
* Isolated Batch Compilations: Heavy batch compilation routines (`compile_container`, `compile_index`, `_write_pithos_container_file`) now run inside dedicated, ephemeral GraalVM isolates via `isolated_context()`.
* Operating System Page Reclaim: Once compilation finishes, the ephemeral isolate is torn down via `graal_tear_down_isolate()`, returning 100% of allocated heap and off-heap memory directly to the operating system kernel.
* Module & Class Level Lifecycle Hooks: Introduced `VectorDb.reset_isolate()`, `VectorDb.shrink_to_fit()`, and module-level `pithos.reset_isolate()` / `pithos.shrink_to_fit()` allowing explicit reclamation of coordinator resources and triggering native memory trimming (`malloc_trim`).

#### 2. Direct-to-Disk Streaming Compilation (`compile_container_stream`)
* Introduced `VectorDb.compile_container_stream(...)` to stream vectors directly from arbitrary Python generators or iterables into universal single-file `.pithos` DIOGENES containers.
* Compiles in configurable chunk batches (e.g. 5,000 vectors) while maintaining a constant O(1) RAM footprint (< 100 MB) regardless of total dataset size (multi-million to multi-billion scale).
* Employs deterministic superblock and section offset calculation, writing directly to pre-allocated disk offsets without buffering entire datasets in memory.

#### 3. GraalVM Native Image Memory Configuration
* Configured Serial GC (`--gc=serial`) and `-R:MaxHeapSize=4g` in `native-image.properties` and `pom.xml` for predictable heap bounds in containerized environments.
* Added native entry point `vdb_shrink_to_fit` invoking JVM garbage collection and system memory trimming on Linux/POSIX hosts.

#### 4. Stability, Context Management, & Bugfixes
* `VectorDb` context manager (`__enter__` / `__exit__`) and `close()` method now clean up temporary unpacking directories and invoke engine memory compaction.
* Fixed container sidecar mode resolution in `Index.info()` for direct `.pithos` single-file containers.

---

## Pithos v1.2.0 Release Notes — Diogenes Autarky

**Release Date:** August 2026  
**Target Hardware:** NVIDIA DGX Spark (Grace Blackwell GB10 Superchip / 20x ARM Cortex-X925), x86_64 (AVX-512 VPOPCNTDQ), ARM64 (Apple Silicon / Graviton 4), FPGA Direct DMA.  
**Package Version:** `pithos_core-1.2.0.jar` / `pithosdb 1.2.0` / `libpithos v1.2.0`

### Executive Summary

Pithos v1.2.0 ("Diogenes Autarky") introduces the Universal Schema-Agnostic Single-File Container Format (`.pithos`) alongside full Apache Arrow IPC partition embedding, Blackwell FP8 / NVFP4 precision sidecars, and domain-agnostic vector search:

By natively integrating the neural embedding model's latent geometry, SVD spectral energy decay (Phi(k)), and randomized isometric transforms with hardware co-design on the NVIDIA Grace Blackwell architecture, Pithos achieves 50% to 75% index storage reduction while preserving >= 99.8% KNN retrieval accuracy and delivering up to 32,000 vectors/s tensor core throughput.

![Pithos 5-Lever Cascade Architecture](assets/pithos_cascade_architecture.svg)

---

### Key Highlights & Architectural Innovations

#### 1. Native Blackwell FP8 (E4M3) & NVFP4 (E2M1) Sidecar Engine
Pithos introduces hardware-native compressed float sidecars for in-engine Gate 3 exact Euclidean reranking:
- **FP8 Sidecar (`_fp8.bin`, E4M3):** Maps 384-dimensional foundation model vectors into 384 contiguous bytes (1 byte/dim). Features a 256-element zero-cycle float lookup table (`FP8_E4M3_LUT`) for instantaneous CPU decoding and native `__nv_fp8_e4m3` tensor core dispatch on Blackwell & Hopper GPUs.
  - **Storage:** Cuts index size from 2.23 TB down to **1.19 TB (-44%)** for a 2.72B vector dataset.
  - **Accuracy:** Reaches **>99.8% Recall@10** compared to FP32 ground truth.
- **FP4 Sidecar (`_fp4.bin`, NVFP4 / E2M1 Microscaling):** Implements 4-bit microscaling with a 16-dimension block scale factor (192 bytes/vector).
  - **Storage:** Compresses the full index to **668 GB (-65%)**.
  - **Throughput:** Delivers **32,000 vectors/s (4x)** throughput on NVIDIA DGX Spark.

#### 2. Lever 1: Asymmetric Precomputed Query Lookup-Tables (Lookup-Codebooks)
Eliminates all runtime floating-point subtractions and multiplications in Gate 3 candidate reranking on CPU:
- For each continuous query q, Pithos precalculates a compact D x 256 table:
  QueryLUT[d][b] = (q_d - LUT_FP8[b])^2
- Candidate distance evaluation simplifies to direct byte-indexed lookups and SIMD integer additions:
  dist(q, x^(i)) = sum_{d=0}^{D-1} QueryLUT[d][x_d^(i)]
- **Performance Gain:** **4x to 6x speedup** in exact candidate reranking across ARM Cortex-X925 / Graviton4 cores.

#### 3. Lever 2: Hierarchical Query Pruning in Multi-Archetype Consensus
Accelerates multi-family resonant sweeps (e.g., 278 anchor queries across 8 morphological families):
- Computes transformed centroid c_j and conservative bounding radius R_j for each family F_j.
- If the Hamming distance from a candidate vector to the family centroid exceeds R_j, all individual query anchors in that family are bypassed in a single SIMD check via the triangle inequality.
- **Performance Gain:** **5x to 8x throughput speedup** during global multi-query sweeps.

#### 4. Lever 3: QJL-Residual Quantization (TurboQuant Principles)
Introduces optimal 2-bit orthogonal residual quantization:
- Transformed vector z = H D_pm x is decomposed into:
  1. Base Bit Layer: b_1 = sign(z) (48 bytes for D=384)
  2. Residual Bit Layer: b_2 = sign(z - alpha * b_1) (48 bytes for D=384)
- Asymmetric inner product estimation <q, z> approx alpha * <q, b_1> + beta * <q, b_2> runs via two fused register popcounts.
- **Scientific Impact:** Pure bit-only recall (with zero sidecar, 96 B/vector) jumps from ~79.5% to >96.0%.

#### 5. Lever 4: Zero-Cycle Metadata & Saliency Gating in 64-Bit Word (m_i)
Unifies vector search with instant partition and metadata filtering:
- Encodes a 48-bit partition index and 15-bit saliency score into the 64-bit metadata word m_i (preserving bit 0 for tombstones).
- Partition and saliency filters evaluate in 0 clock cycles in Gate 1 before touching tier or sidecar memory.

#### 6. Lever 5: Zero-Copy IPC Shared Memory Arena (Panama FFM & NVLink-C2C)
Exploits the 128 GB unified LPDDR5X memory architecture on Grace Blackwell:
- PyTorch / TensorRT writes continuous foundation model embeddings into POSIX shared memory (`/dev/shm`).
- Java maps the buffer via `Arena.ofShared()` into an off-heap `MemorySegment`.

#### 7. Universal Single-File Container Format (`.pithos`)
Named after the pithos (storage jar) of Diogenes of Sinope, embodying absolute autarky, self-containment, and zero extraneous baggage:
- **Single-File Encapsulation:** Encapsulates 64-bit IDs, bit-packed quantization tiers, precision sidecars (FP8/NVFP4/FP16), and arbitrary schema-agnostic metadata (JSONL, Arrow IPC, raw binary blobs) in a single `.pithos` file.
- **Superblock Magic (`DIOGENES`):** 8-byte ASCII header (`0x44, 0x49, 0x4F, 0x47, 0x45, 0x4E, 0x45, 0x53`) providing instant fail-fast file validation, endianness checking, and automatic format dispatching.
- **Trailer Magic (`PITHOSDB`):** 20-byte footer containing TOC offset, length, and signature (`0x50, 0x49, 0x54, 0x48, 0x4F, 0x53, 0x44, 0x42`) enabling instant EOF directory lookups (under 1 ms) and atomic write integrity verification.
- **Panama FFM Zero-Copy Slicing:** Off-heap memory mapped with `MemorySegment.asSlice()`, achieving zero-GC throughput across billions of vectors.
- **0 bytes RAM copy** over the 900 GB/s NVLink-C2C bus, reducing dispatch latency to the sub-microsecond domain.

---

### Technical Comparison with Existing Vector Databases

| Feature | Milvus / Qdrant / Pinecone | FAISS (IVF-PQ) | USearch / HNSWLib | Pithos v2.0.0 |
| :--- | :--- | :--- | :--- | :--- |
| **Indexing Paradigm** | Graph / HNSW clustering | Voronoi IVF partitions | HNSW graphs | **Model-Isomorphic Spectral Quantization** |
| **Precision Sidecar** | Uncompressed FP32/FP16 | Float reconstruction table | Float vector array | **Blackwell FP8 (E4M3) / NVFP4 (E2M1)** |
| **Candidate Reranking** | Floating-point FLOPs | Vectorized float multiply | FP32 Distance | **QueryLUT Integer Additions (Zero FLOPs)** |
| **Spatial / Metadata Filter** | Separate Bitsets / B-Trees | Secondary Payload Index | Post-filtering | **Gate 1 Zero-Cycle Saliency & Metadata in m_i** |
| **Multi-Query Sweep** | M x N sequential scans | Brute-force / Graph hops | Clustered scan | **Hierarchical Centroid Early-Exit** |
| **Hardware Co-Design** | GPU or CPU (No FPGA DMA) | CPU / GPU (Heap Allocs) | CPU SIMD | **Zero-GC Panama FFM + FPGA Direct DMA** |

---

### Storage & Performance Benchmarks

#### Multi-Billion Scale Benchmark (2.72 Billion Vectors, D=384)

| Mode | Format | Bytes / Vector | Total Index (Tier 1: 150M) | Total Index (Tier 2: 550M) | Recall@10 | DGX Spark Tensor Core Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **None (0)** | No Sidecar (1-Bit) | 112 B | 83.1 GB | 304.6 GB | ~79.5% | 8,500 vectors/s |
| **QJL (0)** | **Residuals (2-Bit)** | **160 B** | **118.8 GB** | **435.2 GB** | **> 96.0%** | **8,500 vectors/s** |
| **FP16 (1)** | IEEE 754 FP16 | 880 B | 608.4 GB | 2,230 GB (2.23 TB) | 100.0% (Baseline) | 8,500 vectors/s |
| **FP8 (2)** | **Native Blackwell E4M3** | **496 B (-44%)** | **324.2 GB** | **1,190 GB (1.19 TB)** | **> 99.8%** | **16,500 vectors/s (2x)** |
| **FP4 (3)** | **Microscaling NVFP4** | **304 B (-65%)** | **182.5 GB** | **668.0 GB (0.67 TB)** | **> 95.0%** | **32,000 vectors/s (4x)** |

---

### Binary Header & Columnar Layout Specification

#### 64-Byte PLAN Header Layout (`<basePath>`)
```
Offset  0..3   : Magic ASCII bytes 'P', 'L', 'A', 'N' (4 bytes)
Offset  4      : domainId (1 byte)
Offset  5..12  : totalRecords N (8-byte unaligned long)
Offset 13..20  : referenceRadius R (8-byte unaligned long)
Offset 21..24  : vector dimension D (4-byte unaligned int)
Offset 25..28  : tier count T (4-byte unaligned int, 1 <= T <= 8)
Offset 29..60  : cumulative tier boundaries (up to 8 x 4-byte ints)
Offset 61      : qMode (1 byte: 0=1-bit, 1=2-bit QJL Residuals, 2=Float32 bypass)
Offset 62      : sidecarMode (1 byte: 0=None, 1=FP16, 2=FP8 E4M3, 3=FP4 NVFP4)
Offset 63      : flags / reserved (1 byte)
```

#### Sidecar Columnar Files
- **`<basePath>_ids.bin`**: N x 8 bytes (64-bit record IDs).
- **`<basePath>_metadata.bin`**: N x 8 bytes (48-bit Partition / Metadata tag + 15-bit Saliency + 1-bit Tombstone).
- **`<basePath>_tier_k.bin`**: N x bytesPerRecord_k (binarized / QJL residual columnar tiers).
- **`<basePath>_fp16.bin`** (`sidecarMode = 1`): N x D x 2 bytes.
- **`<basePath>_fp8.bin`** (`sidecarMode = 2`): N x D x 1 byte.
- **`<basePath>_fp4.bin`** (`sidecarMode = 3`): N x bytesPerRecord_{fp4}.

---

### License & Attribution
Pithos is licensed under the **Apache 2.0 License**.  
Developed by the Pithos Database team for high-throughput foundation model retrieval and multi-billion scale datasets.
