# ⚱ Pithos v1.2.0 Release Notes — "Diogenes Autarky"

**Release Date:** August 2026  
**Target Hardware:** NVIDIA DGX Spark (Grace Blackwell GB10 Superchip / 20x ARM Cortex-X925), x86_64 (AVX-512 VPOPCNTDQ), ARM64 (Apple Silicon / Graviton 4), FPGA Direct DMA.  
**Package Version:** `pithos_core-1.2.0.jar` / `pithosdb 1.2.0` / `libpithos v1.2.0`

---

## Executive Summary

**Pithos v1.2.0 ("Diogenes Autarky")** introduces the **Universal Schema-Agnostic Single-File Container Format (`.pithos`)** alongside full Apache Arrow IPC partition embedding, Blackwell FP8 / NVFP4 precision sidecars, and domain-agnostic vector search:

By natively integrating the neural embedding model's latent geometry, SVD spectral energy decay ($\Phi(k)$), and randomized isometric transforms with hardware co-design on the NVIDIA Grace Blackwell architecture, Pithos v1.1.0 achieves **50% to 75% index storage reduction** while preserving **$\ge 99.8\%$ KNN retrieval accuracy** and delivering up to **32,000 vectors/s tensor core throughput**.

![Pithos 5-Lever Cascade Architecture](assets/pithos_cascade_architecture.svg)

---

## Key Highlights & Architectural Innovations

### 1. Native Blackwell FP8 (E4M3) & NVFP4 (E2M1) Sidecar Engine
Pithos v1.1.0 introduces hardware-native compressed float sidecars for in-engine Gate 3 exact Euclidean reranking:
- **FP8 Sidecar (`_fp8.bin`, E4M3):** Maps 384-dimensional foundation model vectors into 384 contiguous bytes (1 byte/dim). Features a 256-element zero-cycle float lookup table (`FP8_E4M3_LUT`) for instantaneous CPU decoding and native `__nv_fp8_e4m3` tensor core dispatch on Blackwell & Hopper GPUs.
  - **Storage:** Cuts index size from 2.23 TB down to **1.19 TB (-44%)** for a 2.72B vector dataset.
  - **Accuracy:** Reaches **$>99.8\%$ Recall@10** compared to FP32 ground truth.
- **FP4 Sidecar (`_fp4.bin`, NVFP4 / E2M1 Microscaling):** Implements 4-bit microscaling with a 16-dimension block scale factor (192 bytes/vector).
  - **Storage:** Compresses the full index to **668 GB (-65%)**.
  - **Throughput:** Delivers **32,000 vectors/s (4x)** throughput on NVIDIA DGX Spark.

### 2. Lever 1: Asymmetric Precomputed Query Lookup-Tables (Lookup-Codebooks)
Eliminates all runtime floating-point subtractions and multiplications in Gate 3 candidate reranking on CPU:
- For each continuous query $\mathbf{q} \in \mathbb{R}^D$, Pithos precalculates a compact $D \times 256$ table:
  $$\text{QueryLUT}[d][b] = \left(q_d - \text{LUT}_{\text{FP8}}[b]\right)^2$$
- Candidate distance evaluation simplifies to direct byte-indexed lookups and SIMD integer additions:
  $$\text{dist}(\mathbf{q}, \mathbf{x}^{(i)}) = \sum_{d=0}^{D-1} \text{QueryLUT}[d]\left[x_d^{(i)}\right]$$
- **Performance Gain:** **4x to 6x speedup** in exact candidate reranking across ARM Cortex-X925 / Graviton4 cores.

### 3. Lever 2: Hierarchical Query Pruning in Multi-Archetype Consensus
Accelerates multi-family resonant sweeps (e.g., 278 anchor queries across 8 morphological families):
- Computes transformed centroid $\mathbf{c}_j$ and conservative bounding radius $R_j$ for each family $F_j$.
- If the Hamming distance from a candidate vector to the family centroid exceeds $R_j$, **all individual query anchors in that family are bypassed in a single SIMD check** via the triangle inequality.
- **Performance Gain:** **5x to 8x throughput speedup** during global multi-query sweeps.

### 4. Lever 3: QJL-Residual Quantization (TurboQuant Principles)
Introduces optimal 2-bit orthogonal residual quantization:
- Transformed vector $\mathbf{z} = \mathbf{H} \mathbf{D}_{\pm} \mathbf{x}$ is decomposed into:
  1. Base Bit Layer: $\mathbf{b}_1 = \text{sign}(\mathbf{z})$ (48 bytes for $D=384$)
  2. Residual Bit Layer: $\mathbf{b}_2 = \text{sign}(\mathbf{z} - \alpha \mathbf{b}_1)$ (48 bytes for $D=384$)
- Asymmetric inner product estimation $\langle \mathbf{q}, \mathbf{z} \rangle \approx \alpha \langle \mathbf{q}, \mathbf{b}_1 \rangle + \beta \langle \mathbf{q}, \mathbf{b}_2 \rangle$ runs via two fused register popcounts.
- **Scientific Impact:** Pure bit-only recall (with **zero sidecar**, 96 B/vector) jumps from **~79.5% to >96.0%**.

### 5. Lever 4: Zero-Cycle Metadata & Saliency Gating in 64-Bit Word ($m_i$)
Unifies vector search with instant partition and metadata filtering:
- Encodes a 48-bit partition index and 15-bit saliency score into the 64-bit metadata word $m_i$ (preserving bit 0 for tombstones).
- Partition and saliency filters evaluate in **0 clock cycles in Gate 1** before touching tier or sidecar memory.

### 6. Lever 5: Zero-Copy IPC Shared Memory Arena (Panama FFM & NVLink-C2C)
Exploits the 128 GB unified LPDDR5X memory architecture on Grace Blackwell:
- PyTorch / TensorRT writes continuous foundation model embeddings into POSIX shared memory (`/dev/shm`).
- Java maps the buffer via `Arena.ofShared()` into an off-heap `MemorySegment`.

### 7. Universal Single-File Container Format (`.pithos`)
Named after the *pithos* (storage jar) of **Diogenes of Sinope**, embodying absolute autarky, self-containment, and zero extraneous baggage:
- **Single-File Encapsulation:** Encapsulates 64-bit IDs, bit-packed quantization tiers, precision sidecars (FP8/NVFP4/FP16), and arbitrary schema-agnostic metadata (JSONL, Arrow IPC, raw binary blobs) in a single `.pithos` file.
- **Superblock Magic (`DIOGENES`):** 8-byte ASCII header (`0x44, 0x49, 0x4F, 0x47, 0x45, 0x4E, 0x45, 0x53`) providing instant fail-fast file validation, endianness checking, and automatic format dispatching.
- **Trailer Magic (`PITHOSDB`):** 20-byte footer containing TOC offset, length, and signature (`0x50, 0x49, 0x54, 0x48, 0x4F, 0x53, 0x44, 0x42`) enabling instant EOF directory lookups (under 1 ms) and atomic write integrity verification.
- **Panama FFM Zero-Copy Slicing:** Off-heap memory mapped with `MemorySegment.asSlice()`, achieving zero-GC throughput across billions of vectors.
- **0 bytes RAM copy** over the 900 GB/s NVLink-C2C bus, reducing dispatch latency to the sub-microsecond domain.

---

## Technical Comparison with Existing Vector Databases

| Feature | Milvus / Qdrant / Pinecone | FAISS (IVF-PQ) | USearch / HNSWLib | **Pithos v1.2.0** |
| :--- | :--- | :--- | :--- | :--- |
| **Indexing Paradigm** | Graph / HNSW clustering | Voronoi IVF partitions | HNSW graphs | **Model-Isomorphic Spectral Quantization** |
| **Precision Sidecar** | Uncompressed FP32/FP16 | Float reconstruction table | Float vector array | **Blackwell FP8 (E4M3) / NVFP4 (E2M1)** |
| **Candidate Reranking** | Floating-point FLOPs | Vectorized float multiply | FP32 Distance | **QueryLUT Integer Additions (Zero FLOPs)** |
| **Spatial / Metadata Filter** | Separate Bitsets / B-Trees | Secondary Payload Index | Post-filtering | **Gate 1 Zero-Cycle Saliency & Metadata in $m_i$** |
| **Multi-Query Sweep** | $M \times N$ sequential scans | Brute-force / Graph hops | Clustered scan | **Hierarchical Centroid Early-Exit** |
| **Hardware Co-Design** | GPU or CPU (No FPGA DMA) | CPU / GPU (Heap Allocs) | CPU SIMD | **Zero-GC Panama FFM + FPGA Direct DMA** |

---

## Storage & Performance Benchmarks

### Multi-Billion Scale Benchmark (2.72 Billion Vectors, D=384)

| Mode | Format | Bytes / Vector | Total Index (Tier 1: 150M) | Total Index (Tier 2: 550M) | Recall@10 | DGX Spark Tensor Core Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **None (0)** | No Sidecar (1-Bit) | 112 B | 83.1 GB | 304.6 GB | ~79.5% | 8,500 vectors/s |
| **QJL (0)** | **Residuals (2-Bit)** | **160 B** | **118.8 GB** | **435.2 GB** | **> 96.0%** | **8,500 vectors/s** |
| **FP16 (1)** | IEEE 754 FP16 | 880 B | 608.4 GB | 2,230 GB (2.23 TB) | 100.0% (Baseline) | 8,500 vectors/s |
| **FP8 (2)** | **Native Blackwell E4M3** | **496 B (-44%)** | **324.2 GB** | **1,190 GB (1.19 TB)** | **> 99.8%** | **16,500 vectors/s (2x)** |
| **FP4 (3)** | **Microscaling NVFP4** | **304 B (-65%)** | **182.5 GB** | **668.0 GB (0.67 TB)** | **> 95.0%** | **32,000 vectors/s (4x)** |

---

## Binary Header & Columnar Layout Specification

### 64-Byte PLAN Header Layout (`<basePath>`)
```
Offset  0..3   : Magic ASCII bytes 'P', 'L', 'A', 'N' (4 bytes)
Offset  4      : domainId (1 byte)
Offset  5..12  : totalRecords N (8-byte unaligned long)
Offset 13..20  : referenceRadius R (8-byte unaligned long)
Offset 21..24  : vector dimension D (4-byte unaligned int)
Offset 25..28  : tier count T (4-byte unaligned int, 1 <= T <= 8)
Offset 29..60  : cumulative tier boundaries (up to 8 x 4-byte ints)
Offset 61      : qMode (1 byte: 0=1-bit, 1=2-bit QJL Residuals, 2=Float32 bypass)
Offset 62      : sidecarMode (1 byte: 0=None, 1=FP16, 2=FP8 E4M3, 3=FP4 NVFP4)  <-- [NEW in v1.1.0]
Offset 63      : flags / reserved (1 byte)                                      <-- [NEW in v1.1.0]
```

### Sidecar Columnar Files
- **`<basePath>_ids.bin`**: $N \times 8$ bytes (64-bit record IDs).
- **`<basePath>_metadata.bin`**: $N \times 8$ bytes (48-bit Partition / Metadata tag + 15-bit Saliency + 1-bit Tombstone).
- **`<basePath>_tier_k.bin`**: $N \times \text{bytesPerRecord}_k$ (binarized / QJL residual columnar tiers).
- **`<basePath>_fp16.bin`** (`sidecarMode = 1`): $N \times D \times 2$ bytes.
- **`<basePath>_fp8.bin`** (`sidecarMode = 2`): $N \times D \times 1$ byte.
- **`<basePath>_fp4.bin`** (`sidecarMode = 3`): $N \times \text{bytesPerRecord}_{\text{fp4}}$.

---

## API & Language Bindings

### Python (`pithosdb` / `pithos`)
```python
import pithosdb
import numpy as np

with pithosdb.VectorDb() as db:
    records = np.random.randn(50_000, 384).astype(np.float32)
    
    # 1. Compile index with native FP8 (E4M3) sidecar
    pithosdb.VectorDb.compile_container(
        path="temp/fp8_dataset.pithos",
        records=records,
        tiers=[64, 128, 256, 384],
        sidecar_mode=pithosdb.SidecarMode.FP8,
        user_metadata={"dataset": "foundation_embeddings", "curator": "Diogenes"}
    )
    
    # 2. Memory-map index & run search with precomputed query LUT reranking
    index = db.load_index("dataset", "temp/fp8_dataset.pithos")
    queries = np.random.randn(10, 384).astype(np.float32)
    results = index.search(queries, k=10)
    for res in results:
        print(f"Match ID: {res.id}, Distance: {res.distance:.4f}")
```

### C/C++ Native SDK (`pithos.h`)
```c
#include "pithos.h"

// Initialize database coordinator
graal_isolate_t *isolate;
graal_isolatethread_t *thread;
graal_create_isolate(NULL, &isolate, &thread);
vdb_init(thread);

// Compile index with FP8 sidecar
int32_t tiers[] = {64, 128, 256, 384};
vdb_compile_index_file_ext(
    thread, "temp/fp8_dataset", 1, 1737400, 384,
    tiers, 4, ids_ptr, vecs_ptr, num_records,
    PITHOS_QMODE_1BIT, PITHOS_SIDECAR_FP8
);
```

---

## Verification & Test Suite

Pithos includes an exhaustive test and verification suite:

```bash
# 1. Execute Java Core Unit & Bit-Exact Precision Tests
mvn clean test

# 2. Run Python Complete Integration Test Suite
python3 -m unittest discover tests

# 3. Run Precision, Recall & Speedup Verification Benchmark
python3 benchmarks/verify_fp8_sidecar.py --dim 384 --records 50000 --benchmark-levers
```

---

## License & Attribution
Pithos is licensed under the **Apache 2.0 License**.  
Developed by the Pithos Database team for high-throughput foundation model retrieval and multi-billion scale datasets.
