# Pithos – Roadmap & Architectural Extensions

*Architectural Roadmap & Next Steps for the Model-Isomorphic Vector Database*

---

## 1. Flexible & Configurable Quantization Engine

To adapt retrieval recall dynamically to specific mission requirements, Pithos transitions from a fixed 1-bit scheme into a configurable quantization strategy. The mode is selected during index compilation:

### 1.1 Quantization Modes (`QuantizationMode`)
* **`MODE_1BIT` (Default):**
    * *Mechanism:* Stores strictly the sign bit ($+1$ / $-1$).
    * *Suitability:* Maximum throughput, ultra-low memory bus bandwidth. Ideal for edge processors and satellite sensor payloads.
* **`MODE_2BIT` (Ternary / Amplitude-Aware QJL Residuals):**
    * *Status:* Fully implemented and verified.
    * *Mechanism:* Encodes two bits per dimension to represent three states: $-1$, $0$ (values near zero / ambient noise), and $+1$.
    * *Suitability:* Significantly higher recall ($>96\%$) by masking noise dimensions with the $0$ state. Minimal compute overhead over 1-bit.
* **`MODE_FLOAT_HYBRID` (Raw Precision Bypass):**
    * *Mechanism:* Preserves raw 32-bit floating point arrays for low dimensions ($D \le 32$), completely bypassing quantization.

### 1.2 C Initialization API (`pithos.h`)
The index compilation interface supports flexible quantization modes and sidecars:

```c
typedef enum {
    QMODE_1BIT = 0,
    QMODE_2BIT = 1,
    QMODE_FLOAT_HYBRID = 2
} QuantizationMode;

// Index compilation interface with flexible quantization
int vdb_compile_index_file(
    graal_isolatethread_t* thread, 
    char* path, 
    uint8_t planetId, 
    int64_t planetRadius, 
    int32_t dimension, 
    int32_t* tiers, 
    int32_t numTiers, 
    int64_t* ids, 
    float* vectors, 
    int32_t numRecords,
    QuantizationMode qMode
);
```

---

## 2. Two-Stage In-Engine Reranking (Hybrid Sidecar Index)

To achieve near 100% recall for large dimensions, Pithos implements integrated in-engine candidate reranking directly on the off-heap native memory layer without losing sequential memory-bus advantages:

```mermaid
graph LR
    classDef darkBox fill:#1e293b,stroke:#475569,stroke-width:1.5px,color:#f8fafc;
    
    A[Query Vector q]:::darkBox --> B["Gate 2: 1-Bit / 2-Bit Scan"]:::darkBox
    B --> C["Filter Top-K Candidates (IDs)"]:::darkBox
    C --> D["Gate 3: Asymmetric LUT / FP8 Rerank"]:::darkBox
    D --> E["Exact Ranked Top-K Neighbors"]:::darkBox
```

### Implementation Details:
1. **Columnar Layout:** Beside the binary tier files, the index stores original vectors in native **FP8 (E4M3)** or **FP16** sidecar files (`_fp8.bin` / `_fp16.bin`).
2. **Cascaded Pipeline:** The XOR-popcount Hamming scan filters top candidate IDs ($K_{\text{candidate}} = 500$) in microseconds.
3. **Local Rerank:** The native Java Panama engine immediately maps candidate FP8/FP16 vectors off-heap and evaluates exact Euclidean distances using precomputed query LUTs (zero floating-point multiplications).
4. **Result:** Near-perfect recall ($>99.8\%$) with minimal I/O overhead since only candidate records are decoded.

---

## 3. Log-Structured Merge Index (Writable Delta Buffer)

To support real-time insertions without fragmenting the ultra-fast linear memory-mapped (`mmap`) columnar layout, Pithos implements an LSM-tree architecture:

* **Delta Buffer (MemTable):** A compact, lock-free in-memory buffer accepting real-time vector inserts with sub-millisecond latency.
* **Base Index (Immutable SSTable):** The primary, read-only columnar database mapped via POSIX virtual memory.
* **Unified Search:** Queries scan the base index and the dynamic delta buffer concurrently; candidate streams are merged in the native C/Java coordinator.
* **Background Flush:** When the delta buffer reaches its capacity threshold (e.g., 50,000 vectors), it is binarized in the background and merged sequentially into the base index.

---

## 4. Dimension-Adaptive SIMD Kernels (Java Vector API & AVX-512)

For low-dimensional spaces ($D \le 32$), Pithos bypasses bit-compression and dispatches directly to hardware SIMD registers via AVX-512 and ARM Neon intrinsics:

* **Auto-Dispatching:** Dimension checks occur automatically during index initialization.
* **Execution Paths:**
    * For $D \le 32$: Dispatches optimized continuous floating-point L2 Euclidean kernels.
    * For $D \ge 64$: Dispatches the Matryoshka cascaded Hamming scan with binary projection.
* Ensures maximum performance across all dimensionality spectrums.
