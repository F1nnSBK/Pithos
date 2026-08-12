# C-API Reference & Runtime

This document details the native C interface exposed by the compiled `libpithos` shared library, runtime configuration, and tuning guidelines.

---

## 1. Core Lifecycle & Initialization

Before any search operations can be performed, the GraalVM isolate context must be created and the Pithos engine initialized.

```c
// Creates a GraalVM isolate context for JVM execution
int graal_create_isolate(graal_isolate_params_t* params, graal_isolate_t** isolate, graal_isolatethead_t** thread);

// Initializes the Pithos database coordinator
int vdb_init(graal_isolatethead_t* thread);
```

## 2. Index Compilation & The FP16 Sidecar

Pithos translates raw floating-point vectors into a highly optimized, dimensionally reduced binary format. When compiling an index, you can decide whether to include an **FP16 Sidecar**.

```c
// Compiles raw float records into a multi-tier database file layout
// writeFp16: 1=Include FP16 Sidecar (Exact Rescoring), 0=Exclude (84% Storage Reduction)
int vdb_compile_index_file_ext(
    graal_isolatethead_t* thread, char* path, char planetId, long long planetRadius, 
    int dimension, int* tiers, int numTiers, long long* ids, float* vectors, 
    int numRecords, int qMode, int writeFp16
);
```

### Understanding the `writeFp16` Option
*   **With FP16 Sidecar (`writeFp16 = 1`)**: Pithos writes the raw vectors as Half-Precision (FP16) floats alongside the binary index. This allows the search cascade to perform exact $L2$ distance rescoring on the final candidates (Gate 3).
*   **Without FP16 Sidecar (`writeFp16 = 0`)**: The sidecar is omitted. Final rescoring falls back to asymmetric Hamming distances. This saves **~84% disk space** with only a minor drop in single-query Recall@10. **Note:** *Multi-Family Resonant Voting is unaffected by this setting.*

## 3. Database Loading & Configuration

Once an index is compiled, it is loaded (mmap'd) entirely off-heap.

```c
// Maps an existing multi-tier database off-heap
int vdb_load_index(graal_isolatethead_t* thread, char* name, char* path);

// Retrieves database metadata attributes
int vdb_get_info(graal_isolatethead_t* thread, char* indexName, int* outDimension, long long* outSize, char* outPlanetId, long long* outPlanetRadius, int* outTiersCount);
```

### Runtime Tuning
Pithos exposes runtime parameters to control search latency and hardware utilization.

```c
// Sets the parallel chunk sweep size
int vdb_set_chunk_size(graal_isolatethead_t* thread, char* indexName, long long chunkSize);

// Sets the active energy budget (0.0 to 1.0) to prune lower tiers dynamically
int vdb_set_energy_budget(graal_isolatethead_t* thread, char* indexName, double tau);
```

## 4. Search Operations

Pithos offers two primary modalities for search: Single-Query KNN and Multi-Family Voting.

```c
// Batch KNN search over raw float vectors
int vdb_batch_search(graal_isolatethead_t* thread, char* indexName, float* queries, int numQueries, int k, long long* outIds, int* outDistances);

// Multi-Family Resonant Voting search
long long vdb_query_planetary_grid(graal_isolatethead_t* thread, char* indexName, float* queries, int* queryFamilies, int* queryThresholds, int numQueries, char* votingMask);
```

---

## 5. Advanced: CUDA Acceleration (Experimental)

:::warning UNSTABLE API
The CUDA integration for Pithos is currently experimental and highly unstable. It is not recommended for production environments. The API and Native JNI bindings are subject to breaking changes.
:::

When compiled with `-Pcuda`, Pithos provides hardware-accelerated entry points for large batches.

```c
// Initializes the CUDA runtime context
int vdb_cuda_init(graal_isolatethread_t* thread, int deviceId);

// Massively parallel batch search on NVIDIA GPUs
int vdb_cuda_batch_search(graal_isolatethread_t* thread, char* indexName, float* queries, int numQueries, int k, long long* outIds, int* outDistances);
```

## 6. Shutdown

```c
// Drops/closes an index
int vdb_drop_index(graal_isolatethead_t* thread, char* indexName);

// Shuts down database and frees mapped pages
int vdb_close(graal_isolatethead_t* thread);

// Tears down GraalVM isolate thread
int graal_tear_down_isolate(graal_isolatethead_t* thread);

// LSM Writeable Delta-Buffer Functions:
// Creates a writeable in-memory delta buffer for an index
int vdb_create_delta_buffer(graal_isolatethead_t* thread, char* indexName, int flushThreshold);

// Inserts a raw float vector into the writeable delta buffer (transactions logged to WAL)
int vdb_insert(graal_isolatethead_t* thread, char* indexName, long long id, float* vector);

// Marks a record as deleted (tombstoned) in the delta buffer (logged to WAL)
int vdb_delete_from_delta(graal_isolatethead_t* thread, char* indexName, long long id);

// Returns current record count in the delta buffer
int vdb_delta_size(graal_isolatethead_t* thread, char* indexName);

// Returns 1 if delta buffer size exceeds flush threshold, 0 otherwise
int vdb_needs_flush(graal_isolatethead_t* thread, char* indexName);

// Runs a unified batch search across both base index and writeable delta buffer
int vdb_search_merged(graal_isolatethead_t* thread, char* indexName, float* queries, int numQueries, int k, long long* outIds, int* outDistances);

// Backups/flushes the current delta buffer state into a binary backup file
int vdb_backup_delta(graal_isolatethead_t* thread, char* indexName, char* backupPath);

// Restores delta buffer state from a binary backup file (with mode parameter)
int vdb_restore_delta(graal_isolatethead_t* thread, char* indexName, char* backupPath, int mode);

// ====================================================================
// CUDA Acceleration Functions
// ====================================================================

// Initializes CUDA with specified device ID
int vdb_cuda_init(graal_isolatethread_t* thread, int deviceId);

// Shuts down CUDA resources
int vdb_cuda_shutdown(graal_isolatethread_t* thread);

// Checks if CUDA is available (returns 1 if available, 0 otherwise)
int vdb_cuda_is_available(graal_isolatethread_t* thread);

// Performs CUDA-accelerated batch search
int vdb_cuda_batch_search(graal_isolatethread_t* thread, char* indexName, float* queries, int numQueries, int k, long long* outIds, int* outDistances);

// Performs CUDA-accelerated multi-family resonant voting
long long vdb_cuda_query_planetary_grid(graal_isolatethread_t* thread, char* indexName, float* queries, int* queryFamilies, int* queryThresholds, int numQueries, char* votingMask);
```

---

## Runtime Configuration Guide

### 1. Quantization & Formats (`qMode`)
Configured during compilation via the `qMode` parameter in `vdb_compile_index_file`. The mode is saved in the header and automatically applied at load time:
- **`0`**: 1-bit sign-only (highest compression).
- **`1`**: 2-bit ternary (active mask + signs, enabling exact asymmetric binary/ternary distance estimators).
- **`2`**: FP32 raw bypass (skips quantization, saves raw rotated 32-bit floating point values for low dimensions).

### 2. FP16 Stage 2 Reranking & Optional Sidecar
By default, Pithos compiles and exports the raw vectors in IEEE 754 half-precision to a sidecar file named `<basePath>_fp16.bin` for high-recall Stage 2 reranking.
- **Optional Compilation**: You can bypass FP16 sidecar creation via `vdb_compile_index_file_ext` by setting `writeFp16 = 0` (or `write_fp16=False` in Python). This results in an **84% reduction in disk footprint** and **2.6x faster index compilation**.
- **Auto-detection & Fallback**: If Pithos finds the `<basePath>_fp16.bin` file when loading the index via `vdb_load_index`, it maps it off-heap and enables Stage 2 reranking automatically. If absent or deleted, the search path dynamically falls back to asymmetric L2 distance calculations directly on the binarized/ternary columns.
- **Performance Trade-Off**:
  - *With FP16*: Primarily a **recall-maximizer**, bringing KNN Recall@10 up to exact levels (e.g., ~53% on synthetic hyper-spheres) through native Stage-2 float reranking.
  - *Without FP16*: A **speed-and-space optimizer** (84% smaller size). KNN recall drops (e.g. to ~30%), but **Multi-Family Resonant Voting** remains completely unaffected, executing at maximum speed and identical match counts.
- **Bulk FFM Copy Optimization**: POINT-lookup accesses during Stage 2 are optimized using native FFM `MemorySegment.copy` (bulk copies replacing element-by-element off-heap JVM crossings) to deliver native speedups over FAISS.

### 3. Search & Runtime Parameters
- **Information Budget ($\tau$)**: Change the dynamic pruning threshold on the fly via `vdb_set_energy_budget`. E.g., setting $\tau = 0.90$ bypasses columns corresponding to less significant singular vectors, reducing memory bandwidth usage.
- **Parallel Chunk Size**: Optimize Disruptor worker granularity using `vdb_set_chunk_size`.

### 4. FPGA / Custom Hardware Acceleration (Co-Design)
Pithos is specifically designed for hybrid CPU-FPGA/GPU acceleration workflows, where the host CPU handles the application orchestration and the hardware accelerator performs massive Hamming sweeps:
- **Zero-Copy DMA Acceleration (`vdb_get_tier_address`)**: Custom PCIe hardware kernels or FPGA DMA controllers can retrieve the exact virtual off-heap memory-mapped address and length of specific tier buffers. Because these buffers are read-only, cache-aligned, and contiguous, they can be streamed directly into custom acceleration engines via DMA, bypassing Java GC, JVM boundaries, and CPU overhead.
- **Asymmetric Vector Offloading (`vdb_transform_and_quantize`)**: A host system can quickly transform and binarize incoming query vectors on the CPU using Pithos's Rademacher preconditioning and Walsh-Hadamard rotations. The resulting query bit vectors can then be passed to the FPGA/GPU to perform low-latency binary Hamming distance sweeps directly against the raw off-heap database buffers.

### 5. CUDA GPU Acceleration
Pithos now includes native CUDA support for GPU-accelerated operations:
- **CUDA Hamming Distance Kernels**: Parallel computation of Hamming distances across thousands of threads for massive batch search operations.
- **Multi-Family Voting Kernel**: GPU-accelerated resonant voting for planetary-scale anomaly detection.
- **Walsh-Hadamard Transform Kernel**: GPU-accelerated transformation of query vectors.
- **Zero-Copy Memory Mapping**: Database tiers are mapped to GPU memory via CUDA pointers, enabling direct GPU access without CPU-GPU memory transfers.

To enable CUDA support, build with the `-Pcuda` Maven profile.
