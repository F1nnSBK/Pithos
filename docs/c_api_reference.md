# Pithos C-API Reference & Configuration Guide

This document details the native C interface exposed by the compiled `libpithos` shared library, along with runtime configuration and performance tuning guidelines.

---

## Developer Integration Demos

For direct references on how to link and call Pithos:
- **[ZeroCostDemo.java](file:///Users/finnhertsch/projects/lcvk/examples/java/ZeroCostDemo.java):** Demonstrates Java-level GC-free memory mapping via Project Panama's Foreign Function & Memory (FFM) API.
- **[demo.c](file:///Users/finnhertsch/projects/lcvk/examples/cpp/demo.c):** A complete, self-contained C/C++ search client showing isolate management and native query scans.

---

## Complete C API Declarations

The compiled native shared library exposes the following dynamic functions:

```c
// Creates a GraalVM isolate context for JVM execution
int graal_create_isolate(graal_isolate_params_t* params, graal_isolate_t** isolate, graal_isolatethread_t** thread);

// Initializes the Pithos database coordinator
int vdb_init(graal_isolatethread_t* thread);

// Maps an existing multi-tier database off-heap (equal spectral distribution fallback)
int vdb_load_index(graal_isolatethread_t* thread, char* name, char* path);

// Maps an existing database and supplies frozen LoRA weight matrices to compute spectral energy
int vdb_load_index_with_weights(graal_isolatethread_t* thread, char* name, char* path, float* weights, int loraDim);

// Retrieves database metadata attributes (dimension, size, planet settings, tiers count)
int vdb_get_info(graal_isolatethread_t* thread, char* indexName, int* outDimension, long long* outSize, char* outPlanetId, long long* outPlanetRadius, int* outTiersCount);

// Compiles raw float records into a multi-tier database file layout with configurable quantization (qMode: 0=1-bit, 1=2-bit, 2=FP32 bypass)
int vdb_compile_index_file(graal_isolatethread_t* thread, char* path, char planetId, long long planetRadius, int dimension, int* tiers, int numTiers, long long* ids, float* vectors, int numRecords, int qMode);

// Compiles raw float records into a multi-tier database file layout with optional FP16/FP8 sidecar
int vdb_compile_index_file_ext(graal_isolatethread_t* thread, char* path, char planetId, long long planetRadius, int dimension, int* tiers, int numTiers, long long* ids, float* vectors, int numRecords, int qMode, int sidecarMode);

// Compiles raw float records into a universal schema-agnostic single-file .pithos container (DIOGENES format)
int vdb_compile_container(graal_isolatethread_t *thread, const char *path, int32_t dimension, const int32_t *tiers, int32_t num_tiers, const int64_t *ids, const float *vectors, int32_t num_records, int32_t metric_type, int32_t q_mode, int32_t sidecar_mode, const char *metadata_payload, int32_t metadata_len, const char *metadata_format, const char *user_metadata_json);

// Retrieves the user metadata JSON string embedded in a loaded single-file .pithos container
int vdb_get_user_metadata(graal_isolatethread_t *thread, const char *name, char *out_buf, int32_t max_len);

// Compacts multiple compiled indexes into a single consolidated index
int vdb_compact_indexes(graal_isolatethread_t* thread, char* sourcePathsJoined, char* targetPath);

// Retrieves the raw off-heap virtual memory address and length of a specific index tier (FPGA/DMA direct access)
int vdb_get_tier_address(graal_isolatethread_t* thread, char* indexName, int tierIdx, long long* outAddress, long long* outLength);

// Binarizes a single float vector using the index's Walsh-Hadamard preconditioning (asymmetric offloading)
int vdb_transform_and_quantize(graal_isolatethread_t* thread, char* indexName, float* inVector, long long* outPacked);

// Batch KNN search over raw float vectors
int vdb_batch_search(graal_isolatethread_t* thread, char* indexName, float* queries, int numQueries, int k, long long* outIds, int* outDistances);

// Multi-Family Resonant Voting search over raw float queries
long long vdb_query_planetary_grid(graal_isolatethread_t* thread, char* indexName, float* queries, int* queryFamilies, int* queryThresholds, int numQueries, char* votingMask);

// Sets the parallel Disruptor chunk sweep size
int vdb_set_chunk_size(graal_isolatethread_t* thread, char* indexName, long long chunkSize);

// Sets the active energy budget (0.0 to 1.0) to prune lower tiers dynamically
int vdb_set_energy_budget(graal_isolatethread_t* thread, char* indexName, double tau);

// Returns record size of mapped index
long long vdb_size(graal_isolatethread_t* thread, char* indexName);

// Drops/closes an index
int vdb_drop_index(graal_isolatethread_t* thread, char* indexName);

// Shuts down database and frees mapped pages
int vdb_close(graal_isolatethread_t* thread);

// Tears down GraalVM isolate thread
int graal_tear_down_isolate(graal_isolatethread_t* thread);

// LSM Writeable Delta-Buffer Functions:
// Creates a writeable in-memory delta buffer for an index
int vdb_create_delta_buffer(graal_isolatethread_t* thread, char* indexName, int flushThreshold);

// Inserts a raw float vector into the writeable delta buffer (transactions logged to WAL)
int vdb_insert(graal_isolatethread_t* thread, char* indexName, long long id, float* vector);

// Marks a record as deleted (tombstoned) in the delta buffer (logged to WAL)
int vdb_delete_from_delta(graal_isolatethread_t* thread, char* indexName, long long id);

// Returns current record count in the delta buffer
int vdb_delta_size(graal_isolatethread_t* thread, char* indexName);

// Returns 1 if delta buffer size exceeds flush threshold, 0 otherwise
int vdb_needs_flush(graal_isolatethread_t* thread, char* indexName);

// Runs a unified batch search across both base index and writeable delta buffer
int vdb_search_merged(graal_isolatethread_t* thread, char* indexName, float* queries, int numQueries, int k, long long* outIds, int* outDistances);

// Backups/flushes the current delta buffer state into a binary backup file
int vdb_backup_delta(graal_isolatethread_t* thread, char* indexName, char* backupPath);

// Restores delta buffer state from a binary backup file (with mode parameter)
int vdb_restore_delta(graal_isolatethread_t* thread, char* indexName, char* backupPath, int mode);

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
Pithos includes native CUDA support for GPU-accelerated operations:
- **CUDA Hamming Distance Kernels**: Parallel computation of Hamming distances across thousands of threads for massive batch search operations.
- **Multi-Family Voting Kernel**: GPU-accelerated resonant voting for multi-archetype consensus & anomaly detection.
- **Walsh-Hadamard Transform Kernel**: GPU-accelerated transformation of query vectors.
- **Zero-Copy Memory Mapping**: Database tiers are mapped to GPU memory via CUDA pointers, enabling direct GPU access without CPU-GPU memory transfers.

To enable CUDA support, build with the `-Pcuda` Maven profile. See [Dockerfile.cuda](https://github.com/F1nnSBK/Pithos/blob/main/Dockerfile.cuda) for a complete CUDA build environment.


---

## C / C++ SDK & CMake Integration

The native C/C++ SDK tarball (`pithos-c-sdk-<platform>.tar.gz`) is automatically generated on every release:

### Using CMake
```cmake
find_package(Pithos REQUIRED)

add_executable(my_vector_app main.cpp)
target_link_libraries(my_vector_app PRIVATE Pithos::pithos)
```

### Using pkg-config / Make
```bash
gcc -O3 main.c $(pkg-config --cflags --libs pithos) -o my_vector_app
```

