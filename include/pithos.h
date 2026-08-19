/**
 * @file pithos.h
 * @brief High-Performance Model-Isomorphic Vector Database (MIDB) Engine - Native C/C++ API.
 *
 * Pithos is an Ahead-of-Time (AOT) compiled, dimension-agnostic vector search engine
 * optimized for Matryoshka-structured binary embeddings at planetary scale,
 * with zero-GC off-heap virtual memory mapping and direct FPGA/DMA hardware co-design support.
 *
 * Compatible with C99, C11, C++17, C++20, OpenCL, and custom PCIe hardware drivers.
 */

#ifndef PITHOS_H
#define PITHOS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * GraalVM Native Image Isolate Types
 * ========================================================================= */
#ifndef GRAAL_ISOLATE_H
typedef struct __graal_isolate_t graal_isolate_t;
typedef struct __graal_isolatethread_t graal_isolatethread_t;

int graal_create_isolate(void *params, graal_isolate_t **isolate, graal_isolatethread_t **thread);
int graal_tear_down_isolate(graal_isolatethread_t *thread);
int graal_detach_thread(graal_isolatethread_t *thread);
#endif

/* =========================================================================
 * Return Codes
 * ========================================================================= */
#define PITHOS_SUCCESS                 0
#define PITHOS_ERR_DB_NOT_INIT        -1
#define PITHOS_ERR_INDEX_NOT_FOUND    -2
#define PITHOS_ERR_INVALID_PARAM      -3
#define PITHOS_ERR_INTERNAL           -4
#define PITHOS_ERR_FILE_IO            -5
#define PITHOS_ERR_UNSUPPORTED_LAYOUT -6

/* =========================================================================
 * Quantization Modes & Sidecar Formats
 * ========================================================================= */
typedef enum {
    PITHOS_QMODE_1BIT    = 0,  /**< 1-bit sign quantization (1 bit / dimension) */
    PITHOS_QMODE_2BIT    = 1,  /**< 2-bit ternary / QJL residual quantization (2 bits / dimension) */
    PITHOS_QMODE_FLOAT32 = 2   /**< Unquantized raw 32-bit float bypass */
} pithos_quantization_mode_t;

typedef enum {
    PITHOS_SIDECAR_NONE = 0,  /**< No float sidecar (asymmetric rotated L2 fallback) */
    PITHOS_SIDECAR_FP16 = 1,  /**< IEEE 754 half-precision float sidecar (_fp16.bin, 2 B/dim) */
    PITHOS_SIDECAR_FP8  = 2,  /**< OCP/NVIDIA Blackwell FP8 E4M3 sidecar (_fp8.bin, 1 B/dim) */
    PITHOS_SIDECAR_FP4  = 3   /**< Blackwell NVFP4 E2M1 block microscaling (_fp4.bin, 0.5 B/dim + scale) */
} pithos_sidecar_mode_t;

/* =========================================================================
 * Hardware Co-Design & FPGA DMA Descriptor
 * ========================================================================= */
typedef struct {
    int32_t   tier_index;
    int32_t   tier_dimension;
    int64_t   record_count;
    uintptr_t tier_base_address;
    size_t    tier_byte_length;
    uintptr_t metadata_base_address;
    size_t    metadata_byte_length;
    uintptr_t ids_base_address;
    size_t    ids_byte_length;
    int32_t   words_per_record;
} pithos_fpga_descriptor_t;

/* =========================================================================
 * Core Database Coordinator Lifecycle
 * ========================================================================= */

/**
 * Initializes the global thread-safe Pithos database coordinator.
 * @param thread GraalVM isolate thread context.
 * @return 0 on success, negative error code on failure.
 */
int vdb_init(graal_isolatethread_t *thread);

/**
 * Closes the database coordinator and unmaps all loaded indices.
 * @param thread GraalVM isolate thread context.
 * @return 0 on success, negative error code on failure.
 */
int vdb_close(graal_isolatethread_t *thread);

/**
 * Triggers explicit garbage collection and memory compaction inside the GraalVM isolate.
 * @param thread GraalVM isolate thread context.
 * @return 0 on success, negative error code on failure.
 */
int vdb_shrink_to_fit(graal_isolatethread_t *thread);

/* =========================================================================
 * Index Loading, Metadata & Configuration
 * ========================================================================= */

/**
 * Memory-maps an existing compiled index into virtual memory off-heap.
 * @param thread GraalVM isolate thread context.
 * @param name Unique logical name identifier.
 * @param path Base filepath prefix on disk.
 * @return 0 on success, negative error code on failure.
 */
int vdb_load_index(graal_isolatethread_t *thread, const char *name, const char *path);

/**
 * Memory-maps an index with frozen projection/LoRA weights for SVD spectral energy truncation.
 * @param thread GraalVM isolate thread context.
 * @param name Unique logical name identifier.
 * @param path Base filepath prefix on disk.
 * @param weights Contiguous float array of size (D * loraDim).
 * @param lora_dim Inner bottleneck projection dimension.
 * @return 0 on success, negative error code on failure.
 */
int vdb_load_index_with_weights(graal_isolatethread_t *thread, const char *name, const char *path,
                                const float *weights, int32_t lora_dim);

/**
 * Retrieves index attributes and dimensions.
 * @param thread GraalVM isolate thread context.
 * @param name Unique logical name identifier.
 * @param out_dim Output pointer for vector dimension D.
 * @param out_size Output pointer for record count N.
 * @param out_planet_id Output pointer for planetary body ID.
 * @param out_planet_radius Output pointer for equatorial radius in meters.
 * @param out_tiers_count Output pointer for cumulative tier count.
 * @return 0 on success, negative error code on failure.
 */
int vdb_get_info(graal_isolatethread_t *thread, const char *name,
                 int32_t *out_dim, int64_t *out_size,
                 int8_t *out_planet_id, int64_t *out_planet_radius,
                 int32_t *out_tiers_count);

/**
 * Returns the total live record count N for a loaded index.
 */
int64_t vdb_size(graal_isolatethread_t *thread, const char *name);

/**
 * Returns the sidecar format mode (0=None, 1=FP16, 2=FP8, 3=FP4) for a loaded index.
 */
int vdb_get_sidecar_mode(graal_isolatethread_t *thread, const char *name);

/**
 * Unmaps and releases all virtual memory segments for the given index.
 */
int vdb_drop_index(graal_isolatethread_t *thread, const char *name);

/**
 * Sets the parallel batch scan chunk size for LMAX Disruptor ring buffer worker threads.
 */
int vdb_set_chunk_size(graal_isolatethread_t *thread, const char *name, int64_t chunk_size);

/**
 * Sets the Matryoshka cumulative spectral energy budget threshold tau in (0, 1].
 */
int vdb_set_energy_budget(graal_isolatethread_t *thread, const char *name, double tau);

/* =========================================================================
 * Nearest Neighbor Search & Resonant Voting
 * ========================================================================= */

/**
 * Performs batch k-NN search across raw continuous float query vectors.
 * @param thread GraalVM isolate thread context.
 * @param name Unique logical name identifier.
 * @param queries Contiguous float buffer of shape (num_queries * D).
 * @param num_queries Number of query vectors in the batch.
 * @param k Number of nearest neighbors to retrieve per query.
 * @param out_ids Pre-allocated int64 output buffer of shape (num_queries * k).
 * @param out_distances Pre-allocated int32 output buffer of shape (num_queries * k) (scaled by 1,000,000).
 * @return 0 on success, negative error code on failure.
 */
int vdb_batch_search(graal_isolatethread_t *thread, const char *name,
                     const float *queries, int32_t num_queries, int32_t k,
                     int64_t *out_ids, int32_t *out_distances);

/**
 * Performs multi-family resonant voting across scientific criteria.
 * @param thread GraalVM isolate thread context.
 * @param name Unique logical name identifier.
 * @param queries Contiguous float buffer of shape (num_queries * D).
 * @param query_families Int array of semantic family codes (0..7).
 * @param query_thresholds Int array of Hamming distance cutoff thresholds.
 * @param num_queries Number of queries in the batch.
 * @param voting_mask Pre-allocated byte buffer of size N bytes.
 * @return Number of resonant candidate records with >= 5 votes, or negative error code.
 */
int64_t vdb_query_planetary_grid(graal_isolatethread_t *thread, const char *name,
                                 const float *queries, const int32_t *query_families,
                                 const int32_t *query_thresholds, int32_t num_queries,
                                 uint8_t *voting_mask);

/* =========================================================================
 * Direct FPGA / DMA Hardware Co-Design Interfaces
 * ========================================================================= */

/**
 * Retrieves the raw virtual memory address and byte length of a specific index tier.
 * Enables zero-copy PCIe DMA transfer directly from off-heap virtual memory to FPGA RAM.
 */
int vdb_get_tier_address(graal_isolatethread_t *thread, const char *name, int32_t tier_idx,
                         uintptr_t *out_addr, int64_t *out_length);

/**
 * Retrieves the raw virtual memory address and byte length of the metadata sidecar buffer.
 */
int vdb_get_metadata_address(graal_isolatethread_t *thread, const char *name,
                             uintptr_t *out_addr, int64_t *out_length);

/**
 * Retrieves the raw virtual memory address and byte length of the 64-bit record IDs buffer.
 */
int vdb_get_ids_address(graal_isolatethread_t *thread, const char *name,
                        uintptr_t *out_addr, int64_t *out_length);

/**
 * Binarizes a continuous float vector using Rademacher signs preconditioning
 * and block-diagonal Fast Walsh-Hadamard rotation on host CPU before streaming to FPGA.
 * @param thread GraalVM isolate thread context.
 * @param name Logical index name (supplies sign seed and Hadamard block structure).
 * @param in_vector Continuous float vector of dimension D.
 * @param out_packed Pre-allocated 64-bit uint64 buffer of size ceil(D / 64).
 */
int vdb_transform_and_quantize(graal_isolatethread_t *thread, const char *name,
                               const float *in_vector, uint64_t *out_packed);

/* =========================================================================
 * LSM DeltaBuffer Real-Time Ingestion & Merged Search
 * ========================================================================= */

/**
 * Attaches a writable in-memory LSM DeltaBuffer for real-time inserts and soft-deletes.
 */
int vdb_create_delta_buffer(graal_isolatethread_t *thread, const char *name, int32_t flush_threshold);

/**
 * Inserts a new vector into the active DeltaBuffer.
 */
int vdb_insert(graal_isolatethread_t *thread, const char *name, int64_t record_id, const float *vector);

/**
 * Soft-deletes a record from the DeltaBuffer (sets tombstone bit).
 * @return 1 if record was present and tombstoned, 0 if not found, negative on error.
 */
int vdb_delete_from_delta(graal_isolatethread_t *thread, const char *name, int64_t record_id);

/**
 * Returns the count of live records currently in the DeltaBuffer.
 */
int64_t vdb_delta_size(graal_isolatethread_t *thread, const char *name);

/**
 * Returns 1 if the DeltaBuffer size >= flush_threshold, 0 otherwise.
 */
int vdb_needs_flush(graal_isolatethread_t *thread, const char *name);

/**
 * Executes a unified search querying both immutable base index and mutable DeltaBuffer.
 */
int vdb_search_merged(graal_isolatethread_t *thread, const char *name,
                      const float *query, int32_t k,
                      int64_t *out_ids, int32_t *out_distances);

/**
 * Serializes the DeltaBuffer to a binary backup file.
 */
int vdb_backup_delta(graal_isolatethread_t *thread, const char *name, const char *path);

/**
 * Restores a DeltaBuffer from a binary backup file.
 */
int vdb_restore_delta(graal_isolatethread_t *thread, const char *name, const char *path, int32_t flush_threshold);

/* =========================================================================
 * Index Compilation & Compaction
 * ========================================================================= */

/**
 * Compiles raw float records into a multi-tier binary columnar format on disk.
 */
int vdb_compile_index_file(graal_isolatethread_t *thread, const char *path, int8_t planet_id,
                           int64_t planet_radius, int32_t dimension,
                           const int32_t *tiers, int32_t num_tiers,
                           const int64_t *ids, const float *vectors,
                           int32_t num_records, int32_t q_mode);

/**
 * Compiles raw float records with optional FP16 sidecar generation for Stage 2 reranking.
 */
int vdb_compile_index_file_ext(graal_isolatethread_t *thread, const char *path, int8_t planet_id,
                               int64_t planet_radius, int32_t dimension,
                               const int32_t *tiers, int32_t num_tiers,
                               const int64_t *ids, const float *vectors,
                               int32_t num_records, int32_t q_mode, int32_t write_fp16);

/**
 * Compiles raw float records into a universal schema-agnostic single-file .pithos container (DIOGENES format).
 * @param thread GraalVM isolate thread context.
 * @param path Destination container filepath (e.g. "dataset.pithos").
 * @param dimension Vector dimensionality D.
 * @param tiers Matryoshka cumulative step boundaries array.
 * @param num_tiers Number of tiers (1 to 8).
 * @param ids Vector 64-bit unique IDs array.
 * @param vectors Contiguous float array (num_records * D).
 * @param num_records Total record count N.
 * @param metric_type Distance metric (0=Cosine, 1=L2, 2=DotProduct).
 * @param q_mode Quantization mode (0=1-bit, 1=2-bit QJL, 2=FP32 bypass).
 * @param sidecar_mode Precision sidecar (0=None, 1=FP16, 2=FP8 E4M3, 3=NVFP4).
 * @param metadata_payload Raw bytes of generic metadata (e.g. JSONL / Arrow IPC / binary blobs) or NULL.
 * @param metadata_len Byte length of metadata_payload or 0.
 * @param metadata_format Format descriptor string ("jsonl", "arrow", "raw") or NULL.
 * @param user_metadata_json Arbitrary key-value JSON dictionary string or NULL.
 * @return 0 on success, negative error code on failure.
 */
int vdb_compile_container(graal_isolatethread_t *thread, const char *path, int32_t dimension,
                          const int32_t *tiers, int32_t num_tiers,
                          const int64_t *ids, const float *vectors, int32_t num_records,
                          int32_t metric_type, int32_t q_mode, int32_t sidecar_mode,
                          const char *metadata_payload, int32_t metadata_len,
                          const char *metadata_format, const char *user_metadata_json);

/**
 * Compiles a single-file container inside an ephemeral GraalVM isolate,
 * guaranteeing complete memory reclamation (0 byte residual RSS) upon completion.
 */
int vdb_compile_container_isolated(const char *path, int32_t dimension,
                                   const int32_t *tiers, int32_t num_tiers,
                                   const int64_t *ids, const float *vectors, int32_t num_records,
                                   int32_t metric_type, int32_t q_mode, int32_t sidecar_mode,
                                   const char *metadata_payload, int32_t metadata_len,
                                   const char *metadata_format, const char *user_metadata_json);

/**
 * Retrieves the user metadata JSON string embedded in a loaded single-file .pithos container.
 * @param thread GraalVM isolate thread context.
 * @param name Unique logical name identifier.
 * @param out_buf Pre-allocated char buffer to receive UTF-8 JSON.
 * @param max_len Maximum capacity of out_buf.
 * @return Total bytes written, 0 if no user metadata, negative on error.
 */
int vdb_get_user_metadata(graal_isolatethread_t *thread, const char *name, char *out_buf, int32_t max_len);

/**
 * Compacts multiple compiled Pithos indices into a consolidated index file layout.
 * @param source_paths_joined Semicolon-delimited basepaths ("path1;path2;path3").
 * @param target_path Destination basepath for the consolidated index.
 */
int vdb_compact_indexes(graal_isolatethread_t *thread, const char *source_paths_joined, const char *target_path);

/* =========================================================================
 * NVIDIA CUDA Hardware Acceleration (Optional)
 * ========================================================================= */

int vdb_cuda_init(graal_isolatethread_t *thread, int32_t device_id);
int vdb_cuda_shutdown(graal_isolatethread_t *thread);
int vdb_cuda_is_available(graal_isolatethread_t *thread);
int vdb_cuda_batch_search(graal_isolatethread_t *thread, const char *name,
                          const float *queries, int32_t num_queries, int32_t k,
                          int64_t *out_ids, int32_t *out_distances);
int64_t vdb_cuda_query_planetary_grid(graal_isolatethread_t *thread, const char *name,
                                      const float *queries, const int32_t *query_families,
                                      const int32_t *query_thresholds, int32_t num_queries,
                                      uint8_t *voting_mask);

#ifdef __cplusplus
}
#endif

#endif /* PITHOS_H */
