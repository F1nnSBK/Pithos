#ifndef PITHOS_CUDA_H
#define PITHOS_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes the CUDA runtime context on the target GPU device.
 *
 * @param deviceId zero-indexed CUDA device identifier
 * @return 0 on success, or CUDA error code
 */
int pithos_cuda_init(int deviceId);

/**
 * Releases CUDA streams and shuts down device context.
 *
 * @return 0 on success, or CUDA error code
 */
int pithos_cuda_shutdown(void);

/**
 * Checks whether a CUDA-capable device is available and accessible.
 *
 * @return 1 if available, 0 otherwise
 */
int pithos_cuda_is_available(void);

/**
 * Returns the count of CUDA-capable GPU devices detected by the driver.
 *
 * @return device count
 */
int pithos_cuda_get_device_count(void);

/**
 * Allocates page-locked (pinned) host memory for high-throughput DMA transfers.
 */
int pithos_cuda_alloc_pinned(void** ptr, size_t size);

/**
 * Frees page-locked (pinned) host memory.
 */
int pithos_cuda_free_pinned(void* ptr);

/**
 * Allocates device memory in GPU VRAM.
 */
int pithos_cuda_alloc_device(void** ptr, size_t size);

/**
 * Frees device memory in GPU VRAM.
 */
int pithos_cuda_free_device(void* ptr);

/**
 * Synchronously copies memory from host to GPU device.
 */
int pithos_cuda_copy_to_device(void* dst, void* src, size_t size);

/**
 * Synchronously copies memory from GPU device to host.
 */
int pithos_cuda_copy_from_device(void* dst, void* src, size_t size);

/**
 * Launches batch Hamming distance kernel across multi-tier binary vectors.
 *
 * @param db_vectors pointer to GPU device memory containing database vectors
 * @param query_vectors pointer to GPU device memory containing query vectors
 * @param distances output pointer in host/device memory for computed distances
 * @param num_db_vectors total count of database vectors (N)
 * @param num_queries number of query vectors in batch
 * @param num_tiers number of cumulative search tiers
 * @param tier_offsets byte/word offsets per tier
 * @param tier_sizes word lengths per tier
 * @return 0 on success, or CUDA error code
 */
int pithos_cuda_launch_batch_hamming(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_tiers,
    const int* tier_offsets,
    const int* tier_sizes
);

/**
 * Launches multi-family resonant voting kernel across multi-tier binary vectors.
 *
 * @param db_vectors pointer to GPU device memory containing database vectors
 * @param query_vectors pointer to GPU device memory containing query vectors
 * @param families array of semantic family identifiers (0..7)
 * @param thresholds array of distance cutoff thresholds
 * @param voting_mask output bitmask buffer of size N bytes
 * @param num_db_vectors total database records (N)
 * @param num_queries number of query vectors
 * @param num_families number of distinct semantic families
 * @param num_words_per_vector number of 64-bit words per vector
 * @return 0 on success, or CUDA error code
 */
int pithos_cuda_launch_voting(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    const int* families,
    const int* thresholds,
    uint8_t* voting_mask,
    int num_db_vectors,
    int num_queries,
    int num_families,
    int num_words_per_vector
);

/**
 * Asynchronously copies memory from host to GPU device on the specified CUDA stream.
 */
int pithos_cuda_copy_to_device_async(void* dst, void* src, size_t size, void* stream);

/**
 * Asynchronously copies memory from GPU device to host on the specified CUDA stream.
 */
int pithos_cuda_copy_from_device_async(void* dst, void* src, size_t size, void* stream);

/**
 * Blocks CPU execution until all operations in the specified CUDA stream complete.
 */
int pithos_cuda_stream_synchronize(void* stream);

#ifdef __cplusplus
}
#endif

#endif /* PITHOS_CUDA_H */
