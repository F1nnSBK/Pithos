#ifndef PITHOS_KERNELS_H
#define PITHOS_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#if defined(__CUDACC__) || defined(__NVCC__)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#elif defined(__has_include) && __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#else
#include <stdlib.h>
#include <string.h>

typedef void* cudaStream_t;
typedef int cudaError_t;

enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3
};

enum {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__

#ifdef __cplusplus
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};
extern "C" {
    unsigned cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void* stream = 0);
}
static struct { unsigned int x, y, z; } blockIdx, threadIdx, blockDim, gridDim;
static inline void __syncthreads() {}
static inline int __popcll(uint64_t x) { return (int)__builtin_popcountll(x); }
#endif

static inline cudaError_t cudaGetLastError(void) { return (cudaError_t)0; }
static inline cudaError_t cudaMalloc(void** devPtr, size_t size) { *devPtr = malloc(size); return (cudaError_t)0; }
static inline cudaError_t cudaFree(void* devPtr) { free(devPtr); return (cudaError_t)0; }
static inline cudaError_t cudaMallocHost(void** ptr, size_t size) { *ptr = malloc(size); return (cudaError_t)0; }
static inline cudaError_t cudaFreeHost(void* ptr) { free(ptr); return (cudaError_t)0; }
static inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, cudaStream_t stream) { memcpy(dst, src, count); (void)kind; (void)stream; return (cudaError_t)0; }
static inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; return (cudaError_t)0; }
static inline cudaError_t cudaStreamCreate(cudaStream_t* pStream) { *pStream = (void*)1; return (cudaError_t)0; }
static inline cudaError_t cudaStreamDestroy(cudaStream_t stream) { (void)stream; return (cudaError_t)0; }
static inline cudaError_t cudaSetDevice(int device) { (void)device; return (cudaError_t)0; }
static inline cudaError_t cudaDeviceReset(void) { return (cudaError_t)0; }
static inline cudaError_t cudaGetDeviceCount(int* count) { *count = 0; return (cudaError_t)0; }
static inline cudaError_t cudaGetDeviceProperties(void* prop, int device) { (void)prop; (void)device; return (cudaError_t)0; }

#endif /* !CUDA */

#define MAX_WORDS_PER_VECTOR 6
#define MAX_TIERS 8
#define MAX_FAMILIES 8

#ifdef __cplusplus
extern "C" {
#endif

int launch_batch_hamming_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_tiers,
    const int* tier_offsets,
    const int* tier_sizes,
    cudaStream_t stream
);

int launch_batch_hamming_optimized_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_words_per_vector,
    cudaStream_t stream
);

int launch_multi_family_voting_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    const int* families,
    const int* thresholds,
    uint8_t* voting_mask,
    int num_db_vectors,
    int num_queries,
    int num_families,
    int num_words_per_vector,
    cudaStream_t stream
);

int launch_walsh_hadamard_kernel(
    float* input,
    float* output,
    int num_vectors,
    int dimension,
    cudaStream_t stream
);

int launch_batch_rerank_fp8_kernel(
    const uint8_t* db_fp8_vectors,
    const float* query_vectors,
    float* out_distances,
    int num_db_vectors,
    int num_queries,
    int dimension,
    cudaStream_t stream
);

int cuda_alloc_pinned(void** ptr, size_t size);
int cuda_free_pinned(void* ptr);
int cuda_alloc_device(void** ptr, size_t size);
int cuda_free_device(void* ptr);
int cuda_copy_to_device_async(void* dst, void* src, size_t size, cudaStream_t stream);
int cuda_copy_from_device_async(void* dst, void* src, size_t size, cudaStream_t stream);
int cuda_stream_synchronize(cudaStream_t stream);
int cuda_create_stream(cudaStream_t* stream);
int cuda_destroy_stream(cudaStream_t stream);
int cuda_init_device(int deviceId);
int cuda_shutdown_device();
int cuda_is_available();
int cuda_get_device_count();
int cuda_get_device_properties(int deviceId, void* prop);

#ifdef __cplusplus
}
#endif

#endif
