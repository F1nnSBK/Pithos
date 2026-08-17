#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "pithos_kernels.h"

static cudaStream_t pithos_default_stream = 0;
static bool cuda_initialized = false;
static int current_device_id = -1;

int cuda_init_device(int deviceId) {
    if (cuda_initialized && current_device_id == deviceId) {
        return 0;
    }
    
    if (cuda_initialized) {
        cuda_shutdown_device();
    }
    
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        return err;
    }
    
    err = cudaStreamCreate(&pithos_default_stream);
    if (err != cudaSuccess) {
        return err;
    }
    
    cuda_initialized = true;
    current_device_id = deviceId;
    return 0;
}

int cuda_shutdown_device() {
    if (!cuda_initialized) {
        return 0;
    }
    
    cudaError_t err = cudaStreamDestroy(pithos_default_stream);
    pithos_default_stream = 0;
    cuda_initialized = false;
    current_device_id = -1;
    return err;
}

int cuda_is_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int cuda_get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

int cuda_alloc_pinned(void** ptr, size_t size) {
    if (!ptr) return cudaErrorInvalidValue;
    return cudaMallocHost(ptr, size);
}

int cuda_free_pinned(void* ptr) {
    if (!ptr) return cudaErrorInvalidValue;
    return cudaFreeHost(const_cast<void*>(ptr));
}

int cuda_alloc_device(void** ptr, size_t size) {
    if (!ptr) return cudaErrorInvalidValue;
    return cudaMalloc(ptr, size);
}

int cuda_free_device(void* ptr) {
    if (!ptr) return cudaErrorInvalidValue;
    return cudaFree(ptr);
}

int cuda_copy_to_device_async(void* dst, void* src, size_t size, cudaStream_t stream) {
    if (!dst || !src) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

int cuda_copy_from_device_async(void* dst, void* src, size_t size, cudaStream_t stream) {
    if (!dst || !src) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

int cuda_stream_synchronize(cudaStream_t stream) {
    if (!stream) stream = pithos_default_stream;
    return cudaStreamSynchronize(stream);
}

int cuda_create_stream(cudaStream_t* stream) {
    if (!stream) return cudaErrorInvalidValue;
    return cudaStreamCreate(stream);
}

int cuda_destroy_stream(cudaStream_t stream) {
    if (!stream) return cudaErrorInvalidValue;
    return cudaStreamDestroy(stream);
}

cudaStream_t cuda_get_default_stream() {
    return pithos_default_stream;
}

int cuda_is_initialized() {
    return cuda_initialized;
}

int cuda_get_current_device() {
    return current_device_id;
}

int cuda_get_device_properties(int deviceId, void* prop) {
    if (!prop) return cudaErrorInvalidValue;
    return cudaGetDeviceProperties(reinterpret_cast<cudaDeviceProp*>(prop), deviceId);
}

int cuda_get_free_memory(size_t* free, size_t* total) {
    if (!free || !total) return cudaErrorInvalidValue;
    return cudaMemGetInfo(free, total);
}
