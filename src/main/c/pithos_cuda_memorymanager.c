#include "pithos_cuda.h"
#include <jni.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/**
 * Allocates page-locked pinned host memory.
 */
JNIEXPORT jlong JNICALL Java_org_pithos_CudaMemoryManager_allocPinned(
    JNIEnv* env, 
    jclass clazz,
    jlong size
) {
    (void)env;
    (void)clazz;
    void* ptr = NULL;
    int result = pithos_cuda_alloc_pinned(&ptr, (size_t)size);
    return result == 0 ? (jlong)ptr : 0;
}

/**
 * Frees page-locked pinned host memory.
 */
JNIEXPORT void JNICALL Java_org_pithos_CudaMemoryManager_freePinned(
    JNIEnv* env, 
    jclass clazz,
    jlong pointer
) {
    (void)env;
    (void)clazz;
    if (pointer != 0) {
        pithos_cuda_free_pinned((void*)pointer);
    }
}

/**
 * Allocates device memory in GPU VRAM.
 */
JNIEXPORT jlong JNICALL Java_org_pithos_CudaMemoryManager_allocDevice(
    JNIEnv* env, 
    jclass clazz,
    jlong size
) {
    (void)env;
    (void)clazz;
    void* ptr = NULL;
    int result = pithos_cuda_alloc_device(&ptr, (size_t)size);
    return result == 0 ? (jlong)ptr : 0;
}

/**
 * Frees device memory in GPU VRAM.
 */
JNIEXPORT void JNICALL Java_org_pithos_CudaMemoryManager_freeDevice(
    JNIEnv* env, 
    jclass clazz,
    jlong pointer
) {
    (void)env;
    (void)clazz;
    if (pointer != 0) {
        pithos_cuda_free_device((void*)pointer);
    }
}

/**
 * Copies bytes synchronously from host to GPU device.
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_copyToDevice(
    JNIEnv* env, 
    jclass clazz,
    jlong dst,
    jlong src,
    jlong size
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_copy_to_device((void*)dst, (void*)src, (size_t)size);
}

/**
 * Copies bytes synchronously from GPU device to host.
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_copyFromDevice(
    JNIEnv* env, 
    jclass clazz,
    jlong dst,
    jlong src,
    jlong size
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_copy_from_device((void*)dst, (void*)src, (size_t)size);
}

/**
 * Creates an asynchronous CUDA stream.
 */
JNIEXPORT jlong JNICALL Java_org_pithos_CudaMemoryManager_createStream(
    JNIEnv* env, 
    jclass clazz
) {
    (void)env;
    (void)clazz;
    cudaStream_t* stream = malloc(sizeof(cudaStream_t));
    if (!stream) return 0;
    if (cudaStreamCreate(stream) != cudaSuccess) {
        free(stream);
        return 0;
    }
    return (jlong)stream;
}

/**
 * Destroys a CUDA stream.
 */
JNIEXPORT void JNICALL Java_org_pithos_CudaMemoryManager_destroyStream(
    JNIEnv* env, 
    jclass clazz,
    jlong stream
) {
    (void)env;
    (void)clazz;
    if (stream != 0) {
        cudaStreamDestroy(*(cudaStream_t*)stream);
        free((void*)stream);
    }
}

/**
 * Retrieves the raw memory pointer for a direct ByteBuffer.
 */
JNIEXPORT jlong JNICALL Java_org_pithos_CudaMemoryManager_getDirectBufferAddress(
    JNIEnv* env, 
    jclass clazz,
    jobject buffer
) {
    (void)clazz;
    return (jlong)(*env)->GetDirectBufferAddress(env, buffer);
}

/**
 * Asynchronously copies bytes from host to GPU device.
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_copyToDeviceAsync(
    JNIEnv* env, 
    jclass clazz,
    jlong dst,
    jlong src,
    jlong size,
    jlong stream
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_copy_to_device_async((void*)dst, (void*)src, (size_t)size, (cudaStream_t*)stream);
}

/**
 * Asynchronously copies bytes from GPU device to host.
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_copyFromDeviceAsync(
    JNIEnv* env, 
    jclass clazz,
    jlong dst,
    jlong src,
    jlong size,
    jlong stream
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_copy_from_device_async((void*)dst, (void*)src, (size_t)size, (cudaStream_t*)stream);
}

/**
 * Synchronizes the specified CUDA stream.
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_streamSynchronize(
    JNIEnv* env, 
    jclass clazz,
    jlong stream
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_stream_synchronize((cudaStream_t*)stream);
}