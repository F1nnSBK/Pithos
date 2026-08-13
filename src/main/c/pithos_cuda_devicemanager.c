#include "pithos_cuda.h"
#include <jni.h>

/**
 * JNI wrapper for CudaDeviceManager.initialize(deviceId).
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaDeviceManager_initialize(
    JNIEnv* env, 
    jclass clazz,
    jint deviceId
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_init(deviceId);
}

/**
 * JNI wrapper for CudaDeviceManager.shutdown().
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaDeviceManager_shutdown(
    JNIEnv* env, 
    jclass clazz
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_shutdown();
}

/**
 * JNI wrapper for CudaDeviceManager.isAvailable().
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaDeviceManager_isAvailable(
    JNIEnv* env, 
    jclass clazz
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_is_available();
}

/**
 * JNI wrapper for CudaDeviceManager.getDeviceCount().
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaDeviceManager_getDeviceCount(
    JNIEnv* env, 
    jclass clazz
) {
    (void)env;
    (void)clazz;
    return pithos_cuda_get_device_count();
}