#include "pithos_cuda.h"
#include <jni.h>

/**
 * JNI bridge for CudaNativeBindings.pithos_cuda_launch_batch_hamming.
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaNativeBindings_pithos_1cuda_1launch_1batch_1hamming(
    JNIEnv* env,
    jclass clazz,
    jlongArray deviceTierBuffers,
    jlong deviceQueries,
    jlong hostDistances,
    jint numDbVectors,
    jint numQueries,
    jint numTiers,
    jintArray tierOffsets,
    jintArray tierSizes
) {
    (void)clazz;
    jlong* buffers = (*env)->GetLongArrayElements(env, deviceTierBuffers, NULL);
    jint* offsets = (*env)->GetIntArrayElements(env, tierOffsets, NULL);
    jint* sizes = (*env)->GetIntArrayElements(env, tierSizes, NULL);
    
    int result = pithos_cuda_launch_batch_hamming(
        (const uint64_t*)buffers,
        (const uint64_t*)deviceQueries,
        (int*)hostDistances,
        numDbVectors,
        numQueries,
        numTiers,
        offsets,
        sizes
    );
    
    (*env)->ReleaseLongArrayElements(env, deviceTierBuffers, buffers, JNI_ABORT);
    (*env)->ReleaseIntArrayElements(env, tierOffsets, offsets, JNI_ABORT);
    (*env)->ReleaseIntArrayElements(env, tierSizes, sizes, JNI_ABORT);
    
    return result;
}

/**
 * JNI bridge for CudaNativeBindings.pithos_cuda_launch_voting.
 */
JNIEXPORT jint JNICALL Java_org_pithos_CudaNativeBindings_pithos_1cuda_1launch_1voting(
    JNIEnv* env,
    jclass clazz,
    jlongArray deviceTierBuffers,
    jlong deviceQueries,
    jlong deviceFamilies,
    jlong deviceThresholds,
    jlong deviceVotingMask,
    jint numDbVectors,
    jint numQueries,
    jint numFamilies,
    jint numWordsPerVector
) {
    (void)clazz;
    jlong* buffers = (*env)->GetLongArrayElements(env, deviceTierBuffers, NULL);
    
    long result = pithos_cuda_launch_voting(
        (const uint64_t*)buffers,
        (const uint64_t*)deviceQueries,
        (const int*)deviceFamilies,
        (const int*)deviceThresholds,
        (uint8_t*)deviceVotingMask,
        numDbVectors,
        numQueries,
        numFamilies,
        numWordsPerVector
    );
    
    (*env)->ReleaseLongArrayElements(env, deviceTierBuffers, buffers, JNI_ABORT);
    
    return (jint)result;
}