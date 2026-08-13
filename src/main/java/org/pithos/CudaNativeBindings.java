package org.pithos;

/// # CudaNativeBindings
///
/// Native JNI binding methods for launching CUDA batch Hamming search and multi-family resonant voting kernels.
public class CudaNativeBindings {

    /// Launches CUDA batch Hamming distance kernel across multi-tier vectors.
    static native int pithos_cuda_launch_batch_hamming(
        long[] deviceTierBuffers, long deviceQueries, long hostDistances,
        int numDbVectors, int numQueries, int numTiers, int[] tierOffsets, int[] tierSizes
    );

    /// Launches CUDA resonant voting kernel across multi-tier vectors.
    static native int pithos_cuda_launch_voting(
        long[] deviceTierBuffers, long deviceQueries, long deviceFamilies, long deviceThresholds,
        long deviceVotingMask, int numDbVectors, int numQueries, int numFamilies, int numWordsPerVector
    );
}
