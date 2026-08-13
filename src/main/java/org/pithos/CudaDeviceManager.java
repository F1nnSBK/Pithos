package org.pithos;

/// # CudaDeviceManager
///
/// Native JNI interface for CUDA device initialization, query, and shutdown.
public class CudaDeviceManager {

    /// Initializes CUDA runtime on the specified device.
    ///
    /// @param deviceId zero-indexed CUDA device ID
    /// @return `0` on success, non-zero error code on failure
    public static native int initialize(int deviceId);

    /// Releases CUDA context and shuts down runtime.
    public static native int shutdown();

    /// Checks if a compatible CUDA-capable GPU runtime is available.
    ///
    /// @return `1` if available, `0` otherwise
    public static native int isAvailable();

    /// Returns the number of CUDA-capable devices detected by the driver.
    public static native int getDeviceCount();
}
