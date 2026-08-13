package org.pithos;

import java.nio.ByteBuffer;

/// # CudaMemoryManager
///
/// Native JNI interface managing CUDA device memory allocations, page-locked (pinned) host memory,
/// and asynchronous multi-stream memory transfers.
public class CudaMemoryManager {

    private static final int DEFAULT_STREAM_COUNT = 4;

    private final long[] streams;
    private final long pinnedBuffer;
    private final long deviceBuffer;
    private final long bufferSize;

    /// Initializes stream pool, page-locked pinned host buffer, and device memory buffer.
    ///
    /// @param bufferSize byte size of managed memory buffers
    public CudaMemoryManager(long bufferSize) {
        this.bufferSize = bufferSize;
        this.streams = new long[DEFAULT_STREAM_COUNT];
        for (int i = 0; i < streams.length; i++) {
            streams[i] = createStream();
        }
        this.pinnedBuffer = allocPinned(bufferSize);
        this.deviceBuffer = allocDevice(bufferSize);
    }

    /// Allocates page-locked (pinned) host memory for DMA transfers.
    public static native long allocPinned(long size);

    /// Frees allocated page-locked host memory.
    public static native void freePinned(long pointer);

    /// Allocates device (GPU VRAM) memory.
    public static native long allocDevice(long size);

    /// Frees allocated device memory.
    public static native void freeDevice(long pointer);

    /// Synchronously copies bytes from host to device memory.
    public static native int copyToDevice(long dst, long src, long size);

    /// Synchronously copies bytes from device to host memory.
    public static native int copyFromDevice(long dst, long src, long size);

    /// Creates an asynchronous CUDA stream.
    public static native long createStream();

    /// Destroys a CUDA stream.
    public static native void destroyStream(long stream);

    /// Enqueues asynchronous host-to-device transfer on the specified stream.
    public void asyncTransferToDevice(ByteBuffer hostBuffer, long devicePtr, int streamIndex) {
        long size = hostBuffer.remaining();
        long hostPtr = getDirectBufferAddress(hostBuffer);
        copyToDeviceAsync(devicePtr, hostPtr, size, streams[streamIndex]);
    }

    /// Enqueues asynchronous device-to-host transfer on the specified stream.
    public void asyncTransferFromDevice(long hostPtr, long devicePtr, long size, int streamIndex) {
        copyFromDeviceAsync(hostPtr, devicePtr, size, streams[streamIndex]);
    }

    /// Synchronizes the specified CUDA stream, blocking until pending operations complete.
    public void synchronizeStream(int streamIndex) {
        streamSynchronize(streams[streamIndex]);
    }

    /// Returns the virtual address of the pinned host buffer.
    public long getPinnedBuffer() {
        return pinnedBuffer;
    }

    /// Returns the virtual address of the device buffer in GPU VRAM.
    public long getDeviceBuffer() {
        return deviceBuffer;
    }

    /// Returns the native handle of the specified CUDA stream.
    public long getStream(int index) {
        return streams[index];
    }

    /// Destroys all streams and frees allocated buffers.
    public void shutdown() {
        for (long stream : streams) {
            destroyStream(stream);
        }
        freePinned(pinnedBuffer);
        freeDevice(deviceBuffer);
    }

    /// Retrieves raw pointer address for a direct `ByteBuffer`.
    public static native long getDirectBufferAddress(ByteBuffer buffer);

    private static native int copyToDeviceAsync(long dst, long src, long size, long stream);
    private static native int copyFromDeviceAsync(long dst, long src, long size, long stream);
    private static native int streamSynchronize(long stream);
}
