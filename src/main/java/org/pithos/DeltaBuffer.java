package org.pithos;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/// # DeltaBuffer
///
/// Log-Structured Merge (LSM) in-memory write buffer with Write-Ahead Log (WAL) persistence for real-time inserts.
///
/// ### Architecture & Lifecycle:
/// 1. **Real-time Ingestion:** Newly inserted vectors land in this in-memory flat buffer without mutating immutable base `.pithos` files.
/// 2. **Durability (WAL):** Each insert and delete operation is appended synchronously to a binary WAL file on disk:
///    - Insert record: `[byte type=1][long id][float[D] vector]`
///    - Delete tombstone: `[byte type=2][long id]`
/// 3. **Unified Hybrid Search:** Searches query both the base memory-mapped index and the `DeltaBuffer` concurrently, merging
///    and deduplicating results.
/// 4. **Exact $L_2$ Distance Evaluation:** Queries against the delta buffer evaluate unquantized Euclidean distance:
///    $$d(\mathbf{q}, \mathbf{x}) = \sum_{d=0}^{D-1} (q_d - x_d)^2$$
/// 5. **Flush & Compaction:** When `liveSize() >= flushThreshold`, the buffer can be drained and merged into the base index.
public class DeltaBuffer {

    /// A single buffered entry containing record ID, raw float vector, and tombstone status.
    private record BufferEntry(long id, float[] vector, boolean tombstone) {}

    private final int dimension;
    private final int flushThreshold;
    private final String walPath;
    private FileChannel walChannel;

    /// Ordered list of inserted entries (append-only, tombstones included).
    private final List<BufferEntry> entries;

    /// Count of live (non-tombstoned) entries.
    private final AtomicInteger liveCount = new AtomicInteger(0);

    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /// Constructs a `DeltaBuffer` without persistent WAL logging.
    ///
    /// @param dimension vector dimensionality ($D$)
    /// @param flushThreshold soft limit on live entries before flush is recommended
    public DeltaBuffer(int dimension, int flushThreshold) {
        this(dimension, flushThreshold, null);
    }

    /// Constructs a `DeltaBuffer` with optional Write-Ahead Log (WAL) backing.
    ///
    /// @param dimension vector dimensionality ($D$)
    /// @param flushThreshold soft limit on live entries before flush is recommended
    /// @param walPath filepath for the WAL file (or `null` for in-memory only)
    public DeltaBuffer(int dimension, int flushThreshold, String walPath) {
        this.dimension = dimension;
        this.flushThreshold = flushThreshold;
        this.walPath = walPath;
        this.entries = new CopyOnWriteArrayList<>();
        if (walPath != null) {
            try {
                Path path = Path.of(walPath);
                boolean exists = Files.exists(path);
                this.walChannel = FileChannel.open(path,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.READ);
                if (exists && walChannel.size() > 0) {
                    replayWal();
                }
            } catch (IOException e) {
                throw new RuntimeException("Failed to initialize WAL log: " + walPath, e);
            }
        }
    }

    /// Inserts a new vector record into the delta buffer and appends it to the WAL.
    ///
    /// @param id unique 64-bit record identifier
    /// @param vector raw continuous float vector of length `dimension`
    public void insert(long id, float[] vector) {
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "Vector dimension mismatch: expected " + dimension + ", got " + vector.length);
        }
        lock.writeLock().lock();
        try {
            if (walChannel != null) {
                writeInsertToWal(id, vector);
            }
            entries.add(new BufferEntry(id, vector.clone(), false));
            liveCount.incrementAndGet();
        } catch (IOException e) {
            throw new RuntimeException("Failed to write insert to WAL", e);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /// Soft-deletes a record by writing a tombstone.
    ///
    /// @param id unique record identifier
    /// @return `true` if at least one active entry was tombstoned
    public boolean delete(long id) {
        lock.writeLock().lock();
        try {
            boolean found = false;
            for (int i = 0; i < entries.size(); i++) {
                BufferEntry e = entries.get(i);
                if (e.id() == id && !e.tombstone()) {
                    entries.set(i, new BufferEntry(e.id(), e.vector(), true));
                    liveCount.decrementAndGet();
                    found = true;
                }
            }
            if (found && walChannel != null) {
                writeDeleteToWal(id);
            }
            return found;
        } catch (IOException e) {
            throw new RuntimeException("Failed to write delete to WAL", e);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /// Returns the number of active (non-tombstoned) records.
    public int liveSize() {
        return liveCount.get();
    }

    /// Returns the total entry count including tombstones.
    public int totalSize() {
        return entries.size();
    }

    /// Returns `true` if the live entry count has reached or exceeded the configured flush threshold.
    public boolean needsFlush() {
        return liveCount.get() >= flushThreshold;
    }

    /// Searches the delta buffer for the top $k$ nearest neighbors to the query vector
    /// using exact Euclidean $L_2$ distance without quantization.
    ///
    /// @param query raw float query vector
    /// @param k number of neighbors to retrieve
    /// @return list of nearest neighbors sorted ascending by score
    public List<Index.SearchResult> searchKnn(float[] query, int k) {
        if (k <= 0 || liveCount.get() == 0) {
            return List.of();
        }
        // Max-heap of size k keyed by distance bits for O(log k) eviction of worst candidate
        PriorityQueue<long[]> heap = new PriorityQueue<>(
                (a, b) -> Long.compare(b[0], a[0]));

        lock.readLock().lock();
        try {
            for (BufferEntry e : entries) {
                if (e.tombstone()) continue;
                float dist = exactL2(query, e.vector());
                long distBits = Float.floatToRawIntBits(dist) & 0xFFFFFFFFL;
                if (heap.size() < k) {
                    heap.offer(new long[]{distBits, e.id()});
                } else if (distBits < heap.peek()[0]) {
                    heap.poll();
                    heap.offer(new long[]{distBits, e.id()});
                }
            }
        } finally {
            lock.readLock().unlock();
        }

        List<Index.SearchResult> results = new ArrayList<>(heap.size());
        while (!heap.isEmpty()) {
            long[] entry = heap.poll();
            float d = Float.intBitsToFloat((int) entry[0]);
            results.add(new Index.SearchResult(entry[1], (int) (d * 1_000_000f)));
        }
        results.sort((a, b) -> Integer.compare(a.score(), b.score()));
        return results;
    }

    private static float exactL2(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    /// Serializes all live entries to a binary backup file.
    ///
    /// ### Binary Layout:
    /// ```text
    /// [int] dimension
    /// [int] num_live_entries
    /// for each entry:
    ///   [long] id
    ///   [float[dimension]] vector
    /// ```
    ///
    /// @param path target destination filepath
    /// @throws IOException on I/O failure
    public void serializeToPath(String path) throws IOException {
        lock.readLock().lock();
        try (DataOutputStream out = new DataOutputStream(
                Files.newOutputStream(Path.of(path)))) {
            List<BufferEntry> snapshot = new ArrayList<>();
            for (BufferEntry e : entries) {
                if (!e.tombstone()) {
                    snapshot.add(e);
                }
            }
            out.writeInt(dimension);
            out.writeInt(snapshot.size());
            for (BufferEntry e : snapshot) {
                out.writeLong(e.id());
                for (float v : e.vector()) {
                    out.writeFloat(v);
                }
            }
        } finally {
            lock.readLock().unlock();
        }
    }

    /// Deserializes a `DeltaBuffer` from a previously serialized binary file.
    ///
    /// @param path path to the backup file
    /// @param flushThreshold flush threshold for the restored buffer
    /// @return restored `DeltaBuffer`
    /// @throws IOException on I/O failure
    public static DeltaBuffer deserializeFromPath(String path, int flushThreshold) throws IOException {
        try (DataInputStream in = new DataInputStream(
                Files.newInputStream(Path.of(path)))) {
            int dim = in.readInt();
            int numEntries = in.readInt();
            DeltaBuffer buf = new DeltaBuffer(dim, flushThreshold);
            for (int i = 0; i < numEntries; i++) {
                long id = in.readLong();
                float[] vec = new float[dim];
                for (int d = 0; d < dim; d++) {
                    vec[d] = in.readFloat();
                }
                buf.insert(id, vec);
            }
            return buf;
        }
    }

    /// Drains and returns all live entries, clearing the buffer and truncating the WAL.
    ///
    /// @return list of live vector records
    public List<VectorRecord> drainLiveEntries() {
        lock.writeLock().lock();
        try {
            List<VectorRecord> result = new ArrayList<>();
            for (BufferEntry e : entries) {
                if (!e.tombstone()) {
                    result.add(new VectorRecord(e.id(), e.vector()));
                }
            }
            entries.clear();
            liveCount.set(0);
            if (walChannel != null) {
                try {
                    walChannel.truncate(0);
                    walChannel.force(false);
                } catch (IOException e) {
                    // Ignore truncation errors
                }
            }
            return result;
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void replayWal() throws IOException {
        long size = walChannel.size();
        ByteBuffer buffer = ByteBuffer.allocate(9 + dimension * 4);
        walChannel.position(0);
        while (walChannel.position() < size) {
            buffer.clear();
            buffer.limit(9);
            int read = walChannel.read(buffer);
            if (read < 9) break;
            buffer.flip();
            byte type = buffer.get();
            long id = buffer.getLong();

            if (type == 1) {
                buffer.clear();
                buffer.limit(dimension * 4);
                read = walChannel.read(buffer);
                if (read < dimension * 4) break;
                buffer.flip();
                float[] vec = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    vec[d] = buffer.getFloat();
                }
                entries.add(new BufferEntry(id, vec, false));
                liveCount.incrementAndGet();
            } else if (type == 2) {
                for (int i = 0; i < entries.size(); i++) {
                    BufferEntry e = entries.get(i);
                    if (e.id() == id && !e.tombstone()) {
                        entries.set(i, new BufferEntry(e.id(), e.vector(), true));
                        liveCount.decrementAndGet();
                    }
                }
            }
        }
        walChannel.position(size);
    }

    private void writeInsertToWal(long id, float[] vector) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(9 + dimension * 4);
        bb.put((byte) 1);
        bb.putLong(id);
        for (float v : vector) {
            bb.putFloat(v);
        }
        bb.flip();
        while (bb.hasRemaining()) {
            walChannel.write(bb);
        }
        walChannel.force(false);
    }

    private void writeDeleteToWal(long id) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(9);
        bb.put((byte) 2);
        bb.putLong(id);
        bb.flip();
        while (bb.hasRemaining()) {
            walChannel.write(bb);
        }
        walChannel.force(false);
    }

    /// Closes the delta buffer and releases any open WAL file channels.
    public void close() {
        lock.writeLock().lock();
        try {
            if (walChannel != null) {
                walChannel.close();
                walChannel = null;
            }
        } catch (IOException e) {
            // Ignore
        } finally {
            lock.writeLock().unlock();
        }
    }
}
