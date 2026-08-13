package org.pithos;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/// # VectorDb
///
/// Coordinate layer managing active multi-tier `Index` instances, LSM `DeltaBuffer` states, index compilation, and compaction.
///
/// ### Binary Database Layout:
/// A compiled Pithos index consists of the following contiguous binary columnar files:
/// 1. **`<basePath>` (64-byte Header):**
///    - Offset 0..3: Magic ASCII bytes `'P'`, `'L'`, `'A'`, `'N'`
///    - Offset 4: `planetId` (1 byte, e.g. 1 for Moon, 2 for Mars)
///    - Offset 5..12: Total record count $N$ (8-byte unaligned long)
///    - Offset 13..20: Equatorial radius $R$ in meters (8-byte unaligned long)
///    - Offset 21..24: Vector dimension $D$ (4-byte unaligned int)
///    - Offset 25..28: Cumulative tier count $T$ (4-byte unaligned int, $1 \le T \le 8$)
///    - Offset 29..60: Cumulative dimension boundaries for each tier (up to 8 ints)
///    - Offset 61: Quantization mode `qMode` (0 = 1-bit, 1 = 2-bit ternary, 2 = float32 bypass)
/// 2. **`<basePath>_ids.bin` ($N \times 8$ bytes):** 64-bit record IDs.
/// 3. **`<basePath>_metadata.bin` ($N \times 8$ bytes):** Bitmask flags and tombstones (bit 0 = tombstone).
/// 4. **`<basePath>_tier_k.bin` ($N \times \text{bytesPerRecord}_k$ bytes):** Binarized columnar vectors.
/// 5. **`<basePath>_fp16.bin` ($N \times D \times 2$ bytes, optional):** Half-precision IEEE 754 raw floats for in-engine Stage 2 reranking.
public class VectorDb {
    private final Map<String, Index> indices;
    private final Map<String, DeltaBuffer> deltaBuffers;
    private final Map<String, String> indexPaths;

    /// Initializes a new thread-safe `VectorDb` instance.
    public VectorDb() {
        this.indices = new ConcurrentHashMap<>();
        this.deltaBuffers = new ConcurrentHashMap<>();
        this.indexPaths = new ConcurrentHashMap<>();
    }

    /// Maps an existing multi-tier index off-heap and registers it under the given logical name.
    ///
    /// @param name logical index registration name
    /// @param basePath base filepath of the compiled index on disk
    /// @param weights optional projection/LoRA weights of size $D \times D_0$
    /// @param loraDim bottleneck dimension $D_0$
    /// @return registered `Index` instance
    /// @throws IOException on I/O error
    public Index loadIndex(String name, String basePath, float[] weights, int loraDim) throws IOException {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Index name cannot be empty");
        }
        Index index = FlatIndex.mapFile(basePath, weights, loraDim);
        indices.put(name, index);
        indexPaths.put(name, basePath);
        return index;
    }

    /// Returns the registered index with the specified logical name, or `null` if not found.
    public Index getIndex(String name) {
        return indices.get(name);
    }

    /// Unmaps and closes an index, freeing its off-heap memory and attached delta buffer.
    ///
    /// @param name logical index name
    /// @return `true` if the index was found and dropped, `false` otherwise
    public boolean dropIndex(String name) {
        DeltaBuffer buf = deltaBuffers.remove(name);
        if (buf != null) {
            buf.close();
        }
        indexPaths.remove(name);
        Index index = indices.remove(name);
        if (index != null) {
            try {
                index.close();
            } catch (Exception e) {
                // ignore
            }
        }
        return index != null;
    }

    /// Closes all active indices and delta buffers, releasing all off-heap resources.
    public void close() {
        for (Index index : indices.values()) {
            try {
                index.close();
            } catch (Exception e) {
                // ignore
            }
        }
        for (DeltaBuffer buf : deltaBuffers.values()) {
            buf.close();
        }
        indices.clear();
        deltaBuffers.clear();
        indexPaths.clear();
    }

    // -------------------------------------------------------------------------
    // Delta Buffer API (LSM layer)
    // -------------------------------------------------------------------------

    /// Creates an in-memory `DeltaBuffer` attached to the named index.
    ///
    /// @param indexName name of the registered base index
    /// @param flushThreshold soft limit on live entries before flush is recommended
    /// @return new `DeltaBuffer`
    /// @throws IllegalArgumentException if the index is not registered
    public DeltaBuffer createDeltaBuffer(String indexName, int flushThreshold) {
        Index index = indices.get(indexName);
        if (index == null)
            throw new IllegalArgumentException("Unknown index: " + indexName);
        String basePath = indexPaths.get(indexName);
        String walPath = (basePath != null) ? basePath + "_wal.bin" : null;
        DeltaBuffer buf = new DeltaBuffer(index.getDimension(), flushThreshold, walPath);
        deltaBuffers.put(indexName, buf);
        return buf;
    }

    /// Returns the active `DeltaBuffer` for the given index, or `null` if none exists.
    public DeltaBuffer getDeltaBuffer(String indexName) {
        return deltaBuffers.get(indexName);
    }

    /// Inserts a vector into the delta buffer attached to the given index.
    ///
    /// @param indexName name of the target index
    /// @param id unique record ID
    /// @param vector raw float vector
    /// @return `true` on success, `false` if no delta buffer is registered
    public boolean insertIntoDelta(String indexName, long id, float[] vector) {
        DeltaBuffer buf = deltaBuffers.get(indexName);
        if (buf == null)
            return false;
        buf.insert(id, vector);
        return true;
    }

    /// Soft-deletes a record from the delta buffer (tombstone).
    ///
    /// @param indexName target index name
    /// @param id record ID to tombstone
    /// @return `true` if at least one entry was tombstoned
    public boolean deleteFromDelta(String indexName, long id) {
        DeltaBuffer buf = deltaBuffers.get(indexName);
        if (buf == null)
            return false;
        return buf.delete(id);
    }

    /// Backs up all live entries from the delta buffer to a binary file.
    ///
    /// @param indexName name of the index
    /// @param path destination filepath
    /// @throws IOException on I/O failure
    /// @throws IllegalStateException if no delta buffer is attached
    public void backupDelta(String indexName, String path) throws IOException {
        DeltaBuffer buf = deltaBuffers.get(indexName);
        if (buf == null)
            throw new IllegalStateException("No delta buffer for index: " + indexName);
        buf.serializeToPath(path);
    }

    /// Restores a delta buffer from a previously serialized binary file.
    ///
    /// @param indexName target index name
    /// @param path path to the backup file
    /// @param flushThreshold flush threshold for the restored buffer
    /// @throws IOException on I/O failure
    public void restoreDelta(String indexName, String path, int flushThreshold) throws IOException {
        DeltaBuffer buf = DeltaBuffer.deserializeFromPath(path, flushThreshold);
        deltaBuffers.put(indexName, buf);
    }

    /// Performs a unified search querying both the immutable base index and the mutable `DeltaBuffer`,
    /// merging and deduplicating results to return the combined top-$k$.
    ///
    /// @param indexName name of the index
    /// @param query raw float query vector
    /// @param k nearest neighbor count
    /// @return merged top-$k$ search results
    public List<Index.SearchResult> searchMerged(String indexName, float[] query, int k) {
        Index index = indices.get(indexName);
        if (index == null)
            throw new IllegalArgumentException("Unknown index: " + indexName);

        List<Index.SearchResult> baseResults = index.search(query, k);

        DeltaBuffer buf = deltaBuffers.get(indexName);
        if (buf == null || buf.liveSize() == 0) {
            return baseResults;
        }

        List<Index.SearchResult> deltaResults = buf.searchKnn(query, k);

        // Merge and deduplicate by ID, then take top-K by score
        Map<Long, Index.SearchResult> merged = new LinkedHashMap<>();
        for (Index.SearchResult r : baseResults)
            merged.put(r.id(), r);
        for (Index.SearchResult r : deltaResults)
            merged.putIfAbsent(r.id(), r);

        return merged.values().stream()
                .sorted((a, b) -> Integer.compare(a.score(), b.score()))
                .limit(k)
                .toList();
    }

    /// Compiles raw float records into a multi-tier, cache-aligned database file layout (1-bit default).
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records) throws IOException {
        compileIndexFile(basePath, planetId, planetRadius, dimension, tiers, records, 0);
    }

    /// Compiles raw float records into a multi-tier, cache-aligned database file layout with configurable quantization mode.
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records, int qMode) throws IOException {
        compileIndexFile(basePath, planetId, planetRadius, dimension, tiers, records, qMode, true);
    }

    /// Compiles raw continuous float vector records into a multi-tier binary columnar format on disk.
    ///
    /// @param basePath base filepath prefix
    /// @param planetId target planetary body identifier code
    /// @param planetRadius equatorial radius in meters
    /// @param dimension vector dimensionality ($D$)
    /// @param tiers cumulative Matryoshka tier step boundaries (up to 8 tiers)
    /// @param records input vector records
    /// @param qMode quantization mode: 0 = 1-bit, 1 = 2-bit ternary, 2 = float32 bypass
    /// @param writeFp16 whether to generate the `_fp16.bin` sidecar file for Stage 2 reranking
    /// @throws IOException on I/O failure
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records, int qMode, boolean writeFp16) throws IOException {
        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException("Records list cannot be null or empty");
        }
        if (tiers == null || tiers.length == 0 || tiers.length > 8) {
            throw new IllegalArgumentException("Tiers must have between 1 and 8 step boundaries");
        }

        long totalRecords = records.size();

        // 1. Write base .pithos config file containing the 64-byte PLAN header
        Path mainPath = Path.of(basePath);
        try (FileChannel channel = FileChannel.open(mainPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, 64, Arena.global());

            // Magic bytes
            mapped.set(ValueLayout.JAVA_BYTE, 0, (byte) 'P');
            mapped.set(ValueLayout.JAVA_BYTE, 1, (byte) 'L');
            mapped.set(ValueLayout.JAVA_BYTE, 2, (byte) 'A');
            mapped.set(ValueLayout.JAVA_BYTE, 3, (byte) 'N');
            mapped.set(ValueLayout.JAVA_BYTE, 4, planetId);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 5, totalRecords);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 13, planetRadius);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 21, dimension);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 25, tiers.length);
            for (int i = 0; i < tiers.length; i++) {
                mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 29 + (i * 4), tiers[i]);
            }
            // Write qMode to offset 61
            mapped.set(ValueLayout.JAVA_BYTE, 61, (byte) qMode);
            mapped.force();
        }

        // 2. Write IDs file
        Path idsPath = Path.of(basePath + "_ids.bin");
        try (FileChannel channel = FileChannel.open(idsPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalRecords * 8, Arena.global());
            for (int i = 0; i < totalRecords; i++) {
                mapped.set(ValueLayout.JAVA_LONG, i * 8L, records.get(i).id());
            }
            mapped.force();
        }

        // 3. Write Metadata file (tombstones & attributes, default value is 2 for all-active)
        Path metadataPath = Path.of(basePath + "_metadata.bin");
        try (FileChannel channel = FileChannel.open(metadataPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalRecords * 8, Arena.global());
            for (int i = 0; i < totalRecords; i++) {
                mapped.set(ValueLayout.JAVA_LONG, i * 8L, 2L);
            }
            mapped.force();
        }

        // 4. Transform, Binarize, and Write Tier files
        TransformOperator transformer = new TransformOperator(dimension, tiers);
        int numTiers = tiers.length;

        int[] tierLongs = new int[numTiers];
        FileChannel[] tierChannels = new FileChannel[numTiers];
        MemorySegment[] tierMappeds = new MemorySegment[numTiers];

        int prevBound = 0;
        for (int k = 0; k < numTiers; k++) {
            int width = tiers[k] - prevBound;
            tierLongs[k] = width / 64;
            prevBound = tiers[k];

            Path tierPath = Path.of(basePath + "_tier_" + k + ".bin");
            tierChannels[k] = FileChannel.open(tierPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.READ,
                    StandardOpenOption.TRUNCATE_EXISTING);
            long bytesPerRecord = switch (qMode) {
                case 1 -> (width / 4); // 2-bit: 2 bits/dim -> width/4 bytes
                case 2 -> (width * 4L); // Float-Hybrid: raw float32 -> 4 bytes/dim
                default -> (width / 8); // 1-bit: 1 bit/dim -> width/8 bytes
            };
            tierMappeds[k] = tierChannels[k].map(FileChannel.MapMode.READ_WRITE, 0, totalRecords * bytesPerRecord,
                    Arena.global());
        }

        try {
            for (int i = 0; i < totalRecords; i++) {
                VectorRecord rec = records.get(i);
                if (qMode == 1) { // 2-bit mode
                    float[] z = transformer.preconditionAndRotate(rec.vector());
                    float threshold = TransformOperator.calculatePercentileThreshold(z, 0.20f);
                    long[][] packed = transformer.quantize2Bit(z, threshold);
                    long[] signPacked = packed[0];
                    long[] maskPacked = packed[1];

                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k];
                        long baseOffset = i * (count * 16L);
                        for (int l = 0; l < count; l++) {
                            tierMappeds[k].set(ValueLayout.JAVA_LONG, baseOffset + (l * 8L), signPacked[longOffset + l]);
                            tierMappeds[k].set(ValueLayout.JAVA_LONG, baseOffset + (count * 8L) + (l * 8L),
                                    maskPacked[longOffset + l]);
                        }
                        longOffset += count;
                    }
                } else if (qMode == 2) { // Float-Hybrid: write raw float32 values
                    float[] z = transformer.preconditionAndRotate(rec.vector());
                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k];
                        int startDim = (k == 0) ? 0 : tiers[k - 1];
                        int width = tiers[k] - startDim;
                        long baseOffset = (long) i * width * 4;
                        for (int l = 0; l < width; l++) {
                            int raw = Float.floatToRawIntBits(z[startDim + l]);
                            tierMappeds[k].set(ValueLayout.JAVA_INT_UNALIGNED, baseOffset + (l * 4L), raw);
                        }
                        longOffset += count;
                    }
                } else { // 1-bit mode
                    long[] packed = transformer.transformAndQuantize(rec.vector());
                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k];
                        long baseOffset = i * (count * 8L);
                        for (int l = 0; l < count; l++) {
                            tierMappeds[k].set(ValueLayout.JAVA_LONG, baseOffset + (l * 8L), packed[longOffset + l]);
                        }
                        longOffset += count;
                    }
                }
            }
            for (int k = 0; k < numTiers; k++) {
                tierMappeds[k].force();
            }
        } finally {
            for (int k = 0; k < numTiers; k++) {
                if (tierChannels[k] != null) {
                    tierChannels[k].close();
                }
            }
        }

        // 5. Write FP16 sidecar: stores original (pre-rotation) vectors in IEEE 754 half-precision
        if (writeFp16) {
            Path fp16Path = Path.of(basePath + "_fp16.bin");
            try (FileChannel channel = FileChannel.open(fp16Path,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.READ,
                    StandardOpenOption.TRUNCATE_EXISTING)) {
                long fp16Bytes = totalRecords * dimension * 2L;
                MemorySegment fp16Mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, fp16Bytes, Arena.global());
                for (int i = 0; i < totalRecords; i++) {
                    float[] vec = records.get(i).vector();
                    long rowOffset = (long) i * dimension * 2L;
                    for (int d = 0; d < dimension; d++) {
                        short fp16 = Float.floatToFloat16(vec[d]);
                        fp16Mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, rowOffset + d * 2L, fp16);
                    }
                }
                fp16Mapped.force();
            }
        }
    }

    // =========================================================================
    // CUDA Acceleration Support
    // =========================================================================

    private boolean cudaEnabled = false;
    private int cudaDeviceId = 0;

    /// Initializes CUDA support on the specified device.
    public int cudaInit(int deviceId) {
        this.cudaDeviceId = deviceId;
        this.cudaEnabled = true;
        return CudaDeviceManager.initialize(deviceId);
    }

    /// Shuts down CUDA resources.
    public void cudaShutdown() {
        CudaDeviceManager.shutdown();
        this.cudaEnabled = false;
    }

    /// Checks if CUDA is active and initialized.
    public boolean cudaIsAvailable() {
        return cudaEnabled && CudaDeviceManager.isAvailable() != 0;
    }

    /// Dispatches a CUDA batch search on the named index.
    public List<Index.SearchResult>[] cudaBatchSearch(String indexName, float[][] queries, int k) {
        Index index = getIndex(indexName);
        if (index == null) {
            throw new IllegalArgumentException("Index not found: " + indexName);
        }
        return index.cudaBatchSearch(queries, k);
    }

    /// Dispatches a CUDA multi-family resonant voting query on the named index.
    public long cudaQueryPlanetaryGrid(String indexName, float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        Index index = getIndex(indexName);
        if (index == null) {
            throw new IllegalArgumentException("Index not found: " + indexName);
        }
        return index.cudaQueryPlanetaryGrid(queries, families, thresholds, votingMask);
    }

    /// Compacts multiple compiled Pithos indices into a single consolidated index file layout.
    ///
    /// Validates schema compatibility ($D, \text{tiers}, \text{qMode}, \text{planetId}$) and performs zero-copy
    /// sidecar merging via `FileChannel.transferTo`.
    ///
    /// @param sourcePathsJoined semicolon-separated list of source index basepaths
    /// @param targetPath destination basepath for the consolidated index
    /// @throws IOException on I/O error
    public static void compactIndexes(String sourcePathsJoined, String targetPath) throws IOException {
        String[] sourcePaths = sourcePathsJoined.split(";");
        if (sourcePaths.length == 0) {
            throw new IllegalArgumentException("No source paths specified for compaction");
        }

        Path firstHeaderPath = Path.of(sourcePaths[0]);
        if (!Files.exists(firstHeaderPath)) {
            throw new FileNotFoundException("Source index not found: " + sourcePaths[0]);
        }

        byte firstPlanetId;
        long firstPlanetRadius;
        int firstDimension;
        int firstTiersCount;
        int[] firstTiers;
        byte firstQMode;

        try (FileChannel channel = FileChannel.open(firstHeaderPath, StandardOpenOption.READ)) {
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, 64, Arena.global());
            if (mapped.get(ValueLayout.JAVA_BYTE, 0) != 'P' ||
                mapped.get(ValueLayout.JAVA_BYTE, 1) != 'L' ||
                mapped.get(ValueLayout.JAVA_BYTE, 2) != 'A' ||
                mapped.get(ValueLayout.JAVA_BYTE, 3) != 'N') {
                throw new IOException("Invalid Pithos index magic bytes in: " + sourcePaths[0]);
            }
            firstPlanetId = mapped.get(ValueLayout.JAVA_BYTE, 4);
            firstPlanetRadius = mapped.get(ValueLayout.JAVA_LONG_UNALIGNED, 13);
            firstDimension = mapped.get(ValueLayout.JAVA_INT_UNALIGNED, 21);
            firstTiersCount = mapped.get(ValueLayout.JAVA_INT_UNALIGNED, 25);
            firstTiers = new int[firstTiersCount];
            for (int i = 0; i < firstTiersCount; i++) {
                firstTiers[i] = mapped.get(ValueLayout.JAVA_INT_UNALIGNED, 29 + (i * 4));
            }
            firstQMode = mapped.get(ValueLayout.JAVA_BYTE, 61);
        }

        long combinedRecords = 0;
        for (String sourcePathStr : sourcePaths) {
            Path headerPath = Path.of(sourcePathStr);
            if (!Files.exists(headerPath)) {
                throw new FileNotFoundException("Source index not found: " + sourcePathStr);
            }
            try (FileChannel channel = FileChannel.open(headerPath, StandardOpenOption.READ)) {
                MemorySegment mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, 64, Arena.global());
                if (mapped.get(ValueLayout.JAVA_BYTE, 0) != 'P' ||
                    mapped.get(ValueLayout.JAVA_BYTE, 1) != 'L' ||
                    mapped.get(ValueLayout.JAVA_BYTE, 2) != 'A' ||
                    mapped.get(ValueLayout.JAVA_BYTE, 3) != 'N') {
                    throw new IOException("Invalid Pithos index magic bytes in: " + sourcePathStr);
                }

                byte pid = mapped.get(ValueLayout.JAVA_BYTE, 4);
                long size = mapped.get(ValueLayout.JAVA_LONG_UNALIGNED, 5);
                long radius = mapped.get(ValueLayout.JAVA_LONG_UNALIGNED, 13);
                int dim = mapped.get(ValueLayout.JAVA_INT_UNALIGNED, 21);
                int tiersCnt = mapped.get(ValueLayout.JAVA_INT_UNALIGNED, 25);
                byte qm = mapped.get(ValueLayout.JAVA_BYTE, 61);

                if (pid != firstPlanetId || radius != firstPlanetRadius || dim != firstDimension || tiersCnt != firstTiersCount || qm != firstQMode) {
                    throw new IllegalArgumentException("Index schema mismatch for source: " + sourcePathStr);
                }

                for (int i = 0; i < tiersCnt; i++) {
                    int t = mapped.get(ValueLayout.JAVA_INT_UNALIGNED, 29 + (i * 4));
                    if (t != firstTiers[i]) {
                        throw new IllegalArgumentException("Index tiers mismatch for source: " + sourcePathStr);
                    }
                }
                combinedRecords += size;
            }
        }

        Path targetHeaderPath = Path.of(targetPath);
        Path parentDir = targetHeaderPath.getParent();
        if (parentDir != null) {
            Files.createDirectories(parentDir);
        }

        try (FileChannel channel = FileChannel.open(targetHeaderPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, 64, Arena.global());
            mapped.set(ValueLayout.JAVA_BYTE, 0, (byte) 'P');
            mapped.set(ValueLayout.JAVA_BYTE, 1, (byte) 'L');
            mapped.set(ValueLayout.JAVA_BYTE, 2, (byte) 'A');
            mapped.set(ValueLayout.JAVA_BYTE, 3, (byte) 'N');
            mapped.set(ValueLayout.JAVA_BYTE, 4, firstPlanetId);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 5, combinedRecords);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 13, firstPlanetRadius);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 21, firstDimension);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 25, firstTiersCount);
            for (int i = 0; i < firstTiersCount; i++) {
                mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 29 + (i * 4), firstTiers[i]);
            }
            mapped.set(ValueLayout.JAVA_BYTE, 61, firstQMode);
            mapped.force();
        }

        mergeSidecarFiles(sourcePaths, targetPath, "_ids.bin");
        mergeSidecarFiles(sourcePaths, targetPath, "_metadata.bin");
        for (int k = 0; k < firstTiersCount; k++) {
            mergeSidecarFiles(sourcePaths, targetPath, "_tier_" + k + ".bin");
        }

        Path firstFp16Path = Path.of(sourcePaths[0] + "_fp16.bin");
        if (Files.exists(firstFp16Path)) {
            mergeSidecarFiles(sourcePaths, targetPath, "_fp16.bin");
        }
    }

    private static void mergeSidecarFiles(String[] sourcePaths, String targetBasePath, String suffix) throws IOException {
        Path targetPath = Path.of(targetBasePath + suffix);
        try (FileChannel targetChannel = FileChannel.open(targetPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            for (String sourceBasePath : sourcePaths) {
                Path sourcePath = Path.of(sourceBasePath + suffix);
                if (Files.exists(sourcePath)) {
                    try (FileChannel sourceChannel = FileChannel.open(sourcePath, StandardOpenOption.READ)) {
                        long size = sourceChannel.size();
                        long position = 0;
                        while (position < size) {
                            position += sourceChannel.transferTo(position, size - position, targetChannel);
                        }
                    }
                }
            }
        }
    }
}
