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
///    - Offset 4: `domainId` (1 byte domain tag)
///    - Offset 5..12: Total record count N (8-byte unaligned long)
///    - Offset 13..20: Reference radius R in meters (8-byte unaligned long)
///    - Offset 21..24: Vector dimension D (4-byte unaligned int)
///    - Offset 25..28: Cumulative tier count T (4-byte unaligned int, 1 ≤ T ≤ 8)
///    - Offset 29..60: Cumulative dimension boundaries for each tier (up to 8 ints)
///    - Offset 61: Quantization mode `qMode` (0 = 1-bit, 1 = 2-bit ternary, 2 = float32 bypass)
/// 2. **`<basePath>_ids.bin` (N × 8 bytes):** 64-bit record IDs.
/// 3. **`<basePath>_metadata.bin` (N × 8 bytes):** Bitmask flags and tombstones (bit 0 = tombstone).
/// 4. **`<basePath>_tier_k.bin` (N × bytesPerRecord_k bytes):** Binarized columnar vectors.
/// 5. **`<basePath>_fp16.bin` (N × D × 2 bytes, optional):** Half-precision IEEE 754 raw floats for in-engine Stage 2 reranking.
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
    /// @param weights optional projection/LoRA weights of size D × D₀
    /// @param loraDim bottleneck dimension D₀
    /// @return registered `Index` instance
    /// @throws IOException on I/O error
    public Index loadIndex(String name, String basePath, float[] weights, int loraDim) throws IOException {
        if (name == null || name.isBlank() || name.contains("\0")) {
            throw new IllegalArgumentException("Index name cannot be empty or contain null bytes");
        }
        if (basePath == null || basePath.isBlank() || basePath.contains("\0")) {
            throw new IllegalArgumentException("Base path cannot be empty or contain null bytes");
        }
        String normalizedPath = java.nio.file.Path.of(basePath).normalize().toString();
        Index index = FlatIndex.mapFile(normalizedPath, weights, loraDim);
        indices.put(name, index);
        indexPaths.put(name, normalizedPath);
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
    /// merging and deduplicating results to return the combined top-k.
    ///
    /// @param indexName name of the index
    /// @param query raw float query vector
    /// @param k nearest neighbor count
    /// @return merged top-k search results
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

    public static final int SIDECAR_NONE = 0;
    public static final int SIDECAR_FP16 = 1;
    public static final int SIDECAR_FP8  = 2;
    public static final int SIDECAR_FP4  = 3;

    public static final float[] FP4_E2M1_TABLE = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    /// Converts a 32-bit float to an 8-bit OCP/NVIDIA FP8 E4M3 standard byte.
    public static byte encodeFP8_E4M3(float val) {
        int bits = Float.floatToRawIntBits(val);
        int sign = (bits >>> 31) & 0x1;
        int exp = (bits >>> 23) & 0xFF;
        int mant = bits & 0x7FFFFF;

        if (Float.isNaN(val)) {
            return (byte) ((sign << 7) | 0x7F);
        }
        if (Float.isInfinite(val)) {
            return (byte) ((sign << 7) | 0x7E); // Clamp to max finite (448.0)
        }

        float absVal = Math.abs(val);
        if (absVal >= 448.0f) {
            return (byte) ((sign << 7) | 0x7E); // 448.0 is max finite in E4M3
        }
        if (absVal < (0.5f / 512.0f)) { // Underflow to zero
            return (byte) (sign << 7);
        }

        // Subnormal in E4M3: absVal < 2^(-6) = 0.015625
        if (absVal < 0.015625f) {
            int m = Math.round(absVal * 512.0f);
            if (m > 7) m = 7;
            return (byte) ((sign << 7) | (m & 0x7));
        }

        // Normal E4M3: exponent bias is 7
        int e = exp - 127 + 7;
        if (e < 1) {
            int m = Math.round(absVal * 512.0f);
            if (m > 7) m = 7;
            return (byte) ((sign << 7) | (m & 0x7));
        }
        if (e > 15) {
            return (byte) ((sign << 7) | 0x7E);
        }

        int m = (mant + (1 << 19)) >>> 20;
        if (m >= 8) {
            e += 1;
            m = 0;
            if (e >= 16) {
                return (byte) ((sign << 7) | 0x7E);
            }
        }
        if (e == 15 && m == 7) {
            m = 6;
        }
        return (byte) ((sign << 7) | (e << 3) | (m & 0x7));
    }

    /// Converts an 8-bit OCP/NVIDIA FP8 E4M3 byte to a 32-bit float.
    public static float decodeFP8_E4M3(byte b) {
        int u = b & 0xFF;
        int sign = (u >>> 7) & 1;
        int exp = (u >>> 3) & 0xF;
        int mant = u & 0x7;

        if (exp == 0xF && mant == 0x7) {
            return sign == 1 ? -Float.NaN : Float.NaN;
        }
        float signMult = (sign == 1) ? -1.0f : 1.0f;
        if (exp == 0) {
            return signMult * (mant / 512.0f);
        } else {
            float scale = (float) Math.scalb(1.0f, exp - 7);
            return signMult * scale * (1.0f + mant / 8.0f);
        }
    }

    /// Quantizes a single float into a 4-bit NVFP4 E2M1 nibble.
    public static byte encodeFP4_E2M1_Nibble(float val) {
        int sign = (Float.floatToRawIntBits(val) >>> 31) & 1;
        float absVal = Math.abs(val);
        int bestIdx = 0;
        float bestDiff = absVal;
        for (int i = 1; i < 8; i++) {
            float diff = Math.abs(absVal - FP4_E2M1_TABLE[i]);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIdx = i;
            }
        }
        return (byte) ((sign << 3) | (bestIdx & 0x7));
    }

    /// Decodes a 4-bit NVFP4 E2M1 nibble to a 32-bit float.
    public static float decodeFP4_E2M1_Nibble(int nibble) {
        return FP4_E2M1_TABLE[nibble & 0xF];
    }

    /// Encodes planetary Latitude and Longitude into a 48-bit geodetic Morton code.
    public static long encodeGeodeticMorton(double latDeg, double lonDeg) {
        double normLat = Math.max(0.0, Math.min(1.0, (latDeg + 90.0) / 180.0));
        double normLon = Math.max(0.0, Math.min(1.0, (lonDeg + 180.0) / 360.0));

        long x = (long) (normLon * ((1L << 24) - 1));
        long y = (long) (normLat * ((1L << 24) - 1));

        long result = 0;
        for (int i = 0; i < 24; i++) {
            result |= ((x & (1L << i)) << i) | ((y & (1L << i)) << (i + 1));
        }
        return result;
    }

    /// Compiles raw float records into a multi-tier, cache-aligned database file layout (1-bit default).
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records) throws IOException {
        compileIndexFile(basePath, planetId, planetRadius, dimension, tiers, records, 0, SIDECAR_NONE);
    }

    /// Compiles raw float records into a multi-tier, cache-aligned database file layout with configurable quantization mode.
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records, int qMode) throws IOException {
        compileIndexFile(basePath, planetId, planetRadius, dimension, tiers, records, qMode, SIDECAR_FP16);
    }

    /// Compiles raw continuous float vector records with boolean writeFp16 flag (backward compatible).
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records, int qMode, boolean writeFp16) throws IOException {
        compileIndexFile(basePath, planetId, planetRadius, dimension, tiers, records, qMode, writeFp16 ? SIDECAR_FP16 : SIDECAR_NONE);
    }

    /// Compiles raw continuous float vector records into a universal schema-agnostic single-file .pithos container.
    public static void compileContainer(String filePath, int dimension, int[] tiers, List<VectorRecord> records,
            int metricType, int qMode, int sidecarMode, byte[] metadataPayload, String metadataFormat, String userMetadataJson)
            throws IOException {
        Path path = Path.of(filePath.endsWith(".pithos") ? filePath : filePath + ".pithos");
        PithosContainer.writeContainer(path, dimension, tiers, records, metricType, qMode, sidecarMode,
                metadataPayload, metadataFormat, userMetadataJson);
    }

    /// Overload for compiling single-file container with default metadata.
    public static void compileContainer(String filePath, int dimension, int[] tiers, List<VectorRecord> records,
            int qMode, int sidecarMode) throws IOException {
        compileContainer(filePath, dimension, tiers, records, PithosContainer.METRIC_COSINE, qMode, sidecarMode, null, null, null);
    }

    /// Returns the user metadata JSON string for a registered index, or null if not available.
    public String getUserMetadata(String indexName) {
        Index index = getIndex(indexName);
        if (index instanceof FlatIndex flat) {
            return flat.getUserMetadataJson();
        }
        return null;
    }

    /// Returns the raw off-heap metadata payload segment for a registered index, or null if not available.
    public MemorySegment getMetadataPayload(String indexName) {
        Index index = getIndex(indexName);
        if (index instanceof FlatIndex flat) {
            return flat.getMetadataPayloadSegment();
        }
        return null;
    }

    /// Compiles raw continuous float vector records into a multi-tier binary columnar format on disk.
    ///
    /// @param basePath base filepath prefix
    /// @param planetId target planetary body identifier code
    /// @param planetRadius equatorial radius in meters
    /// @param dimension vector dimensionality (D)
    /// @param tiers cumulative Matryoshka tier step boundaries (up to 8 tiers)
    /// @param records input vector records
    /// @param qMode quantization mode: 0 = 1-bit, 1 = 2-bit ternary, 2 = float32 bypass
    /// @param sidecarMode sidecar format: 0 = None, 1 = FP16, 2 = FP8 E4M3, 3 = FP4 NVFP4
    /// @throws IOException on I/O failure
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records, int qMode, int sidecarMode) throws IOException {
        compileIndexFile(basePath, planetId, planetRadius, dimension, tiers, records, qMode, sidecarMode, null, null);
    }

    /// Compiles raw continuous float vector records with optional geospatial coordinates.
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records, int qMode, int sidecarMode, double[] latitudes, double[] longitudes) throws IOException {

        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException("Records list cannot be null or empty");
        }
        if (tiers == null || tiers.length == 0 || tiers.length > 8) {
            throw new IllegalArgumentException("Tiers must have between 1 and 8 step boundaries");
        }

        long totalRecords = records.size();
        boolean hasSpatial = (latitudes != null && longitudes != null && latitudes.length == totalRecords && longitudes.length == totalRecords);

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
            // Write sidecarMode to offset 62
            mapped.set(ValueLayout.JAVA_BYTE, 62, (byte) sidecarMode);
            // Write flags to offset 63
            mapped.set(ValueLayout.JAVA_BYTE, 63, (byte) (hasSpatial ? 1 : 0));
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

        // 3. Write Metadata file (tombstones & Morton spatial flags)
        Path metadataPath = Path.of(basePath + "_metadata.bin");
        try (FileChannel channel = FileChannel.open(metadataPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalRecords * 8, Arena.global());
            for (int i = 0; i < totalRecords; i++) {
                long metaWord = 2L; // bit 1 active, bit 0 tombstone = 0
                if (hasSpatial) {
                    long morton = encodeGeodeticMorton(latitudes[i], longitudes[i]);
                    metaWord = (morton << 16) | 2L;
                }
                mapped.set(ValueLayout.JAVA_LONG, i * 8L, metaWord);
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
            java.util.stream.IntStream.range(0, (int) totalRecords).parallel().forEach(i -> {
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
                        long baseOffset = (long) i * (count * 16L);
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
                        long baseOffset = (long) i * width * 4L;
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
                        long baseOffset = (long) i * (count * 8L);
                        for (int l = 0; l < count; l++) {
                            tierMappeds[k].set(ValueLayout.JAVA_LONG, baseOffset + (l * 8L), packed[longOffset + l]);
                        }
                        longOffset += count;
                    }
                }
            });
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

        // 5. Write Sidecar Files (FP16, FP8 E4M3, or FP4 NVFP4) in parallel
        if (sidecarMode == SIDECAR_FP16) {
            Path fp16Path = Path.of(basePath + "_fp16.bin");
            try (FileChannel channel = FileChannel.open(fp16Path,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.READ,
                    StandardOpenOption.TRUNCATE_EXISTING)) {
                long fp16Bytes = totalRecords * dimension * 2L;
                MemorySegment fp16Mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, fp16Bytes, Arena.global());
                java.util.stream.IntStream.range(0, (int) totalRecords).parallel().forEach(i -> {
                    float[] vec = records.get(i).vector();
                    long rowOffset = (long) i * dimension * 2L;
                    for (int d = 0; d < dimension; d++) {
                        short fp16 = Float.floatToFloat16(vec[d]);
                        fp16Mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, rowOffset + d * 2L, fp16);
                    }
                });
                fp16Mapped.force();
            }
        } else if (sidecarMode == SIDECAR_FP8) {
            Path fp8Path = Path.of(basePath + "_fp8.bin");
            try (FileChannel channel = FileChannel.open(fp8Path,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.READ,
                    StandardOpenOption.TRUNCATE_EXISTING)) {
                long fp8Bytes = totalRecords * dimension * 1L;
                MemorySegment fp8Mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, fp8Bytes, Arena.global());
                java.util.stream.IntStream.range(0, (int) totalRecords).parallel().forEach(i -> {
                    float[] vec = records.get(i).vector();
                    long rowOffset = (long) i * dimension;
                    for (int d = 0; d < dimension; d++) {
                        byte fp8 = encodeFP8_E4M3(vec[d]);
                        fp8Mapped.set(ValueLayout.JAVA_BYTE, rowOffset + d, fp8);
                    }
                });
                fp8Mapped.force();
            }
        } else if (sidecarMode == SIDECAR_FP4) {
            Path fp4Path = Path.of(basePath + "_fp4.bin");
            int blockSize = 16;
            int numBlocks = (dimension + blockSize - 1) / blockSize;
            int bytesPerRecord = numBlocks * 9; // 8 bytes for 16 nibbles + 1 byte FP8 scale factor
            try (FileChannel channel = FileChannel.open(fp4Path,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.READ,
                    StandardOpenOption.TRUNCATE_EXISTING)) {
                long fp4Bytes = totalRecords * (long) bytesPerRecord;
                MemorySegment fp4Mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, fp4Bytes, Arena.global());
                java.util.stream.IntStream.range(0, (int) totalRecords).parallel().forEach(i -> {
                    float[] vec = records.get(i).vector();
                    long rowOffset = (long) i * bytesPerRecord;
                    for (int b = 0; b < numBlocks; b++) {
                        int blockStart = b * blockSize;
                        float maxAbs = 0.0f;
                        for (int j = 0; j < blockSize; j++) {
                            int dimIdx = blockStart + j;
                            if (dimIdx < dimension) {
                                float abs = Math.abs(vec[dimIdx]);
                                if (abs > maxAbs) maxAbs = abs;
                            }
                        }
                        float scale = maxAbs / 6.0f;
                        byte scaleByte = encodeFP8_E4M3(scale);
                        long blockOffset = rowOffset + b * 9L;
                        fp4Mapped.set(ValueLayout.JAVA_BYTE, blockOffset, scaleByte);

                        float invScale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
                        for (int j = 0; j < 8; j++) {
                            int d0 = blockStart + j * 2;
                            int d1 = blockStart + j * 2 + 1;
                            float v0 = (d0 < dimension) ? vec[d0] * invScale : 0.0f;
                            float v1 = (d1 < dimension) ? vec[d1] * invScale : 0.0f;
                            byte n0 = encodeFP4_E2M1_Nibble(v0);
                            byte n1 = encodeFP4_E2M1_Nibble(v1);
                            byte packed = (byte) ((n1 << 4) | (n0 & 0xF));
                            fp4Mapped.set(ValueLayout.JAVA_BYTE, blockOffset + 1 + j, packed);
                        }
                    }
                });
                fp4Mapped.force();
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
    /// Validates schema compatibility (D, tiers, qMode, planetId) and performs zero-copy
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
        byte firstSidecarMode;

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
            firstSidecarMode = mapped.get(ValueLayout.JAVA_BYTE, 62);
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
                byte sc = mapped.get(ValueLayout.JAVA_BYTE, 62);

                if (pid != firstPlanetId || radius != firstPlanetRadius || dim != firstDimension || tiersCnt != firstTiersCount || qm != firstQMode || sc != firstSidecarMode) {
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
            mapped.set(ValueLayout.JAVA_BYTE, 62, firstSidecarMode);
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
        Path firstFp8Path = Path.of(sourcePaths[0] + "_fp8.bin");
        if (Files.exists(firstFp8Path)) {
            mergeSidecarFiles(sourcePaths, targetPath, "_fp8.bin");
        }
        Path firstFp4Path = Path.of(sourcePaths[0] + "_fp4.bin");
        if (Files.exists(firstFp4Path)) {
            mergeSidecarFiles(sourcePaths, targetPath, "_fp4.bin");
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
