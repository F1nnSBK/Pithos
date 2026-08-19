package org.pithos;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;

/// # PithosContainer
///
/// Implements the universal, schema-agnostic single-file `.pithos` container format.
///
/// ### Philosophical Heritage:
/// Named after the *pithos* (πίθος) — the great clay jar in which Diogenes of Sinope dwelt in Athens.
/// Embodying absolute autarky, zero extraneous baggage, and zero runtime pointer indirection.
///
/// ### Binary Container Specification (Version 2):
/// ```text
/// +-------------------------------------------------------------------------+
/// | SUPERBLOCK (128 Bytes): Magic ("DIOGENES"), Version 2, Metric, Dim, N   |
/// +-------------------------------------------------------------------------+
/// | SECTION 1: Item IDs (N x 8 Bytes uint64, 64-Byte Cache Line Aligned)    |
/// +-------------------------------------------------------------------------+
/// | SECTION 2: Quantization Tiers (Bit-packed Ternary/Binary Tier Columns)  |
/// +-------------------------------------------------------------------------+
/// | SECTION 3: Precision Sidecar (FP8 E4M3 / NVFP4 / FP16 / FP32 Matrix)   |
/// +-------------------------------------------------------------------------+
/// | SECTION 4: Generic Metadata Payload (JSONL / Arrow IPC / Raw Blobs)     |
/// +-------------------------------------------------------------------------+
/// | FOOTER DIRECTORY (Variable Length JSON TOC with Section Offsets)        |
/// +-------------------------------------------------------------------------+
/// | FOOTER TRAILER (20 Bytes): toc_offset (uint64), toc_len (u32), "PITHOSDB"|
/// +-------------------------------------------------------------------------+
/// ```
public final class PithosContainer {

    public static final byte[] MAGIC_SUPERBLOCK = new byte[] { 'D', 'I', 'O', 'G', 'E', 'N', 'E', 'S' };
    public static final byte[] MAGIC_TRAILER = new byte[] { 'P', 'I', 'T', 'H', 'O', 'S', 'D', 'B' };
    public static final int FORMAT_VERSION = 2;
    public static final int SUPERBLOCK_SIZE = 128;
    public static final int TRAILER_SIZE = 20; // 8 bytes toc_offset + 4 bytes toc_length + 8 bytes magic
    public static final int ALIGNMENT = 64; // 64-byte cache line alignment

    public static final int METRIC_COSINE = 0;
    public static final int METRIC_L2 = 1;
    public static final int METRIC_DOT_PRODUCT = 2;

    public static final int NUM_PREFIX_BUCKETS = 65536;
    public static final int PREFIX_OFFSETS_COUNT = NUM_PREFIX_BUCKETS + 1;
    public static final long PREFIX_OFFSETS_BYTES = PREFIX_OFFSETS_COUNT * 4L;

    private PithosContainer() {
    }

    /// Aligns a byte offset upwards to the next 64-byte boundary.
    public static long align64(long offset) {
        return (offset + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1L);
    }

    /// Superblock metadata descriptor.
    public record Superblock(
            int version,
            long numVectors,
            int dimension,
            int metricType,
            int sidecarType,
            int numTiers,
            int[] tiers,
            long tocOffset,
            int tocLength,
            int qMode) {
    }

    /// Container Section descriptor.
    public record Section(long offset, long length, String format) {
    }

    /// Extracts metadata section offset, length, and format from TOC JSON.
    public static Section extractMetadataSection(String tocJson) {
        if (tocJson == null) return new Section(0, 0, "none");
        int metaIdx = tocJson.indexOf("\"metadata\"");
        if (metaIdx < 0) return new Section(0, 0, "none");
        int objStart = tocJson.indexOf('{', metaIdx);
        int objEnd = tocJson.indexOf('}', objStart);
        if (objStart < 0 || objEnd < 0) return new Section(0, 0, "none");
        String metaSub = tocJson.substring(objStart, objEnd + 1);

        long offset = extractLongField(metaSub, "offset");
        long length = extractLongField(metaSub, "length");
        String format = extractStringField(metaSub, "format");
        return new Section(offset, length, format);
    }

    /// Extracts prefix_table section offset, length, and format from TOC JSON.
    public static Section extractPrefixTableSection(String tocJson) {
        if (tocJson == null) return new Section(0, 0, "none");
        int idx = tocJson.indexOf("\"prefix_table\"");
        if (idx < 0) return new Section(0, 0, "none");
        int objStart = tocJson.indexOf('{', idx);
        int objEnd = tocJson.indexOf('}', objStart);
        if (objStart < 0 || objEnd < 0) return new Section(0, 0, "none");
        String sub = tocJson.substring(objStart, objEnd + 1);

        long offset = extractLongField(sub, "offset");
        long length = extractLongField(sub, "length");
        String format = extractStringField(sub, "format");
        return new Section(offset, length, format);
    }

    private static long extractLongField(String json, String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return 0;
        int colon = json.indexOf(':', idx);
        if (colon < 0) return 0;
        int start = colon + 1;
        while (start < json.length() && (Character.isWhitespace(json.charAt(start)) || json.charAt(start) == '"')) start++;
        int end = start;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) end++;
        try {
            return Long.parseLong(json.substring(start, end));
        } catch (Exception e) {
            return 0;
        }
    }

    private static String extractStringField(String json, String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return "none";
        int colon = json.indexOf(':', idx);
        if (colon < 0) return "none";
        int start = json.indexOf('"', colon);
        if (start < 0) return "none";
        int end = json.indexOf('"', start + 1);
        if (end < 0) return "none";
        return json.substring(start + 1, end);
    }

    /// Parsed Table of Contents (TOC).
    public record ContainerToc(
            String format,
            long idsOffset,
            long idsLength,
            long[] tierOffsets,
            long[] tierLengths,
            long sidecarOffset,
            long sidecarLength,
            String sidecarFormat,
            long prefixTableOffset,
            long prefixTableLength,
            long metadataOffset,
            long metadataLength,
            String metadataFormat,
            String userMetadataJson) {
    }

    /// Checks if a file starts with the 8-byte `DIOGENES` magic sequence.
    public static boolean isPithosContainer(Path path) {
        if (!Files.exists(path) || !Files.isRegularFile(path)) {
            return false;
        }
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            if (channel.size() < SUPERBLOCK_SIZE + TRAILER_SIZE) {
                return false;
            }
            ByteBuffer buf = ByteBuffer.allocate(8);
            channel.read(buf, 0);
            buf.flip();
            byte[] magic = new byte[8];
            buf.get(magic);
            for (int i = 0; i < 8; i++) {
                if (magic[i] != MAGIC_SUPERBLOCK[i]) {
                    return false;
                }
            }
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /// Reads and parses the 128-byte Superblock from an open FileChannel.
    public static Superblock readSuperblock(FileChannel channel) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(SUPERBLOCK_SIZE);
        buf.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        channel.read(buf, 0);
        buf.flip();

        byte[] magic = new byte[8];
        buf.get(magic);
        for (int i = 0; i < 8; i++) {
            if (magic[i] != MAGIC_SUPERBLOCK[i]) {
                throw new IOException("Invalid Pithos container: Superblock magic must be 'DIOGENES'");
            }
        }

        int version = buf.getInt();
        if (version != FORMAT_VERSION) {
            throw new IOException("Unsupported Pithos container version: " + version + " (expected " + FORMAT_VERSION + ")");
        }

        long numVectors = buf.getLong();
        int dimension = buf.getInt();
        int metricType = buf.getShort() & 0xFFFF;
        int sidecarType = buf.getShort() & 0xFFFF;
        int numTiers = buf.getShort() & 0xFFFF;

        if (numTiers < 1 || numTiers > 8) {
            throw new IOException("Invalid tier count in container: " + numTiers);
        }

        int[] tiers = new int[numTiers];
        for (int i = 0; i < 8; i++) {
            int t = buf.getShort() & 0xFFFF;
            if (i < numTiers) {
                tiers[i] = t;
            }
        }

        long tocOffset = buf.getLong();
        int tocLength = buf.getInt();
        int qMode = buf.getShort() & 0xFFFF;

        return new Superblock(version, numVectors, dimension, metricType, sidecarType, numTiers, tiers, tocOffset, tocLength, qMode);
    }

    /// Reads the Table of Contents (TOC) JSON string from the file.
    public static String readTocJson(FileChannel channel, long tocOffset, int tocLength) throws IOException {
        if (tocOffset <= 0 || tocLength <= 0 || tocOffset + tocLength > channel.size()) {
            throw new IOException("Invalid TOC offset/length: offset=" + tocOffset + ", len=" + tocLength + ", fileSize=" + channel.size());
        }
        ByteBuffer buf = ByteBuffer.allocate(tocLength);
        channel.read(buf, tocOffset);
        buf.flip();
        byte[] bytes = new byte[tocLength];
        buf.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /// Validates the 20-byte Trailer at the end of the file.
    public static void validateTrailer(FileChannel channel, long expectedTocOffset, int expectedTocLength) throws IOException {
        long fileSize = channel.size();
        if (fileSize < SUPERBLOCK_SIZE + TRAILER_SIZE) {
            throw new IOException("Pithos container file is truncated (size=" + fileSize + ")");
        }
        ByteBuffer buf = ByteBuffer.allocate(TRAILER_SIZE);
        buf.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        channel.read(buf, fileSize - TRAILER_SIZE);
        buf.flip();

        long tocOffset = buf.getLong();
        int tocLength = buf.getInt();
        byte[] magic = new byte[8];
        buf.get(magic);

        for (int i = 0; i < 8; i++) {
            if (magic[i] != MAGIC_TRAILER[i]) {
                throw new IOException("Corrupted Pithos container: Trailer signature must be 'PITHOSDB'");
            }
        }

        if (expectedTocOffset > 0 && tocOffset != expectedTocOffset) {
            throw new IOException("TOC offset mismatch: header=" + expectedTocOffset + ", trailer=" + tocOffset);
        }
        if (expectedTocLength > 0 && tocLength != expectedTocLength) {
            throw new IOException("TOC length mismatch: header=" + expectedTocLength + ", trailer=" + tocLength);
        }
    }

    /// Compiles vector records into a complete, self-contained single-file `.pithos` container.
    public static void writeContainer(
            Path targetFile,
            int dimension,
            int[] tiers,
            List<VectorRecord> records,
            int metricType,
            int qMode,
            int sidecarMode,
            byte[] metadataPayload,
            String metadataFormat,
            String userMetadataJson) throws IOException {

        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException("Records cannot be null or empty");
        }
        if (tiers == null || tiers.length == 0 || tiers.length > 8) {
            throw new IllegalArgumentException("Tiers must have between 1 and 8 step boundaries");
        }

        long numVectors = records.size();
        int numTiers = tiers.length;

        // Calculate section sizes
        long idsOffset = align64(SUPERBLOCK_SIZE);
        long idsLength = numVectors * 8L;

        long[] tierOffsets = new long[numTiers];
        long[] tierLengths = new long[numTiers];
        int[] tierLongs = new int[numTiers];

        long currentOffset = align64(idsOffset + idsLength);
        int prevBound = 0;
        for (int k = 0; k < numTiers; k++) {
            int width = tiers[k] - prevBound;
            tierLongs[k] = width / 64;
            prevBound = tiers[k];

            long bytesPerRecord = switch (qMode) {
                case 1 -> (width / 4); // 2-bit QJL: 2 bits/dim
                case 2 -> (width * 4L); // Float-Hybrid: 4 bytes/dim
                default -> (width / 8); // 1-bit: 1 bit/dim
            };
            tierOffsets[k] = currentOffset;
            tierLengths[k] = numVectors * bytesPerRecord;
            currentOffset = align64(currentOffset + tierLengths[k]);
        }

        // Sidecar section
        long sidecarOffset = 0;
        long sidecarLength = 0;
        String sidecarFormat = "none";

        if (sidecarMode == VectorDb.SIDECAR_FP16) {
            sidecarOffset = currentOffset;
            sidecarLength = numVectors * dimension * 2L;
            sidecarFormat = "fp16";
            currentOffset = align64(currentOffset + sidecarLength);
        } else if (sidecarMode == VectorDb.SIDECAR_FP8) {
            sidecarOffset = currentOffset;
            sidecarLength = numVectors * dimension * 1L;
            sidecarFormat = "fp8_e4m3";
            currentOffset = align64(currentOffset + sidecarLength);
        } else if (sidecarMode == VectorDb.SIDECAR_FP4) {
            int blockSize = 16;
            int numBlocks = (dimension + blockSize - 1) / blockSize;
            int bytesPerRecord = numBlocks * 9;
            sidecarOffset = currentOffset;
            sidecarLength = numVectors * (long) bytesPerRecord;
            sidecarFormat = "nvfp4_e2m1";
            currentOffset = align64(currentOffset + sidecarLength);
        }

        // Prefix Table Section (Direct-Mapped 16-Bit Inverted Index, 65536 buckets)
        long prefixTableOffset = currentOffset;
        long prefixPostingsLength = numVectors * 4L;
        long prefixTableLength = PREFIX_OFFSETS_BYTES + prefixPostingsLength;
        currentOffset = align64(currentOffset + prefixTableLength);

        // Metadata payload section
        long metadataOffset = 0;
        long metadataLength = 0;
        String metaFormat = (metadataFormat != null && !metadataFormat.isBlank()) ? metadataFormat : "raw";

        if (metadataPayload != null && metadataPayload.length > 0) {
            metadataOffset = currentOffset;
            metadataLength = metadataPayload.length;
            currentOffset = align64(currentOffset + metadataLength);
        }

        // Build Table of Contents (TOC) JSON
        StringBuilder tocBuilder = new StringBuilder();
        tocBuilder.append("{\n");
        tocBuilder.append("  \"format\": \"pithos_v2\",\n");
        tocBuilder.append("  \"motto\": \"Autarky: Self-contained & Zero Baggage\",\n");
        tocBuilder.append("  \"sections\": {\n");
        tocBuilder.append("    \"ids\": { \"offset\": ").append(idsOffset).append(", \"length\": ").append(idsLength).append(", \"dtype\": \"uint64\" },\n");
        for (int k = 0; k < numTiers; k++) {
            tocBuilder.append("    \"tier_").append(k).append("\": { \"offset\": ").append(tierOffsets[k])
                    .append(", \"length\": ").append(tierLengths[k]).append(", \"dim_boundary\": ").append(tiers[k]).append(" },\n");
        }
        tocBuilder.append("    \"sidecar\": { \"offset\": ").append(sidecarOffset).append(", \"length\": ").append(sidecarLength)
                .append(", \"format\": \"").append(sidecarFormat).append("\" },\n");
        tocBuilder.append("    \"prefix_table\": { \"offset\": ").append(prefixTableOffset).append(", \"length\": ").append(prefixTableLength)
                .append(", \"num_buckets\": ").append(NUM_PREFIX_BUCKETS).append(", \"format\": \"direct_mapped_csr\" },\n");
        tocBuilder.append("    \"metadata\": { \"offset\": ").append(metadataOffset).append(", \"length\": ").append(metadataLength)
                .append(", \"format\": \"").append(metaFormat).append("\" }\n");
        tocBuilder.append("  },\n");
        tocBuilder.append("  \"user_metadata\": ").append((userMetadataJson != null && !userMetadataJson.isBlank()) ? userMetadataJson : "{}").append("\n");
        tocBuilder.append("}");

        byte[] tocBytes = tocBuilder.toString().getBytes(StandardCharsets.UTF_8);
        long tocOffset = currentOffset;
        int tocLength = tocBytes.length;
        currentOffset = align64(currentOffset + tocLength);

        long totalFileSize = currentOffset + TRAILER_SIZE;

        // Open target file and map for writing
        try (FileChannel channel = FileChannel.open(targetFile,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalFileSize, Arena.global());

            // 1. Write Superblock (128 Bytes)
            for (int i = 0; i < 8; i++) {
                mapped.set(ValueLayout.JAVA_BYTE, i, MAGIC_SUPERBLOCK[i]);
            }
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 8, FORMAT_VERSION);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 12, numVectors);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 20, dimension);
            mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, 24, (short) metricType);
            mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, 26, (short) sidecarMode);
            mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, 28, (short) numTiers);
            for (int i = 0; i < 8; i++) {
                short tDim = (i < numTiers) ? (short) tiers[i] : 0;
                mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, 30 + (i * 2L), tDim);
            }
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 46, tocOffset);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 54, tocLength);
            mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, 58, (short) qMode);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 60, prefixTableOffset);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 68, prefixTableLength);

            // Zero reserved bytes 76..127
            for (long p = 76; p < SUPERBLOCK_SIZE; p++) {
                mapped.set(ValueLayout.JAVA_BYTE, p, (byte) 0);
            }

            // 2. Write IDs Section
            for (int i = 0; i < numVectors; i++) {
                mapped.set(ValueLayout.JAVA_LONG, idsOffset + (i * 8L), records.get(i).id());
            }

            // 3. Write Quantization Tiers Section & Compute Prefix Table Postings
            int[] vectorBucketKeys = new int[(int) numVectors];
            int[] bucketCounts = new int[NUM_PREFIX_BUCKETS];

            TransformOperator transformer = new TransformOperator(dimension, tiers);
            for (int i = 0; i < numVectors; i++) {
                VectorRecord rec = records.get(i);
                int bucketKey;
                if (qMode == 1) { // 2-bit QJL Residuals
                    float[] z = transformer.preconditionAndRotate(rec.vector());
                    float threshold = TransformOperator.calculatePercentileThreshold(z, 0.20f);
                    long[][] packed = transformer.quantize2Bit(z, threshold);
                    long[] signPacked = packed[0];
                    long[] maskPacked = packed[1];
                    bucketKey = (int) (signPacked[0] & 0xFFFFL);

                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k];
                        long baseOffset = tierOffsets[k] + (i * (count * 16L));
                        for (int l = 0; l < count; l++) {
                            mapped.set(ValueLayout.JAVA_LONG, baseOffset + (l * 8L), signPacked[longOffset + l]);
                            mapped.set(ValueLayout.JAVA_LONG, baseOffset + (count * 8L) + (l * 8L), maskPacked[longOffset + l]);
                        }
                        longOffset += count;
                    }
                } else if (qMode == 2) { // Float-Hybrid raw float32
                    float[] z = transformer.preconditionAndRotate(rec.vector());
                    int k16 = 0;
                    for (int j = 0; j < Math.min(16, dimension); j++) {
                        if (z[j] >= 0.0f) {
                            k16 |= (1 << j);
                        }
                    }
                    bucketKey = k16;

                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k];
                        int startDim = (k == 0) ? 0 : tiers[k - 1];
                        int width = tiers[k] - startDim;
                        long baseOffset = tierOffsets[k] + ((long) i * width * 4);
                        for (int l = 0; l < width; l++) {
                            int raw = Float.floatToRawIntBits(z[startDim + l]);
                            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, baseOffset + (l * 4L), raw);
                        }
                        longOffset += count;
                    }
                } else { // 1-bit default
                    long[] packed = transformer.transformAndQuantize(rec.vector());
                    bucketKey = (int) (packed[0] & 0xFFFFL);

                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k];
                        long baseOffset = tierOffsets[k] + (i * (count * 8L));
                        for (int l = 0; l < count; l++) {
                            mapped.set(ValueLayout.JAVA_LONG, baseOffset + (l * 8L), packed[longOffset + l]);
                        }
                        longOffset += count;
                    }
                }
                vectorBucketKeys[i] = bucketKey;
                bucketCounts[bucketKey]++;
            }

            // 4. Write Direct-Mapped Prefix Table Section (CSR format)
            int[] bucketOffsets = new int[PREFIX_OFFSETS_COUNT];
            int runningOffset = 0;
            for (int b = 0; b < NUM_PREFIX_BUCKETS; b++) {
                bucketOffsets[b] = runningOffset;
                runningOffset += bucketCounts[b];
            }
            bucketOffsets[NUM_PREFIX_BUCKETS] = (int) numVectors;

            int[] currentBucketPtrs = Arrays.copyOf(bucketOffsets, NUM_PREFIX_BUCKETS);
            int[] postings = new int[(int) numVectors];
            for (int i = 0; i < numVectors; i++) {
                int b = vectorBucketKeys[i];
                int destPos = currentBucketPtrs[b]++;
                postings[destPos] = i;
            }

            for (int b = 0; b < PREFIX_OFFSETS_COUNT; b++) {
                mapped.set(ValueLayout.JAVA_INT_UNALIGNED, prefixTableOffset + (b * 4L), bucketOffsets[b]);
            }
            long postBase = prefixTableOffset + PREFIX_OFFSETS_BYTES;
            for (int i = 0; i < numVectors; i++) {
                mapped.set(ValueLayout.JAVA_INT_UNALIGNED, postBase + (i * 4L), postings[i]);
            }

            // 5. Write Precision Sidecar Section
            if (sidecarMode == VectorDb.SIDECAR_FP16) {
                for (int i = 0; i < numVectors; i++) {
                    float[] vec = records.get(i).vector();
                    long rowOffset = sidecarOffset + ((long) i * dimension * 2L);
                    for (int d = 0; d < dimension; d++) {
                        short fp16 = Float.floatToFloat16(vec[d]);
                        mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, rowOffset + d * 2L, fp16);
                    }
                }
            } else if (sidecarMode == VectorDb.SIDECAR_FP8) {
                for (int i = 0; i < numVectors; i++) {
                    float[] vec = records.get(i).vector();
                    long rowOffset = sidecarOffset + ((long) i * dimension);
                    for (int d = 0; d < dimension; d++) {
                        byte fp8 = VectorDb.encodeFP8_E4M3(vec[d]);
                        mapped.set(ValueLayout.JAVA_BYTE, rowOffset + d, fp8);
                    }
                }
            } else if (sidecarMode == VectorDb.SIDECAR_FP4) {
                int blockSize = 16;
                int numBlocks = (dimension + blockSize - 1) / blockSize;
                int bytesPerRecord = numBlocks * 9;
                for (int i = 0; i < numVectors; i++) {
                    float[] vec = records.get(i).vector();
                    long rowOffset = sidecarOffset + ((long) i * bytesPerRecord);
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
                        byte scaleByte = VectorDb.encodeFP8_E4M3(scale);
                        long blockOffset = rowOffset + b * 9L;
                        mapped.set(ValueLayout.JAVA_BYTE, blockOffset, scaleByte);

                        float invScale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
                        for (int j = 0; j < 8; j++) {
                            int d0 = blockStart + j * 2;
                            int d1 = blockStart + j * 2 + 1;
                            float v0 = (d0 < dimension) ? vec[d0] * invScale : 0.0f;
                            float v1 = (d1 < dimension) ? vec[d1] * invScale : 0.0f;
                            byte n0 = VectorDb.encodeFP4_E2M1_Nibble(v0);
                            byte n1 = VectorDb.encodeFP4_E2M1_Nibble(v1);
                            byte packed = (byte) ((n1 << 4) | (n0 & 0xF));
                            mapped.set(ValueLayout.JAVA_BYTE, blockOffset + 1 + j, packed);
                        }
                    }
                }
            }

            // 5. Write Generic Metadata Payload Section
            if (metadataPayload != null && metadataPayload.length > 0) {
                for (int i = 0; i < metadataPayload.length; i++) {
                    mapped.set(ValueLayout.JAVA_BYTE, metadataOffset + i, metadataPayload[i]);
                }
            }

            // 6. Write TOC Section
            for (int i = 0; i < tocBytes.length; i++) {
                mapped.set(ValueLayout.JAVA_BYTE, tocOffset + i, tocBytes[i]);
            }

            // 7. Write Trailer (Last 20 Bytes: toc_offset [8B], toc_length [4B], "PITHOSDB" [8B])
            long trailerOffset = totalFileSize - TRAILER_SIZE;
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, trailerOffset, tocOffset);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, trailerOffset + 8L, tocLength);
            for (int i = 0; i < 8; i++) {
                mapped.set(ValueLayout.JAVA_BYTE, trailerOffset + 12L + i, MAGIC_TRAILER[i]);
            }

            mapped.force();
        }
    }
}
