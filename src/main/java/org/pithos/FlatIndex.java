package org.pithos;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadFactory;
import java.util.stream.IntStream;

/// # FlatIndex
///
/// High-performance off-heap memory-mapped multi-tier binary vector index implementing the **3-Gate Read-Path Cascade**:
///
/// ### 3-Gate Read-Path Architecture:
/// 1. **Gate 1 (Metadata & Tombstone Filter):** Reads the 64-bit metadata word m_i. If the tombstone bit `(m_i & 1) == 1`,
///    the record is skipped in zero cycles without accessing tier memory.
/// 2. **Gate 2 (Matryoshka Early-Exit Hamming Scan):** Computes cumulative Hamming distance across active tiers up to T(τ):
///    `d_H(q, d) = ∑_{t=0}^{T(τ)} ∑_{l=0}^{W_t-1} popcount(q_{t, l} ⊕ d_{t, l})`
///    If the running distance exceeds the current top-k threshold (`d_H > d_limit`), computation aborts early.
/// 3. **Gate 3 (Exact In-Engine Reranking):**
///    - **FP16 Sidecar Path:** Exact Euclidean L2 distance calculated directly from IEEE 754 half-precision floats:
///      `d_L2²(q, x) = ∑_{d=0}^{D-1} (q_d - fp16ToFloat(x_d^fp16))²`
///    - **Asymmetric Fallback Path:** Exact continuous query against uncompressed rotated coordinates:
///      `d_asym(z_q, b_d) = ||z_q||² + D + 2 · ∑_{j=0}^{D-1} z_{q, j} - 4 · ∑_{j : b_{d, j} = 1} z_{q, j}`
///
/// Threading is coordinated via an **LMAX Disruptor lock-free ring buffer** with thread-local nearest-neighbor heaps.
public class FlatIndex implements Index {

    public static final float[] FP8_E4M3_LUT = new float[256];
    public static final float[] FP4_E2M1_LUT = VectorDb.FP4_E2M1_TABLE;

    static {
        for (int i = 0; i < 256; i++) {
            FP8_E4M3_LUT[i] = VectorDb.decodeFP8_E4M3((byte) i);
        }
    }

    private final MemorySegment baseSegment;
    private final MemorySegment idsSegment;
    private final MemorySegment[] tierSegments;
    private final MemorySegment metadataSegment;
    private final MemorySegment fp16Segment;
    private final MemorySegment fp8Segment;
    private final MemorySegment fp4Segment;
    private final int sidecarMode;
    private final String userMetadataJson;
    private final MemorySegment metadataPayloadSegment;
    private final boolean isSingleFileContainer;
    private final MemorySegment prefixOffsetsSegment;
    private final MemorySegment prefixPostingsSegment;
    private final boolean hasPrefixTable;

    private final byte planetId;
    private final long planetRadius;
    private final int dimension;
    private final int numTiers;
    private final int[] tiers;
    private final long size;

    private final TransformOperator transformOperator;
    private final float[] cumulativeEnergy;
    private double targetEnergyBudget = 0.90;
    private final int qMode;
    private final int[] tierLongs;
    private final int[] tierOffsets;
    private final int[] tierSizes;
    private final ByteBuffer[] tierVectors;

    private static final int SIMD_FLOAT_DIM_THRESHOLD = 32;

    private static final ThreadLocal<long[]> VISITED_SCRATCH = ThreadLocal.withInitial(() -> new long[1024]);
    private final int numWorkers;
    private volatile long chunkSize = 20000;

    /// Sets the parallel chunk size for work distribution across parallel worker threads.
    ///
    /// @param chunkSize number of records processed per parallel task
    public void setChunkSize(long chunkSize) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("Chunk size must be greater than zero");
        }
        this.chunkSize = chunkSize;
    }

    /// Sets the target cumulative spectral energy budget τ ∈ (0, 1] for Matryoshka early-exit tier truncation.
    ///
    /// @param tau cumulative variance budget (e.g. `0.90` captures 90% of spectral variance)
    public void setTargetEnergyBudget(double tau) {
        if (tau <= 0.0 || tau > 1.0) {
            throw new IllegalArgumentException("Energy budget tau must be in (0, 1]");
        }
        this.targetEnergyBudget = tau;
    }

    /// Returns the active sidecar storage format mode.
    public int getSidecarMode() {
        return sidecarMode;
    }

    /// Returns the embedded user metadata JSON string, or null if none.
    public String getUserMetadataJson() {
        return userMetadataJson;
    }

    /// Returns the raw off-heap metadata payload segment, or null if none.
    public MemorySegment getMetadataPayloadSegment() {
        return metadataPayloadSegment;
    }

    /// Returns true if this index was loaded from a universal single-file container.
    public boolean isSingleFileContainer() {
        return isSingleFileContainer;
    }

    public FlatIndex(MemorySegment baseSegment, MemorySegment idsSegment, MemorySegment[] tierSegments,
            MemorySegment metadataSegment, MemorySegment fp16Segment,
            byte planetId, long planetRadius, int dimension, int numTiers, int[] tiers, long size,
            float[] cumulativeEnergy, int qMode) {
        this(baseSegment, idsSegment, tierSegments, metadataSegment, fp16Segment, null, null,
                planetId, planetRadius, dimension, numTiers, tiers, size, cumulativeEnergy, qMode,
                fp16Segment != null ? VectorDb.SIDECAR_FP16 : VectorDb.SIDECAR_NONE);
    }

    public FlatIndex(MemorySegment baseSegment, MemorySegment idsSegment, MemorySegment[] tierSegments,
            MemorySegment metadataSegment, MemorySegment fp16Segment, MemorySegment fp8Segment, MemorySegment fp4Segment,
            byte planetId, long planetRadius, int dimension, int numTiers, int[] tiers, long size,
            float[] cumulativeEnergy, int qMode, int sidecarMode) {
        this(baseSegment, idsSegment, tierSegments, metadataSegment, fp16Segment, fp8Segment, fp4Segment,
                planetId, planetRadius, dimension, numTiers, tiers, size, cumulativeEnergy, qMode, sidecarMode, null, null);
    }

    public FlatIndex(MemorySegment baseSegment, MemorySegment idsSegment, MemorySegment[] tierSegments,
            MemorySegment metadataSegment, MemorySegment fp16Segment, MemorySegment fp8Segment, MemorySegment fp4Segment,
            byte planetId, long planetRadius, int dimension, int numTiers, int[] tiers, long size,
            float[] cumulativeEnergy, int qMode, int sidecarMode, String userMetadataJson, MemorySegment metadataPayloadSegment) {
        this(baseSegment, idsSegment, tierSegments, metadataSegment, fp16Segment, fp8Segment, fp4Segment,
                planetId, planetRadius, dimension, numTiers, tiers, size, cumulativeEnergy, qMode, sidecarMode,
                userMetadataJson, metadataPayloadSegment, null, null);
    }

    public FlatIndex(MemorySegment baseSegment, MemorySegment idsSegment, MemorySegment[] tierSegments,
            MemorySegment metadataSegment, MemorySegment fp16Segment, MemorySegment fp8Segment, MemorySegment fp4Segment,
            byte planetId, long planetRadius, int dimension, int numTiers, int[] tiers, long size,
            float[] cumulativeEnergy, int qMode, int sidecarMode, String userMetadataJson, MemorySegment metadataPayloadSegment,
            MemorySegment prefixOffsetsSegment, MemorySegment prefixPostingsSegment) {
        this.baseSegment = baseSegment;
        this.idsSegment = idsSegment;
        this.tierSegments = tierSegments;
        this.metadataSegment = metadataSegment;
        this.fp16Segment = fp16Segment;
        this.fp8Segment = fp8Segment;
        this.fp4Segment = fp4Segment;
        this.sidecarMode = sidecarMode;
        this.planetId = planetId;
        this.planetRadius = planetRadius;
        this.dimension = dimension;
        this.numTiers = numTiers;
        this.tiers = tiers;
        this.size = size;
        this.cumulativeEnergy = cumulativeEnergy;
        this.qMode = qMode;
        this.userMetadataJson = userMetadataJson;
        this.metadataPayloadSegment = metadataPayloadSegment;
        this.isSingleFileContainer = (userMetadataJson != null || metadataPayloadSegment != null);
        this.prefixOffsetsSegment = prefixOffsetsSegment;
        this.prefixPostingsSegment = prefixPostingsSegment;
        this.hasPrefixTable = (prefixOffsetsSegment != null && prefixPostingsSegment != null);

        this.tierLongs = new int[numTiers];
        int prevBoundVal = 0;
        for (int idx = 0; idx < numTiers; idx++) {
            this.tierLongs[idx] = (tiers[idx] - prevBoundVal) / 64;
            prevBoundVal = tiers[idx];
        }

        this.tierOffsets = new int[numTiers];
        this.tierSizes = new int[numTiers];
        this.tierVectors = new ByteBuffer[numTiers];

        int offset = 0;
        for (int idx = 0; idx < numTiers; idx++) {
            this.tierOffsets[idx] = offset;
            this.tierSizes[idx] = tiers[idx] - (idx == 0 ? 0 : tiers[idx - 1]);
            this.tierVectors[idx] = tierSegments[idx].asByteBuffer();
            offset += this.tierLongs[idx];
        }

        this.transformOperator = new TransformOperator(dimension, tiers);
        this.numWorkers = Runtime.getRuntime().availableProcessors();
    }

    /// Returns the virtual memory address of the given tier segment for zero-copy DMA/FPGA access.
    public long getTierAddress(int tierIdx) {
        if (tierIdx < 0 || tierIdx >= numTiers)
            return 0;
        return tierSegments[tierIdx].address();
    }

    /// Returns the byte size of the given tier segment.
    public long getTierByteSize(int tierIdx) {
        if (tierIdx < 0 || tierIdx >= numTiers)
            return 0;
        return tierSegments[tierIdx].byteSize();
    }

    /// Returns the virtual memory address of the metadata column segment.
    public long getMetadataAddress() {
        return metadataSegment.address();
    }

    /// Returns the byte size of the metadata column segment.
    public long getMetadataByteSize() {
        return metadataSegment.byteSize();
    }

    /// Returns the virtual memory address of the IDs column segment.
    public long getIdsAddress() {
        return idsSegment.address();
    }

    /// Returns the byte size of the IDs column segment.
    public long getIdsByteSize() {
        return idsSegment.byteSize();
    }

    /// Returns true if this index contains a direct-mapped Gate 0 prefix table.
    public boolean hasPrefixTable() {
        return hasPrefixTable;
    }

    /// Returns the virtual memory segment holding the 16-bit prefix offset array (65537 int32s).
    public MemorySegment getPrefixOffsetsSegment() {
        return prefixOffsetsSegment;
    }

    /// Returns the virtual memory segment holding the prefix postings array.
    public MemorySegment getPrefixPostingsSegment() {
        return prefixPostingsSegment;
    }

    /// Returns the `TransformOperator` configured for this index.
    public TransformOperator getTransformOperator() {
        return transformOperator;
    }

    /// Maps an existing multi-tier index from disk off-heap into virtual memory.
    ///
    /// @param basePath filepath without suffix
    /// @param weights optional projection weights for SVD energy calculation
    /// @param loraDim bottleneck dimension
    /// @return memory-mapped `FlatIndex`
    /// @throws IOException on I/O error
    public static FlatIndex mapFile(String basePath, float[] weights, int loraDim) throws IOException {
        Path mainPath = Path.of(basePath);
        if (!Files.exists(mainPath) && Files.exists(Path.of(basePath + ".pithos"))) {
            mainPath = Path.of(basePath + ".pithos");
        }
        if (!Files.exists(mainPath)) {
            throw new IOException("Base file path does not exist: " + basePath);
        }

        // 1. Check if this is a Single-File .pithos Container (DIOGENES magic)
        if (PithosContainer.isPithosContainer(mainPath)) {
            try (FileChannel channel = FileChannel.open(mainPath, StandardOpenOption.READ)) {
                PithosContainer.Superblock sb = PithosContainer.readSuperblock(channel);
                PithosContainer.validateTrailer(channel, sb.tocOffset(), sb.tocLength());
                String tocJson = PithosContainer.readTocJson(channel, sb.tocOffset(), sb.tocLength());

                long totalRecords = sb.numVectors();
                int dimension = sb.dimension();
                int numTiers = sb.numTiers();
                int[] tiers = sb.tiers();
                int qMode = sb.qMode();
                int sidecarMode = sb.sidecarType();

                MemorySegment containerSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), Arena.global());

                long idsOffset = PithosContainer.align64(PithosContainer.SUPERBLOCK_SIZE);
                long idsLength = totalRecords * 8L;
                MemorySegment idsSegment = containerSegment.asSlice(idsOffset, idsLength);

                MemorySegment[] tierSegments = new MemorySegment[numTiers];
                int prevBound = 0;
                long currentOffset = PithosContainer.align64(idsOffset + idsLength);
                for (int k = 0; k < numTiers; k++) {
                    int width = tiers[k] - prevBound;
                    long bytesPerRecord = switch (qMode) {
                        case 1 -> (width / 4);
                        case 2 -> (width * 4L);
                        default -> (width / 8);
                    };
                    long tierLen = totalRecords * bytesPerRecord;
                    tierSegments[k] = containerSegment.asSlice(currentOffset, tierLen);
                    currentOffset = PithosContainer.align64(currentOffset + tierLen);
                    prevBound = tiers[k];
                }

                MemorySegment fp16Segment = null;
                MemorySegment fp8Segment = null;
                MemorySegment fp4Segment = null;
                if (sidecarMode == VectorDb.SIDECAR_FP16) {
                    long sidecarLen = totalRecords * dimension * 2L;
                    fp16Segment = containerSegment.asSlice(currentOffset, sidecarLen);
                    currentOffset = PithosContainer.align64(currentOffset + sidecarLen);
                } else if (sidecarMode == VectorDb.SIDECAR_FP8) {
                    long sidecarLen = totalRecords * dimension * 1L;
                    fp8Segment = containerSegment.asSlice(currentOffset, sidecarLen);
                    currentOffset = PithosContainer.align64(currentOffset + sidecarLen);
                } else if (sidecarMode == VectorDb.SIDECAR_FP4) {
                    int blockSize = 16;
                    int numBlocks = (dimension + blockSize - 1) / blockSize;
                    long sidecarLen = totalRecords * (numBlocks * 9L);
                    fp4Segment = containerSegment.asSlice(currentOffset, sidecarLen);
                    currentOffset = PithosContainer.align64(currentOffset + sidecarLen);
                }

                PithosContainer.Section prefixSec = PithosContainer.extractPrefixTableSection(tocJson);
                MemorySegment prefixOffsetsSegment = null;
                MemorySegment prefixPostingsSegment = null;
                if (prefixSec.offset() > 0 && prefixSec.length() >= PithosContainer.MIH_OFFSETS_BYTES) {
                    long offSegLen = PithosContainer.MIH_OFFSETS_BYTES;
                    long postSegLen = totalRecords * 4L * PithosContainer.NUM_MIH_CHUNKS;
                    prefixOffsetsSegment = containerSegment.asSlice(prefixSec.offset(), offSegLen);
                    prefixPostingsSegment = containerSegment.asSlice(prefixSec.offset() + offSegLen, postSegLen);
                }

                PithosContainer.Section metaSec = PithosContainer.extractMetadataSection(tocJson);
                MemorySegment metadataPayloadSegment = null;
                if (metaSec.offset() > 0 && metaSec.length() > 0) {
                    metadataPayloadSegment = containerSegment.asSlice(metaSec.offset(), metaSec.length());
                }

                float[] cumulativeEnergy = new float[numTiers];
                if (weights != null) {
                    float[] allPhi = TransformOperator.computeCumulativeEnergy(weights, dimension, loraDim);
                    for (int k = 0; k < numTiers; k++) {
                        cumulativeEnergy[k] = allPhi[tiers[k] - 1];
                    }
                } else {
                    for (int k = 0; k < numTiers; k++) {
                        cumulativeEnergy[k] = (float) tiers[k] / dimension;
                    }
                }

                return new FlatIndex(containerSegment, idsSegment, tierSegments, null, fp16Segment, fp8Segment, fp4Segment,
                        (byte) 0, 0L, dimension, numTiers, tiers, totalRecords, cumulativeEnergy, qMode, sidecarMode,
                        tocJson, metadataPayloadSegment, prefixOffsetsSegment, prefixPostingsSegment);
            }
        }

        // 2. Legacy Multi-File PLAN Layout Fallback
        MemorySegment mappedBase;
        try (FileChannel channel = FileChannel.open(mainPath, StandardOpenOption.READ)) {
            mappedBase = channel.map(FileChannel.MapMode.READ_ONLY, 0, 64, Arena.global());
        }

        byte m0 = mappedBase.get(ValueLayout.JAVA_BYTE, 0);
        byte m1 = mappedBase.get(ValueLayout.JAVA_BYTE, 1);
        byte m2 = mappedBase.get(ValueLayout.JAVA_BYTE, 2);
        byte m3 = mappedBase.get(ValueLayout.JAVA_BYTE, 3);
        if (m0 != 'P' || m1 != 'L' || m2 != 'A' || m3 != 'N') {
            throw new IllegalArgumentException("Invalid file magic: must be DIOGENES or PLAN");
        }

        byte planetId = mappedBase.get(ValueLayout.JAVA_BYTE, 4);
        long totalRecords = mappedBase.get(ValueLayout.JAVA_LONG_UNALIGNED, 5);
        long planetRadius = mappedBase.get(ValueLayout.JAVA_LONG_UNALIGNED, 13);
        int dimension = mappedBase.get(ValueLayout.JAVA_INT_UNALIGNED, 21);
        int numTiers = mappedBase.get(ValueLayout.JAVA_INT_UNALIGNED, 25);

        int[] tiers = new int[numTiers];
        for (int i = 0; i < numTiers; i++) {
            tiers[i] = mappedBase.get(ValueLayout.JAVA_INT_UNALIGNED, 29 + (i * 4));
        }

        byte qModeByte = mappedBase.get(ValueLayout.JAVA_BYTE, 61);
        int qMode = qModeByte & 0xFF;

        Path idsPath = Path.of(basePath + "_ids.bin");
        MemorySegment idsSegment;
        try (FileChannel channel = FileChannel.open(idsPath, StandardOpenOption.READ)) {
            idsSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * 8, Arena.global());
        }

        Path metadataPath = Path.of(basePath + "_metadata.bin");
        MemorySegment metadataSegment;
        try (FileChannel channel = FileChannel.open(metadataPath, StandardOpenOption.READ)) {
            metadataSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * 8, Arena.global());
        }

        MemorySegment[] tierSegments = new MemorySegment[numTiers];
        int prevBound = 0;
        for (int k = 0; k < numTiers; k++) {
            int width = tiers[k] - prevBound;
            Path tierPath = Path.of(basePath + "_tier_" + k + ".bin");
            long bytesPerRecord = switch (qMode) {
                case 1 -> (width / 4);
                case 2 -> (width * 4L);
                default -> (width / 8);
            };
            try (FileChannel channel = FileChannel.open(tierPath, StandardOpenOption.READ)) {
                tierSegments[k] = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * bytesPerRecord,
                        Arena.global());
            }
            prevBound = tiers[k];
        }

        float[] cumulativeEnergy = new float[numTiers];
        if (weights != null) {
            float[] allPhi = TransformOperator.computeCumulativeEnergy(weights, dimension, loraDim);
            for (int k = 0; k < numTiers; k++) {
                cumulativeEnergy[k] = allPhi[tiers[k] - 1];
            }
        } else {
            for (int k = 0; k < numTiers; k++) {
                cumulativeEnergy[k] = (float) tiers[k] / dimension;
            }
        }

        byte sidecarModeByte = mappedBase.get(ValueLayout.JAVA_BYTE, 62);
        int sidecarMode = sidecarModeByte & 0xFF;

        MemorySegment fp16Segment = null;
        Path fp16Path = Path.of(basePath + "_fp16.bin");
        if (fp16Path.toFile().exists()) {
            try (FileChannel channel = FileChannel.open(fp16Path, StandardOpenOption.READ)) {
                fp16Segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * dimension * 2L,
                        Arena.global());
            }
        }

        MemorySegment fp8Segment = null;
        Path fp8Path = Path.of(basePath + "_fp8.bin");
        if (fp8Path.toFile().exists()) {
            try (FileChannel channel = FileChannel.open(fp8Path, StandardOpenOption.READ)) {
                fp8Segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * dimension * 1L,
                        Arena.global());
            }
        }

        MemorySegment fp4Segment = null;
        Path fp4Path = Path.of(basePath + "_fp4.bin");
        if (fp4Path.toFile().exists()) {
            int numBlocks = (dimension + 15) / 16;
            long bytesPerRecord = numBlocks * 9L;
            try (FileChannel channel = FileChannel.open(fp4Path, StandardOpenOption.READ)) {
                fp4Segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * bytesPerRecord,
                        Arena.global());
            }
        }

        return new FlatIndex(mappedBase, idsSegment, tierSegments, metadataSegment, fp16Segment, fp8Segment, fp4Segment,
                planetId, planetRadius, dimension, numTiers, tiers, totalRecords, cumulativeEnergy, qMode, sidecarMode);
    }

    @Override
    public void insert(VectorRecord record) {
        throw new UnsupportedOperationException("Insert is not supported on read-only memory-mapped Index.");
    }

    @Override
    public List<SearchResult> search(float[] query, int k) {
        List<SearchResult>[] results = batchSearch(new float[][] { query }, k);
        return results[0];
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<SearchResult>[] batchSearch(float[][] queries, int k) {
        if (queries == null || queries.length == 0)
            return new List[0];
        if (k <= 0 || size == 0) {
            List<SearchResult>[] empty = new List[queries.length];
            Arrays.fill(empty, List.of());
            return empty;
        }

        if (hasPrefixTable) {
            return searchWithPrefixRouting(queries, k);
        }

        return searchLinearScanParallel(queries, k);
    }

    @SuppressWarnings("unchecked")
    private List<SearchResult>[] searchLinearScanParallel(float[][] queries, int k) {
        int numQueries = queries.length;
        int tVal = 0;
        for (int i = 0; i < numTiers; i++) {
            if (cumulativeEnergy[i] >= targetEnergyBudget) {
                tVal = i;
                break;
            }
        }
        final int activeT = tVal;
        int kCandidate = (int) Math.min(size, (fp16Segment != null || fp8Segment != null || fp4Segment != null)
                ? Math.max(100, 20 * k)
                : Math.max(50, 3 * k));

        List<SearchResult>[] finalResults = new List[numQueries];

        IntStream.range(0, numQueries).parallel().forEach(q -> {
            float[] query = queries[q];
            float[] zQuery = transformOperator.preconditionAndRotate(query);

            long[] bQuery;
            long[] bQueryMask = null;
            if (qMode == 1) { // 2-bit
                float qThreshold = TransformOperator.calculatePercentileThreshold(zQuery, 0.20f);
                long[][] packed = transformOperator.quantize2Bit(zQuery, qThreshold);
                bQuery = packed[0];
                bQueryMask = packed[1];
            } else if (qMode == 0) { // 1-bit
                bQuery = transformOperator.transformAndQuantize(query);
            } else {
                bQuery = null;
            }

            int[] topDists = new int[kCandidate];
            long[] topRowIds = new long[kCandidate];
            Arrays.fill(topDists, Integer.MAX_VALUE);

            for (long rowIdx = 0; rowIdx < size; rowIdx++) {
                if (metadataSegment != null && (metadataSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8L) & 1L) == 1L) {
                    continue;
                }

                int currentLimit = topDists[kCandidate - 1];
                int totalDist = 0;

                if (qMode == 1) { // 2-bit
                    for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                        int numLongs = tierLongs[tierIdx];
                        int offset = tierOffsets[tierIdx];
                        MemorySegment tierSeg = tierSegments[tierIdx];
                        long baseOffset = rowIdx * (numLongs * 16L);
                        int tierDist = 0;
                        int l = 0;
                        for (; l + 3 < numLongs; l += 4) {
                            long s0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                            long m0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + (l * 8L));
                            long s1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 1) * 8L));
                            long m1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 1) * 8L));
                            long s2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 2) * 8L));
                            long m2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 2) * 8L));
                            long s3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 3) * 8L));
                            long m3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 3) * 8L));

                            tierDist += 4 * Long.bitCount(m0 & bQueryMask[offset + l] & (s0 ^ bQuery[offset + l])) + Long.bitCount(m0 ^ bQueryMask[offset + l])
                                      + 4 * Long.bitCount(m1 & bQueryMask[offset + l + 1] & (s1 ^ bQuery[offset + l + 1])) + Long.bitCount(m1 ^ bQueryMask[offset + l + 1])
                                      + 4 * Long.bitCount(m2 & bQueryMask[offset + l + 2] & (s2 ^ bQuery[offset + l + 2])) + Long.bitCount(m2 ^ bQueryMask[offset + l + 2])
                                      + 4 * Long.bitCount(m3 & bQueryMask[offset + l + 3] & (s3 ^ bQuery[offset + l + 3])) + Long.bitCount(m3 ^ bQueryMask[offset + l + 3]);
                        }
                        for (; l < numLongs; l++) {
                            long dbSign = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                            long dbMask = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + (l * 8L));
                            long qSign = bQuery[offset + l];
                            long qMask = bQueryMask[offset + l];
                            tierDist += 4 * Long.bitCount(dbMask & qMask & (dbSign ^ qSign)) + Long.bitCount(dbMask ^ qMask);
                        }
                        totalDist += tierDist;
                        if (totalDist > currentLimit) break;
                    }
                } else if (qMode == 2) { // Float
                    float[] dbFloat = new float[dimension];
                    int dimOffset = 0;
                    for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                        int width = tiers[tierIdx] - (tierIdx == 0 ? 0 : tiers[tierIdx - 1]);
                        long baseOffset = rowIdx * (width * 4L);
                        MemorySegment.copy(tierSegments[tierIdx], ValueLayout.JAVA_FLOAT, baseOffset, dbFloat, dimOffset, width);
                        dimOffset += width;
                    }
                    totalDist = (int) (transformOperator.computeL2Float(zQuery, dbFloat) * 1000f);
                } else { // 1-bit unrolled 8x
                    for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                        int numLongs = tierLongs[tierIdx];
                        int offset = tierOffsets[tierIdx];
                        MemorySegment tierSeg = tierSegments[tierIdx];
                        long baseOffset = rowIdx * (numLongs * 8L);
                        int tierDist = 0;
                        int l = 0;
                        for (; l + 7 < numLongs; l += 8) {
                            long w0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                            long w1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 1) * 8L));
                            long w2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 2) * 8L));
                            long w3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 3) * 8L));
                            long w4 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 4) * 8L));
                            long w5 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 5) * 8L));
                            long w6 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 6) * 8L));
                            long w7 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 7) * 8L));

                            tierDist += Long.bitCount(bQuery[offset + l] ^ w0)
                                      + Long.bitCount(bQuery[offset + l + 1] ^ w1)
                                      + Long.bitCount(bQuery[offset + l + 2] ^ w2)
                                      + Long.bitCount(bQuery[offset + l + 3] ^ w3)
                                      + Long.bitCount(bQuery[offset + l + 4] ^ w4)
                                      + Long.bitCount(bQuery[offset + l + 5] ^ w5)
                                      + Long.bitCount(bQuery[offset + l + 6] ^ w6)
                                      + Long.bitCount(bQuery[offset + l + 7] ^ w7);
                        }
                        for (; l < numLongs; l++) {
                            long dbWord = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                            tierDist += Long.bitCount(bQuery[offset + l] ^ dbWord);
                        }
                        totalDist += tierDist;
                        if (totalDist > currentLimit) break;
                    }
                }

                if (totalDist < currentLimit) {
                    int pos = kCandidate - 1;
                    while (pos > 0 && totalDist < topDists[pos - 1]) {
                        topDists[pos] = topDists[pos - 1];
                        topRowIds[pos] = topRowIds[pos - 1];
                        pos--;
                    }
                    topDists[pos] = totalDist;
                    topRowIds[pos] = rowIdx;
                }
            }

            List<Long> candidates = new ArrayList<>();
            for (int i = 0; i < kCandidate; i++) {
                if (topDists[i] != Integer.MAX_VALUE) {
                    candidates.add(topRowIds[i]);
                }
            }

            if (qMode == 2) {
                List<SearchResult> queryResults = new ArrayList<>();
                int limit = Math.min(k, candidates.size());
                for (int i = 0; i < limit; i++) {
                    long rowIdx = candidates.get(i);
                    long recordId = idsSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8L);
                    queryResults.add(new SearchResult(recordId, topDists[i]));
                }
                finalResults[q] = queryResults;
                return;
            }

            // Gate 3: Precision Sidecar Reranking with Early Distance Cutoff
            double[] bestDists = new double[k];
            long[] bestRowIds = new long[k];
            Arrays.fill(bestDists, Double.MAX_VALUE);

            if (fp8Segment != null) {
                float[] queryLut = new float[dimension * 256];
                for (int d = 0; d < dimension; d++) {
                    float qVal = query[d];
                    int dBase = d << 8;
                    for (int b = 0; b < 256; b++) {
                        float diff = qVal - FP8_E4M3_LUT[b];
                        queryLut[dBase | b] = diff * diff;
                    }
                }
                for (long rowIdx : candidates) {
                    double currentLimit = bestDists[k - 1];
                    double dist = computeExactL2FP8_LUT(queryLut, rowIdx, currentLimit);
                    if (dist < currentLimit) {
                        int pos = k - 1;
                        while (pos > 0 && dist < bestDists[pos - 1]) {
                            bestDists[pos] = bestDists[pos - 1];
                            bestRowIds[pos] = bestRowIds[pos - 1];
                            pos--;
                        }
                        bestDists[pos] = dist;
                        bestRowIds[pos] = rowIdx;
                    }
                }
            } else if (fp4Segment != null) {
                int numBlocks = (dimension + 15) / 16;
                int bytesPerRecord = numBlocks * 9;
                byte[] localFp4 = new byte[bytesPerRecord];
                for (long rowIdx : candidates) {
                    double currentLimit = bestDists[k - 1];
                    double dist = computeExactL2FP4(query, rowIdx, localFp4, numBlocks, currentLimit);
                    if (dist < currentLimit) {
                        int pos = k - 1;
                        while (pos > 0 && dist < bestDists[pos - 1]) {
                            bestDists[pos] = bestDists[pos - 1];
                            bestRowIds[pos] = bestRowIds[pos - 1];
                            pos--;
                        }
                        bestDists[pos] = dist;
                        bestRowIds[pos] = rowIdx;
                    }
                }
            } else if (fp16Segment != null) {
                short[] localFp16 = new short[dimension];
                for (long rowIdx : candidates) {
                    double currentLimit = bestDists[k - 1];
                    double dist = computeExactL2FP16(query, rowIdx, localFp16, currentLimit);
                    if (dist < currentLimit) {
                        int pos = k - 1;
                        while (pos > 0 && dist < bestDists[pos - 1]) {
                            bestDists[pos] = bestDists[pos - 1];
                            bestRowIds[pos] = bestRowIds[pos - 1];
                            pos--;
                        }
                        bestDists[pos] = dist;
                        bestRowIds[pos] = rowIdx;
                    }
                }
            } else {
                double queryL2Norm = 0.0;
                double querySum = 0.0;
                for (float val : zQuery) {
                    queryL2Norm += val * val;
                    querySum += val;
                }
                for (long rowIdx : candidates) {
                    double currentLimit = bestDists[k - 1];
                    double dist = computeAsymmetricL2DistanceOffHeap(zQuery, queryL2Norm, querySum, rowIdx);
                    if (dist < currentLimit) {
                        int pos = k - 1;
                        while (pos > 0 && dist < bestDists[pos - 1]) {
                            bestDists[pos] = bestDists[pos - 1];
                            bestRowIds[pos] = bestRowIds[pos - 1];
                            pos--;
                        }
                        bestDists[pos] = dist;
                        bestRowIds[pos] = rowIdx;
                    }
                }
            }

            List<SearchResult> queryResults = new ArrayList<>(k);
            for (int i = 0; i < k; i++) {
                if (bestDists[i] != Double.MAX_VALUE) {
                    long recordId = idsSegment.get(ValueLayout.JAVA_LONG, bestRowIds[i] * 8L);
                    queryResults.add(new SearchResult(recordId, (int) (bestDists[i] * 1000000.0)));
                }
            }
            finalResults[q] = queryResults;
        });

        return finalResults;
    }

    /// Evaluates queries using Gate 0 direct-mapped 16-bit prefix routing ($2^{16} = 65,536$ buckets).
    @SuppressWarnings("unchecked")
    private List<SearchResult>[] searchWithPrefixRouting(float[][] queries, int k) {
        int numQueries = queries.length;
        List<SearchResult>[] finalResults = new List[numQueries];

        int tVal = 0;
        for (int i = 0; i < numTiers; i++) {
            if (cumulativeEnergy[i] >= targetEnergyBudget) {
                tVal = i;
                break;
            }
        }
        final int activeT = tVal;

        int kCandidate = (int) Math.min(size, (fp16Segment != null || fp8Segment != null || fp4Segment != null)
                ? Math.max(300, 30 * k)
                : Math.max(50, 3 * k));

        IntStream.range(0, numQueries).parallel().forEach(q -> {
            float[] query = queries[q];
            float[] zQuery = transformOperator.preconditionAndRotate(query);

            long[] bQuery;
            long[] bQueryMask = null;
            int[] queryChunkKeys = new int[4];
            if (qMode == 1) { // 2-bit QJL
                float qThreshold = TransformOperator.calculatePercentileThreshold(zQuery, 0.20f);
                long[][] packed = transformOperator.quantize2Bit(zQuery, qThreshold);
                bQuery = packed[0];
                bQueryMask = packed[1];
                long w0 = bQuery[0];
                queryChunkKeys[0] = (int) (w0 & 0xFFL);
                queryChunkKeys[1] = (int) ((w0 >> 8) & 0xFFL);
                queryChunkKeys[2] = (int) ((w0 >> 16) & 0xFFL);
                queryChunkKeys[3] = (int) ((w0 >> 24) & 0xFFL);
            } else if (qMode == 0) { // 1-bit
                bQuery = transformOperator.transformAndQuantize(query);
                long w0 = bQuery[0];
                queryChunkKeys[0] = (int) (w0 & 0xFFL);
                queryChunkKeys[1] = (int) ((w0 >> 8) & 0xFFL);
                queryChunkKeys[2] = (int) ((w0 >> 16) & 0xFFL);
                queryChunkKeys[3] = (int) ((w0 >> 24) & 0xFFL);
            } else {
                bQuery = null;
                int k32 = 0;
                for (int bit = 0; bit < Math.min(32, dimension); bit++) {
                    if (zQuery[bit] >= 0.0f) {
                        k32 |= (1 << bit);
                    }
                }
                queryChunkKeys[0] = (k32 & 0xFF);
                queryChunkKeys[1] = ((k32 >> 8) & 0xFF);
                queryChunkKeys[2] = ((k32 >> 16) & 0xFF);
                queryChunkKeys[3] = ((k32 >> 24) & 0xFF);
            }

            int[] topDists = new int[kCandidate];
            long[] topRowIds = new long[kCandidate];
            Arrays.fill(topDists, Integer.MAX_VALUE);

            int visitedWords = (int) ((size + 63) >>> 6);
            long[] visited = VISITED_SCRATCH.get();
            if (visited == null || visited.length < visitedWords) {
                visited = new long[visitedWords];
                VISITED_SCRATCH.set(visited);
            } else {
                Arrays.fill(visited, 0, visitedWords, 0L);
            }

            // Adaptive multi-pass probing:
            // Pass 1: Exact matches across all 4 chunks (4 buckets total)
            int gatheredCount = 0;
            for (int c = 0; c < 4; c++) {
                int qKey = queryChunkKeys[c];
                long chunkOffBase = c * (256L + 1L) * 4L;
                long chunkPostBase = (long) c * size * 4L;
                int start = prefixOffsetsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkOffBase + (qKey * 4L));
                int end = prefixOffsetsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkOffBase + ((qKey + 1) * 4L));
                for (int p = start; p < end; p++) {
                    int rowIdxInt = prefixPostingsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkPostBase + (p * 4L));
                    if (rowIdxInt < 0 || rowIdxInt >= size) continue;
                    int wordIdx = rowIdxInt >>> 6;
                    long mask = 1L << (rowIdxInt & 63);
                    if ((visited[wordIdx] & mask) != 0) continue;
                    visited[wordIdx] |= mask;
                    gatheredCount++;
                    long rowIdx = rowIdxInt & 0xFFFFFFFFL;

                    if (metadataSegment != null && (metadataSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8L) & 1L) == 1L) {
                        continue;
                    }

                    int currentLimit = topDists[kCandidate - 1];
                    int totalDist = 0;

                    if (qMode == 1) { // 2-bit
                        for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                            int numLongs = tierLongs[tierIdx];
                            int offset = tierOffsets[tierIdx];
                            MemorySegment tierSeg = tierSegments[tierIdx];
                            long baseOffset = rowIdx * (numLongs * 16L);
                            int tierDist = 0;
                            int l = 0;
                            for (; l + 3 < numLongs; l += 4) {
                                long s0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                long m0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + (l * 8L));
                                long s1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 1) * 8L));
                                long m1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 1) * 8L));
                                long s2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 2) * 8L));
                                long m2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 2) * 8L));
                                long s3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 3) * 8L));
                                long m3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 3) * 8L));

                                tierDist += 4 * Long.bitCount(m0 & bQueryMask[offset + l] & (s0 ^ bQuery[offset + l])) + Long.bitCount(m0 ^ bQueryMask[offset + l])
                                          + 4 * Long.bitCount(m1 & bQueryMask[offset + l + 1] & (s1 ^ bQuery[offset + l + 1])) + Long.bitCount(m1 ^ bQueryMask[offset + l + 1])
                                          + 4 * Long.bitCount(m2 & bQueryMask[offset + l + 2] & (s2 ^ bQuery[offset + l + 2])) + Long.bitCount(m2 ^ bQueryMask[offset + l + 2])
                                          + 4 * Long.bitCount(m3 & bQueryMask[offset + l + 3] & (s3 ^ bQuery[offset + l + 3])) + Long.bitCount(m3 ^ bQueryMask[offset + l + 3]);
                            }
                            for (; l < numLongs; l++) {
                                long dbSign = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                long dbMask = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + (l * 8L));
                                long qSign = bQuery[offset + l];
                                long qMask = bQueryMask[offset + l];
                                tierDist += 4 * Long.bitCount(dbMask & qMask & (dbSign ^ qSign)) + Long.bitCount(dbMask ^ qMask);
                            }
                            totalDist += tierDist;
                            if (totalDist > currentLimit) break;
                        }
                    } else if (qMode == 2) { // Float-Hybrid
                        float[] dbFloat = new float[dimension];
                        int dimOffset = 0;
                        for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                            int width = tiers[tierIdx] - (tierIdx == 0 ? 0 : tiers[tierIdx - 1]);
                            long baseOffset = rowIdx * (width * 4L);
                            MemorySegment.copy(tierSegments[tierIdx], ValueLayout.JAVA_FLOAT, baseOffset, dbFloat, dimOffset, width);
                            dimOffset += width;
                        }
                        totalDist = (int) (transformOperator.computeL2Float(zQuery, dbFloat) * 1000f);
                    } else { // 1-bit unrolled 8x
                        for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                            int numLongs = tierLongs[tierIdx];
                            int offset = tierOffsets[tierIdx];
                            MemorySegment tierSeg = tierSegments[tierIdx];
                            long baseOffset = rowIdx * (numLongs * 8L);
                            int tierDist = 0;
                            int l = 0;
                            for (; l + 7 < numLongs; l += 8) {
                                long w0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                long w1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 1) * 8L));
                                long w2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 2) * 8L));
                                long w3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 3) * 8L));
                                long w4 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 4) * 8L));
                                long w5 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 5) * 8L));
                                long w6 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 6) * 8L));
                                long w7 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 7) * 8L));

                                tierDist += Long.bitCount(bQuery[offset + l] ^ w0)
                                          + Long.bitCount(bQuery[offset + l + 1] ^ w1)
                                          + Long.bitCount(bQuery[offset + l + 2] ^ w2)
                                          + Long.bitCount(bQuery[offset + l + 3] ^ w3)
                                          + Long.bitCount(bQuery[offset + l + 4] ^ w4)
                                          + Long.bitCount(bQuery[offset + l + 5] ^ w5)
                                          + Long.bitCount(bQuery[offset + l + 6] ^ w6)
                                          + Long.bitCount(bQuery[offset + l + 7] ^ w7);
                            }
                            for (; l < numLongs; l++) {
                                long dbWord = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                tierDist += Long.bitCount(bQuery[offset + l] ^ dbWord);
                            }
                            totalDist += tierDist;
                            if (totalDist > currentLimit) break;
                        }
                    }

                    if (totalDist < currentLimit) {
                        int pos = kCandidate - 1;
                        while (pos > 0 && totalDist < topDists[pos - 1]) {
                            topDists[pos] = topDists[pos - 1];
                            topRowIds[pos] = topRowIds[pos - 1];
                            pos--;
                        }
                        topDists[pos] = totalDist;
                        topRowIds[pos] = rowIdx;
                    }
                }
            }

            // Pass 2: Hamming-1 neighbor buckets if candidate pool needs more diversity
            if (gatheredCount < kCandidate * 2) {
                for (int c = 0; c < 4; c++) {
                    int qKey = queryChunkKeys[c];
                    long chunkOffBase = c * (256L + 1L) * 4L;
                    long chunkPostBase = (long) c * size * 4L;
                    for (int bit = 0; bit < 8; bit++) {
                        int bKey = qKey ^ (1 << bit);
                        int start = prefixOffsetsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkOffBase + (bKey * 4L));
                        int end = prefixOffsetsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkOffBase + ((bKey + 1) * 4L));
                        for (int p = start; p < end; p++) {
                            int rowIdxInt = prefixPostingsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkPostBase + (p * 4L));
                            if (rowIdxInt < 0 || rowIdxInt >= size) continue;
                            int wordIdx = rowIdxInt >>> 6;
                            long mask = 1L << (rowIdxInt & 63);
                            if ((visited[wordIdx] & mask) != 0) continue;
                            visited[wordIdx] |= mask;
                            gatheredCount++;
                            long rowIdx = rowIdxInt & 0xFFFFFFFFL;

                            if (metadataSegment != null && (metadataSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8L) & 1L) == 1L) {
                                continue;
                            }

                            int currentLimit = topDists[kCandidate - 1];
                            int totalDist = 0;

                            if (qMode == 1) { // 2-bit
                                for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                                    int numLongs = tierLongs[tierIdx];
                                    int offset = tierOffsets[tierIdx];
                                    MemorySegment tierSeg = tierSegments[tierIdx];
                                    long baseOffset = rowIdx * (numLongs * 16L);
                                    int tierDist = 0;
                                    int l = 0;
                                    for (; l + 3 < numLongs; l += 4) {
                                        long s0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                        long m0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + (l * 8L));
                                        long s1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 1) * 8L));
                                        long m1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 1) * 8L));
                                        long s2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 2) * 8L));
                                        long m2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 2) * 8L));
                                        long s3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 3) * 8L));
                                        long m3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 3) * 8L));

                                        tierDist += 4 * Long.bitCount(m0 & bQueryMask[offset + l] & (s0 ^ bQuery[offset + l])) + Long.bitCount(m0 ^ bQueryMask[offset + l])
                                                  + 4 * Long.bitCount(m1 & bQueryMask[offset + l + 1] & (s1 ^ bQuery[offset + l + 1])) + Long.bitCount(m1 ^ bQueryMask[offset + l + 1])
                                                  + 4 * Long.bitCount(m2 & bQueryMask[offset + l + 2] & (s2 ^ bQuery[offset + l + 2])) + Long.bitCount(m2 ^ bQueryMask[offset + l + 2])
                                                  + 4 * Long.bitCount(m3 & bQueryMask[offset + l + 3] & (s3 ^ bQuery[offset + l + 3])) + Long.bitCount(m3 ^ bQueryMask[offset + l + 3]);
                                    }
                                    for (; l < numLongs; l++) {
                                        long dbSign = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                        long dbMask = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + (l * 8L));
                                        long qSign = bQuery[offset + l];
                                        long qMask = bQueryMask[offset + l];
                                        tierDist += 4 * Long.bitCount(dbMask & qMask & (dbSign ^ qSign)) + Long.bitCount(dbMask ^ qMask);
                                    }
                                    totalDist += tierDist;
                                    if (totalDist > currentLimit) break;
                                }
                            } else if (qMode == 2) { // Float-Hybrid
                                float[] dbFloat = new float[dimension];
                                int dimOffset = 0;
                                for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                                    int width = tiers[tierIdx] - (tierIdx == 0 ? 0 : tiers[tierIdx - 1]);
                                    long baseOffset = rowIdx * (width * 4L);
                                    MemorySegment.copy(tierSegments[tierIdx], ValueLayout.JAVA_FLOAT, baseOffset, dbFloat, dimOffset, width);
                                    dimOffset += width;
                                }
                                totalDist = (int) (transformOperator.computeL2Float(zQuery, dbFloat) * 1000f);
                            } else { // 1-bit unrolled 8x
                                for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                                    int numLongs = tierLongs[tierIdx];
                                    int offset = tierOffsets[tierIdx];
                                    MemorySegment tierSeg = tierSegments[tierIdx];
                                    long baseOffset = rowIdx * (numLongs * 8L);
                                    int tierDist = 0;
                                    int l = 0;
                                    for (; l + 7 < numLongs; l += 8) {
                                        long w0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                        long w1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 1) * 8L));
                                        long w2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 2) * 8L));
                                        long w3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 3) * 8L));
                                        long w4 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 4) * 8L));
                                        long w5 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 5) * 8L));
                                        long w6 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 6) * 8L));
                                        long w7 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 7) * 8L));

                                        tierDist += Long.bitCount(bQuery[offset + l] ^ w0)
                                                  + Long.bitCount(bQuery[offset + l + 1] ^ w1)
                                                  + Long.bitCount(bQuery[offset + l + 2] ^ w2)
                                                  + Long.bitCount(bQuery[offset + l + 3] ^ w3)
                                                  + Long.bitCount(bQuery[offset + l + 4] ^ w4)
                                                  + Long.bitCount(bQuery[offset + l + 5] ^ w5)
                                                  + Long.bitCount(bQuery[offset + l + 6] ^ w6)
                                                  + Long.bitCount(bQuery[offset + l + 7] ^ w7);
                                    }
                                    for (; l < numLongs; l++) {
                                        long dbWord = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                        tierDist += Long.bitCount(bQuery[offset + l] ^ dbWord);
                                    }
                                    totalDist += tierDist;
                                    if (totalDist > currentLimit) break;
                                }
                            }

                            if (totalDist < currentLimit) {
                                int pos = kCandidate - 1;
                                while (pos > 0 && totalDist < topDists[pos - 1]) {
                                    topDists[pos] = topDists[pos - 1];
                                    topRowIds[pos] = topRowIds[pos - 1];
                                    pos--;
                                }
                                topDists[pos] = totalDist;
                                topRowIds[pos] = rowIdx;
                            }
                        }
                    }
                    if (gatheredCount >= kCandidate * 2) break;
                }
            }

            // Pass 3: Hamming-2 neighbor buckets on primary chunk 0 if still undersaturated
            if (gatheredCount < kCandidate) {
                int qKey0 = queryChunkKeys[0];
                long chunkOffBase = 0L;
                long chunkPostBase = 0L;
                for (int b1 = 0; b1 < 8; b1++) {
                    for (int b2 = b1 + 1; b2 < 8; b2++) {
                        int bKey = qKey0 ^ (1 << b1) ^ (1 << b2);
                        int start = prefixOffsetsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkOffBase + (bKey * 4L));
                        int end = prefixOffsetsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkOffBase + ((bKey + 1) * 4L));
                        for (int p = start; p < end; p++) {
                            int rowIdxInt = prefixPostingsSegment.get(ValueLayout.JAVA_INT_UNALIGNED, chunkPostBase + (p * 4L));
                            if (rowIdxInt < 0 || rowIdxInt >= size) continue;
                            int wordIdx = rowIdxInt >>> 6;
                            long mask = 1L << (rowIdxInt & 63);
                            if ((visited[wordIdx] & mask) != 0) continue;
                            visited[wordIdx] |= mask;
                            gatheredCount++;
                            long rowIdx = rowIdxInt & 0xFFFFFFFFL;

                            if (metadataSegment != null && (metadataSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8L) & 1L) == 1L) {
                                continue;
                            }

                            int currentLimit = topDists[kCandidate - 1];
                            int totalDist = 0;

                            if (qMode == 1) { // 2-bit
                                for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                                    int numLongs = tierLongs[tierIdx];
                                    int offset = tierOffsets[tierIdx];
                                    MemorySegment tierSeg = tierSegments[tierIdx];
                                    long baseOffset = rowIdx * (numLongs * 16L);
                                    int tierDist = 0;
                                    int l = 0;
                                    for (; l + 3 < numLongs; l += 4) {
                                        long s0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                        long m0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + (l * 8L));
                                        long s1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 1) * 8L));
                                        long m1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 1) * 8L));
                                        long s2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 2) * 8L));
                                        long m2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 2) * 8L));
                                        long s3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 3) * 8L));
                                        long m3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + ((l + 3) * 8L));

                                        tierDist += 4 * Long.bitCount(m0 & bQueryMask[offset + l] & (s0 ^ bQuery[offset + l])) + Long.bitCount(m0 ^ bQueryMask[offset + l])
                                                  + 4 * Long.bitCount(m1 & bQueryMask[offset + l + 1] & (s1 ^ bQuery[offset + l + 1])) + Long.bitCount(m1 ^ bQueryMask[offset + l + 1])
                                                  + 4 * Long.bitCount(m2 & bQueryMask[offset + l + 2] & (s2 ^ bQuery[offset + l + 2])) + Long.bitCount(m2 ^ bQueryMask[offset + l + 2])
                                                  + 4 * Long.bitCount(m3 & bQueryMask[offset + l + 3] & (s3 ^ bQuery[offset + l + 3])) + Long.bitCount(m3 ^ bQueryMask[offset + l + 3]);
                                    }
                                    for (; l < numLongs; l++) {
                                        long dbSign = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                        long dbMask = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L) + (l * 8L));
                                        long qSign = bQuery[offset + l];
                                        long qMask = bQueryMask[offset + l];
                                        tierDist += 4 * Long.bitCount(dbMask & qMask & (dbSign ^ qSign)) + Long.bitCount(dbMask ^ qMask);
                                    }
                                    totalDist += tierDist;
                                    if (totalDist > currentLimit) break;
                                }
                            } else if (qMode == 2) { // Float-Hybrid
                                float[] dbFloat = new float[dimension];
                                int dimOffset = 0;
                                for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                                    int width = tiers[tierIdx] - (tierIdx == 0 ? 0 : tiers[tierIdx - 1]);
                                    long baseOffset = rowIdx * (width * 4L);
                                    MemorySegment.copy(tierSegments[tierIdx], ValueLayout.JAVA_FLOAT, baseOffset, dbFloat, dimOffset, width);
                                    dimOffset += width;
                                }
                                totalDist = (int) (transformOperator.computeL2Float(zQuery, dbFloat) * 1000f);
                            } else { // 1-bit unrolled 8x
                                for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                                    int numLongs = tierLongs[tierIdx];
                                    int offset = tierOffsets[tierIdx];
                                    MemorySegment tierSeg = tierSegments[tierIdx];
                                    long baseOffset = rowIdx * (numLongs * 8L);
                                    int tierDist = 0;
                                    int l = 0;
                                    for (; l + 7 < numLongs; l += 8) {
                                        long w0 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                        long w1 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 1) * 8L));
                                        long w2 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 2) * 8L));
                                        long w3 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 3) * 8L));
                                        long w4 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 4) * 8L));
                                        long w5 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 5) * 8L));
                                        long w6 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 6) * 8L));
                                        long w7 = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + ((l + 7) * 8L));

                                        tierDist += Long.bitCount(bQuery[offset + l] ^ w0)
                                                  + Long.bitCount(bQuery[offset + l + 1] ^ w1)
                                                  + Long.bitCount(bQuery[offset + l + 2] ^ w2)
                                                  + Long.bitCount(bQuery[offset + l + 3] ^ w3)
                                                  + Long.bitCount(bQuery[offset + l + 4] ^ w4)
                                                  + Long.bitCount(bQuery[offset + l + 5] ^ w5)
                                                  + Long.bitCount(bQuery[offset + l + 6] ^ w6)
                                                  + Long.bitCount(bQuery[offset + l + 7] ^ w7);
                                    }
                                    for (; l < numLongs; l++) {
                                        long dbWord = tierSeg.get(ValueLayout.JAVA_LONG, baseOffset + (l * 8L));
                                        tierDist += Long.bitCount(bQuery[offset + l] ^ dbWord);
                                    }
                                    totalDist += tierDist;
                                    if (totalDist > currentLimit) break;
                                }
                            }

                            if (totalDist < currentLimit) {
                                int pos = kCandidate - 1;
                                while (pos > 0 && totalDist < topDists[pos - 1]) {
                                    topDists[pos] = topDists[pos - 1];
                                    topRowIds[pos] = topRowIds[pos - 1];
                                    pos--;
                                }
                                topDists[pos] = totalDist;
                                topRowIds[pos] = rowIdx;
                            }
                        }
                    }
                    if (gatheredCount >= kCandidate * 2) break;
                }
            }

            List<Long> candidates = new ArrayList<>();
            for (int i = 0; i < kCandidate; i++) {
                if (topDists[i] != Integer.MAX_VALUE) {
                    candidates.add(topRowIds[i]);
                }
            }

            if (candidates.isEmpty()) {
                int limit = (int) Math.min(size, kCandidate);
                for (long rowIdx = 0; rowIdx < limit; rowIdx++) {
                    candidates.add(rowIdx);
                }
            }

            if (qMode == 2) {
                List<SearchResult> queryResults = new ArrayList<>();
                int limit = Math.min(k, candidates.size());
                for (int i = 0; i < limit; i++) {
                    long rowIdx = candidates.get(i);
                    long recordId = idsSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8L);
                    queryResults.add(new SearchResult(recordId, topDists[i]));
                }
                finalResults[q] = queryResults;
                return;
            }

            // Task 4: Proactive Async DMA Prefetch for Gate 3 Precision Sidecars
            if (fp8Segment != null) {
                for (long rId : candidates) {
                    try {
                        fp8Segment.asSlice(rId * (long) dimension, (long) dimension).load();
                    } catch (Exception ignored) {}
                }
            } else if (fp4Segment != null) {
                int bytesPerRec = ((dimension + 15) / 16) * 9;
                for (long rId : candidates) {
                    try {
                        fp4Segment.asSlice(rId * (long) bytesPerRec, (long) bytesPerRec).load();
                    } catch (Exception ignored) {}
                }
            } else if (fp16Segment != null) {
                for (long rId : candidates) {
                    try {
                        fp16Segment.asSlice(rId * (long) dimension * 2L, (long) dimension * 2L).load();
                    } catch (Exception ignored) {}
                }
            }

            // Gate 3: Precision Sidecar Reranking with Early Distance Cutoff
            double[] bestDists = new double[k];
            long[] bestRowIds = new long[k];
            Arrays.fill(bestDists, Double.MAX_VALUE);

            if (fp8Segment != null) {
                float[] queryLut = new float[dimension * 256];
                for (int d = 0; d < dimension; d++) {
                    float qVal = query[d];
                    int dBase = d << 8;
                    for (int b = 0; b < 256; b++) {
                        float diff = qVal - FP8_E4M3_LUT[b];
                        queryLut[dBase | b] = diff * diff;
                    }
                }
                for (long rowIdx : candidates) {
                    double currentLimit = bestDists[k - 1];
                    double dist = computeExactL2FP8_LUT(queryLut, rowIdx, currentLimit);
                    if (dist < currentLimit) {
                        int pos = k - 1;
                        while (pos > 0 && dist < bestDists[pos - 1]) {
                            bestDists[pos] = bestDists[pos - 1];
                            bestRowIds[pos] = bestRowIds[pos - 1];
                            pos--;
                        }
                        bestDists[pos] = dist;
                        bestRowIds[pos] = rowIdx;
                    }
                }
            } else if (fp4Segment != null) {
                int numBlocks = (dimension + 15) / 16;
                int bytesPerRecord = numBlocks * 9;
                byte[] localFp4 = new byte[bytesPerRecord];
                for (long rowIdx : candidates) {
                    double currentLimit = bestDists[k - 1];
                    double dist = computeExactL2FP4(query, rowIdx, localFp4, numBlocks, currentLimit);
                    if (dist < currentLimit) {
                        int pos = k - 1;
                        while (pos > 0 && dist < bestDists[pos - 1]) {
                            bestDists[pos] = bestDists[pos - 1];
                            bestRowIds[pos] = bestRowIds[pos - 1];
                            pos--;
                        }
                        bestDists[pos] = dist;
                        bestRowIds[pos] = rowIdx;
                    }
                }
            } else if (fp16Segment != null) {
                short[] localFp16 = new short[dimension];
                for (long rowIdx : candidates) {
                    double currentLimit = bestDists[k - 1];
                    double dist = computeExactL2FP16(query, rowIdx, localFp16, currentLimit);
                    if (dist < currentLimit) {
                        int pos = k - 1;
                        while (pos > 0 && dist < bestDists[pos - 1]) {
                            bestDists[pos] = bestDists[pos - 1];
                            bestRowIds[pos] = bestRowIds[pos - 1];
                            pos--;
                        }
                        bestDists[pos] = dist;
                        bestRowIds[pos] = rowIdx;
                    }
                }
            } else {
                double queryL2Norm = 0.0;
                double querySum = 0.0;
                for (float val : zQuery) {
                    queryL2Norm += val * val;
                    querySum += val;
                }
                for (long rowIdx : candidates) {
                    double currentLimit = bestDists[k - 1];
                    double dist = computeAsymmetricL2DistanceOffHeap(zQuery, queryL2Norm, querySum, rowIdx);
                    if (dist < currentLimit) {
                        int pos = k - 1;
                        while (pos > 0 && dist < bestDists[pos - 1]) {
                            bestDists[pos] = bestDists[pos - 1];
                            bestRowIds[pos] = bestRowIds[pos - 1];
                            pos--;
                        }
                        bestDists[pos] = dist;
                        bestRowIds[pos] = rowIdx;
                    }
                }
            }

            List<SearchResult> queryResults = new ArrayList<>(k);
            for (int i = 0; i < k; i++) {
                if (bestDists[i] != Double.MAX_VALUE) {
                    long recordId = idsSegment.get(ValueLayout.JAVA_LONG, bestRowIds[i] * 8L);
                    queryResults.add(new SearchResult(recordId, (int) (bestDists[i] * 1000000.0)));
                }
            }
            finalResults[q] = queryResults;
        });

        return finalResults;
    }

    private double computeExactL2FP8_LUT(float[] queryLut, long rowIdx, double currentLimit) {
        long rowOffset = rowIdx * (long) dimension;
        double sum = 0.0;
        int d = 0;
        for (; d + 7 < dimension; d += 8) {
            int b0 = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d) & 0xFF;
            int b1 = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d + 1) & 0xFF;
            int b2 = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d + 2) & 0xFF;
            int b3 = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d + 3) & 0xFF;
            int b4 = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d + 4) & 0xFF;
            int b5 = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d + 5) & 0xFF;
            int b6 = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d + 6) & 0xFF;
            int b7 = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d + 7) & 0xFF;
            sum += queryLut[(d << 8) | b0] + queryLut[((d + 1) << 8) | b1]
                 + queryLut[((d + 2) << 8) | b2] + queryLut[((d + 3) << 8) | b3]
                 + queryLut[((d + 4) << 8) | b4] + queryLut[((d + 5) << 8) | b5]
                 + queryLut[((d + 6) << 8) | b6] + queryLut[((d + 7) << 8) | b7];
            if (sum > currentLimit) {
                return sum;
            }
        }
        for (; d < dimension; d++) {
            int b = fp8Segment.get(ValueLayout.JAVA_BYTE, rowOffset + d) & 0xFF;
            sum += queryLut[(d << 8) | b];
        }
        return sum;
    }

    private double computeExactL2FP4(float[] rawQuery, long rowIdx, byte[] localFp4, int numBlocks, double currentLimit) {
        long rowOffset = rowIdx * (numBlocks * 9L);
        MemorySegment.copy(fp4Segment, ValueLayout.JAVA_BYTE, rowOffset, localFp4, 0, numBlocks * 9);
        double sum = 0.0;
        for (int b = 0; b < numBlocks; b++) {
            int blockOffset = b * 9;
            float scale = FP8_E4M3_LUT[localFp4[blockOffset] & 0xFF];
            int blockStart = b * 16;
            for (int j = 0; j < 8; j++) {
                byte packed = localFp4[blockOffset + 1 + j];
                int n0 = packed & 0xF;
                int n1 = (packed >>> 4) & 0xF;
                int d0 = blockStart + j * 2;
                int d1 = blockStart + j * 2 + 1;
                if (d0 < dimension) {
                    float diff = rawQuery[d0] - (FP4_E2M1_LUT[n0] * scale);
                    sum += diff * diff;
                }
                if (d1 < dimension) {
                    float diff = rawQuery[d1] - (FP4_E2M1_LUT[n1] * scale);
                    sum += diff * diff;
                }
            }
            if (sum > currentLimit) {
                return sum;
            }
        }
        return sum;
    }

    private double computeExactL2FP16(float[] rawQuery, long rowIdx, short[] localFp16, double currentLimit) {
        long rowOffset = rowIdx * dimension * 2L;
        MemorySegment.copy(fp16Segment, ValueLayout.JAVA_SHORT, rowOffset, localFp16, 0, dimension);
        double sum = 0.0;
        for (int d = 0; d < dimension; d++) {
            float dbVal = Float.float16ToFloat(localFp16[d]);
            double diff = rawQuery[d] - dbVal;
            sum += diff * diff;
            if ((d & 7) == 7 && sum > currentLimit) {
                return sum;
            }
        }
        return sum;
    }

    private double computeAsymmetricL2DistanceOffHeap(float[] query, double queryL2Norm, double querySum, long rowIdx) {
        int totalMaskPopcount = 0;
        int queryOffsetLongs = 0;

        if (qMode == 1) { // 2-bit mode
            double sumPositive = 0.0;
            double sumActive = 0.0;
            for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                long baseOffset = rowIdx * (numLongs * 16L);

                for (int l = 0; l < numLongs; l++) {
                    long mask = tierSegments[tierIdx].get(ValueLayout.JAVA_LONG,
                            baseOffset + (numLongs * 8L) + (l * 8));
                    if (mask == 0L)
                        continue;

                    totalMaskPopcount += Long.bitCount(mask);
                    long word = tierSegments[tierIdx].get(ValueLayout.JAVA_LONG, baseOffset + (l * 8));

                    int bitOffset = (queryOffsetLongs + l) * 64;
                    int limit = Math.min(64, query.length - bitOffset);
                    long limitMask = limit == 64 ? -1L : (1L << limit) - 1L;
                    long active = mask & limitMask;
                    while (active != 0) {
                        int bitIdx = Long.numberOfTrailingZeros(active);
                        float qVal = query[bitOffset + bitIdx];
                        sumActive += qVal;
                        if (((word >>> bitIdx) & 1L) != 0L) {
                            sumPositive += qVal;
                        }
                        active &= active - 1;
                    }
                }
                queryOffsetLongs += numLongs;
            }
            return totalMaskPopcount + queryL2Norm - 4.0 * sumPositive + 2.0 * sumActive;
        } else { // 1-bit mode
            double sumPositive = 0.0;
            queryOffsetLongs = 0;
            for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                long baseOffset = rowIdx * (numLongs * 8L);

                for (int l = 0; l < numLongs; l++) {
                    long word = tierSegments[tierIdx].get(ValueLayout.JAVA_LONG, baseOffset + (l * 8));
                    int bitOffset = (queryOffsetLongs + l) * 64;
                    int limit = Math.min(64, query.length - bitOffset);
                    long limitMask = limit == 64 ? -1L : (1L << limit) - 1L;
                    long active = word & limitMask;
                    while (active != 0) {
                        int bitIdx = Long.numberOfTrailingZeros(active);
                        sumPositive += query[bitOffset + bitIdx];
                        active &= active - 1;
                    }
                }
                queryOffsetLongs += numLongs;
            }
            return query.length + queryL2Norm + 2.0 * querySum - 4.0 * sumPositive;
        }
    }

    @Override
    public long queryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        if (queries == null || queries.length == 0)
            return 0;
        int numQueries = queries.length;
        if (size == 0)
            return 0;

        long currentChunkSize = this.chunkSize;
        long numChunks = (size + currentChunkSize - 1) / currentChunkSize;

        Arena arena = Arena.global();
        MemorySegment[] threadLocalMasks = new MemorySegment[numWorkers];
        for (int w = 0; w < numWorkers; w++) {
            threadLocalMasks[w] = arena.allocate(size);
        }

        IntStream.range(0, (int) numChunks).parallel().forEach(c -> {
            long startIdx = c * currentChunkSize;
            long endIdx = Math.min(startIdx + currentChunkSize, size);
            int workerId = (int) (c % numWorkers);
            executeVotingRange(startIdx, endIdx, queries, families, thresholds, threadLocalMasks[workerId]);
        });

        int numThreads = Runtime.getRuntime().availableProcessors();
        long recordsPerThread = size / numThreads;
        if (recordsPerThread == 0) {
            numThreads = 1;
            recordsPerThread = size;
        }
        final int activeThreads = numThreads;
        final long finalRecordsPerThread = recordsPerThread;

        return IntStream.range(0, activeThreads).parallel().mapToLong(t -> {
            long startIdx = t * finalRecordsPerThread;
            long endIdx = (t == activeThreads - 1) ? size : (t + 1) * finalRecordsPerThread;
            long resonantCount = 0;
            for (long i = startIdx; i < endIdx; i++) {
                byte mergedVal = 0;
                for (int w = 0; w < numWorkers; w++) {
                    mergedVal |= threadLocalMasks[w].get(ValueLayout.JAVA_BYTE, i);
                }
                votingMask.set(ValueLayout.JAVA_BYTE, i, mergedVal);
                if (Integer.bitCount(mergedVal & 0xFF) >= 5) {
                    resonantCount++;
                }
            }
            return resonantCount;
        }).sum();
    }

    private void executeVotingRange(long startIdx, long endIdx, float[][] queries, int[] families, int[] thresholds,
            MemorySegment localMask) {
        int numQueries = queries.length;

        long[][] bQueries = new long[numQueries][];
        long[][] bQueriesMask = new long[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            if (qMode == 1) {
                float[] z = transformOperator.preconditionAndRotate(queries[q]);
                float qThreshold = TransformOperator.calculatePercentileThreshold(z, 0.20f);
                long[][] packed = transformOperator.quantize2Bit(z, qThreshold);
                bQueries[q] = packed[0];
                bQueriesMask[q] = packed[1];
            } else {
                bQueries[q] = transformOperator.transformAndQuantize(queries[q]);
            }
        }

        int T = 0;
        for (int i = 0; i < numTiers; i++) {
            if (cumulativeEnergy[i] >= targetEnergyBudget) {
                T = i;
                break;
            }
        }

        int totalLongs = dimension / 64;
        long[] dbWords = new long[totalLongs];
        long[] dbMasks = new long[totalLongs];

        MemorySegment metaSeg = this.metadataSegment;
        MemorySegment[] localTiers = this.tierSegments;
        MemorySegment tier0 = localTiers[0];

        for (long i = startIdx; i < endIdx; i++) {
            long metaVal = (metaSeg != null) ? metaSeg.get(ValueLayout.JAVA_LONG, i * 8) : 2L;
            if ((metaVal & 1L) == 1L)
                continue;

            // Loading Tier 0
            int numLongs0 = tierLongs[0];
            if (qMode == 1) { // 2-bit mode
                long baseOffset0 = i * (numLongs0 * 16L);
                long t0Sign = tier0.get(ValueLayout.JAVA_LONG, baseOffset0);
                long t0Mask = tier0.get(ValueLayout.JAVA_LONG, baseOffset0 + (numLongs0 * 8L));

                dbWords[0] = t0Sign;
                dbMasks[0] = t0Mask;
                if (numLongs0 > 1) {
                    MemorySegment.copy(tier0, ValueLayout.JAVA_LONG, baseOffset0 + 8, dbWords, 1, numLongs0 - 1);
                    MemorySegment.copy(tier0, ValueLayout.JAVA_LONG, baseOffset0 + (numLongs0 * 8L) + 8, dbMasks, 1, numLongs0 - 1);
                }
            } else { // 1-bit mode
                long baseOffset0 = i * (numLongs0 * 8L);
                long t0Val = tier0.get(ValueLayout.JAVA_LONG, baseOffset0);

                dbWords[0] = t0Val;
                if (numLongs0 > 1) {
                    MemorySegment.copy(tier0, ValueLayout.JAVA_LONG, baseOffset0 + 8, dbWords, 1, numLongs0 - 1);
                }
            }

            // Load remaining active tiers up to T
            if (qMode == 1) { // 2-bit mode
                for (int tierIdx = 1; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int offset = tierOffsets[tierIdx];
                    MemorySegment tierSeg = localTiers[tierIdx];
                    long baseOffset = i * (numLongs * 16L);
                    MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset, dbWords, offset, numLongs);
                    MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L), dbMasks, offset, numLongs);
                }
            } else { // 1-bit mode
                for (int tierIdx = 1; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int offset = tierOffsets[tierIdx];
                    MemorySegment tierSeg = localTiers[tierIdx];
                    long baseOffset = i * (numLongs * 8L);
                    MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset, dbWords, offset, numLongs);
                }
            }

            byte maskVal = 0;
            for (int q = 0; q < numQueries; q++) {
                int totalDist = 0;
                boolean earlyExit = false;

                for (int tierIdx = 0; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int offset = tierOffsets[tierIdx];
                    int tierDist = 0;
                    if (qMode == 1) { // 2-bit mode
                        for (int l = 0; l < numLongs; l++) {
                            long qSign = bQueries[q][offset + l];
                            long qMask = bQueriesMask[q][offset + l];
                            long dbSign = dbWords[offset + l];
                            long dbMask = dbMasks[offset + l];

                            long mask4 = dbMask & qMask & (dbSign ^ qSign);
                            long mask1 = dbMask ^ qMask;
                            tierDist += 4 * Long.bitCount(mask4) + Long.bitCount(mask1);
                        }
                    } else { // 1-bit mode
                        for (int l = 0; l < numLongs; l++) {
                            tierDist += Long.bitCount(bQueries[q][offset + l] ^ dbWords[offset + l]);
                        }
                    }
                    totalDist += tierDist;

                    if (totalDist > thresholds[q]) {
                        earlyExit = true;
                        break;
                    }
                }

                if (!earlyExit) {
                    maskVal |= (byte) (1 << families[q]);
                }
            }
            localMask.set(ValueLayout.JAVA_BYTE, i, maskVal);
        }
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public byte getPlanetId() {
        return planetId;
    }

    @Override
    public long getPlanetRadius() {
        return planetRadius;
    }

    @Override
    public int getTierCount() {
        return numTiers;
    }

    @Override
    public void close() {
        // FlatIndex off-heap memory is managed via Arena or direct mmap
    }

    // =========================================================================
    // CUDA Acceleration Implementation & CPU Fallback
    // =========================================================================

    private static final int GPU_BATCH_THRESHOLD = 100;
    private static final int MIN_DIMENSION_FOR_GPU = 64;

    private long[] deviceTierBuffers;
    private boolean cudaInitialized = false;

    private void ensureCudaInitialized() {
        if (cudaInitialized || CudaDeviceManager.isAvailable() == 0) {
            return;
        }
        deviceTierBuffers = new long[tierVectors.length];
        for (int i = 0; i < tierVectors.length; i++) {
            ByteBuffer tierBuffer = tierVectors[i];
            deviceTierBuffers[i] = CudaMemoryManager.allocDevice(tierBuffer.capacity());
            long hostPtr = CudaMemoryManager.getDirectBufferAddress(tierBuffer);
            CudaMemoryManager.copyToDevice(deviceTierBuffers[i], hostPtr, tierBuffer.capacity());
        }
        cudaInitialized = true;
    }

    @Override
    public List<SearchResult>[] cudaBatchSearch(float[][] queries, int k) {
        if (queries.length < GPU_BATCH_THRESHOLD || dimension < MIN_DIMENSION_FOR_GPU) {
            return batchSearch(queries, k);
        }

        ensureCudaInitialized();

        int numQueries = queries.length;

        long hostQueries = CudaMemoryManager.allocPinned(numQueries * dimension * 4L);
        long deviceQueries = CudaMemoryManager.allocDevice(numQueries * dimension * 4L);
        long hostDistances = CudaMemoryManager.allocPinned(numQueries * size * 4L);

        ByteBuffer queryBuffer = ByteBuffer.allocateDirect(numQueries * dimension * 4);
        for (float[] query : queries) {
            for (float val : query) {
                queryBuffer.putFloat(val);
            }
        }
        queryBuffer.rewind();

        long queryBufferPtr = CudaMemoryManager.getDirectBufferAddress(queryBuffer);
        CudaMemoryManager.copyToDevice(deviceQueries, queryBufferPtr, numQueries * dimension * 4L);

        int status = pithos_cuda_launch_batch_hamming(
            deviceTierBuffers, deviceQueries, hostDistances,
            Math.toIntExact(size), numQueries, tierVectors.length, tierOffsets, tierSizes
        );

        if (status != 0) {
            CudaMemoryManager.freePinned(hostQueries);
            CudaMemoryManager.freeDevice(deviceQueries);
            CudaMemoryManager.freePinned(hostDistances);
            return batchSearch(queries, k);
        }

        ByteBuffer distanceBuffer = ByteBuffer.allocateDirect(numQueries * Math.toIntExact(size) * 4);
        long distanceBufferPtr = CudaMemoryManager.getDirectBufferAddress(distanceBuffer);
        CudaMemoryManager.copyFromDevice(distanceBufferPtr, hostDistances, numQueries * Math.toIntExact(size) * 4L);

        List<SearchResult>[] results = new List[numQueries];
        for (int q = 0; q < numQueries; q++) {
            List<SearchResult> queryResults = new ArrayList<>(k);
            for (int i = 0; i < size && i < k; i++) {
                int distance = distanceBuffer.getInt(q * Math.toIntExact(size) + i);
                queryResults.add(new SearchResult(i, distance));
            }
            results[q] = queryResults;
        }

        CudaMemoryManager.freePinned(hostQueries);
        CudaMemoryManager.freeDevice(deviceQueries);
        CudaMemoryManager.freePinned(hostDistances);

        return results;
    }

    @Override
    public long cudaQueryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        if (queries.length < GPU_BATCH_THRESHOLD || dimension < MIN_DIMENSION_FOR_GPU) {
            return queryPlanetaryGrid(queries, families, thresholds, votingMask);
        }

        ensureCudaInitialized();

        int numQueries = queries.length;
        int numWordsPerVector = (dimension + 63) / 64;
        int numFamilies = families.length;

        long hostQueries = CudaMemoryManager.allocPinned(numQueries * dimension * 4L);
        long deviceQueries = CudaMemoryManager.allocDevice(numQueries * dimension * 4L);
        long hostFamilies = CudaMemoryManager.allocPinned(numQueries * 4L);
        long deviceFamilies = CudaMemoryManager.allocDevice(numQueries * 4L);
        long hostThresholds = CudaMemoryManager.allocPinned(numQueries * 4L);
        long deviceThresholds = CudaMemoryManager.allocDevice(numQueries * 4L);
        long hostVotingMask = CudaMemoryManager.allocPinned(size);
        long deviceVotingMask = CudaMemoryManager.allocDevice(size);

        ByteBuffer queryBuffer = ByteBuffer.allocateDirect(numQueries * dimension * 4);
        for (float[] query : queries) {
            for (float val : query) {
                queryBuffer.putFloat(val);
            }
        }
        queryBuffer.rewind();

        ByteBuffer familiesBuffer = ByteBuffer.allocateDirect(numQueries * 4);
        for (int family : families) {
            familiesBuffer.putInt(family);
        }
        familiesBuffer.rewind();

        ByteBuffer thresholdsBuffer = ByteBuffer.allocateDirect(numQueries * 4);
        for (int threshold : thresholds) {
            thresholdsBuffer.putInt(threshold);
        }
        thresholdsBuffer.rewind();

        long queryBufferPtr = CudaMemoryManager.getDirectBufferAddress(queryBuffer);
        long familiesBufferPtr = CudaMemoryManager.getDirectBufferAddress(familiesBuffer);
        long thresholdsBufferPtr = CudaMemoryManager.getDirectBufferAddress(thresholdsBuffer);

        CudaMemoryManager.copyToDevice(deviceQueries, queryBufferPtr, numQueries * dimension * 4L);
        CudaMemoryManager.copyToDevice(deviceFamilies, familiesBufferPtr, numQueries * 4L);
        CudaMemoryManager.copyToDevice(deviceThresholds, thresholdsBufferPtr, numQueries * 4L);

        int status = pithos_cuda_launch_voting(
            deviceTierBuffers, deviceQueries, deviceFamilies, deviceThresholds,
            deviceVotingMask, Math.toIntExact(size), numQueries, numFamilies, numWordsPerVector
        );

        if (status != 0) {
            CudaMemoryManager.freePinned(hostQueries);
            CudaMemoryManager.freeDevice(deviceQueries);
            CudaMemoryManager.freePinned(hostFamilies);
            CudaMemoryManager.freeDevice(deviceFamilies);
            CudaMemoryManager.freePinned(hostThresholds);
            CudaMemoryManager.freeDevice(deviceThresholds);
            CudaMemoryManager.freePinned(hostVotingMask);
            CudaMemoryManager.freeDevice(deviceVotingMask);
            return queryPlanetaryGrid(queries, families, thresholds, votingMask);
        }

        ByteBuffer votingBuffer = ByteBuffer.allocateDirect(Math.toIntExact(size));
        long votingBufferPtr = CudaMemoryManager.getDirectBufferAddress(votingBuffer);
        CudaMemoryManager.copyFromDevice(votingBufferPtr, deviceVotingMask, size);

        long count = 0;
        for (int i = 0; i < size; i++) {
            if (votingBuffer.get(i) != 0) {
                count++;
                votingMask.set(ValueLayout.JAVA_BYTE, i, votingBuffer.get(i));
            }
        }

        CudaMemoryManager.freePinned(hostQueries);
        CudaMemoryManager.freeDevice(deviceQueries);
        CudaMemoryManager.freePinned(hostFamilies);
        CudaMemoryManager.freeDevice(deviceFamilies);
        CudaMemoryManager.freePinned(hostThresholds);
        CudaMemoryManager.freeDevice(deviceThresholds);
        CudaMemoryManager.freePinned(hostVotingMask);
        CudaMemoryManager.freeDevice(deviceVotingMask);

        return count;
    }

    private static int pithos_cuda_launch_batch_hamming(
        long[] deviceTierBuffers, long deviceQueries, long hostDistances,
        int numDbVectors, int numQueries, int numTiers, int[] tierOffsets, int[] tierSizes
    ) {
        return CudaNativeBindings.pithos_cuda_launch_batch_hamming(
            deviceTierBuffers, deviceQueries, hostDistances,
            numDbVectors, numQueries, numTiers, tierOffsets, tierSizes
        );
    }

    private static int pithos_cuda_launch_voting(
        long[] deviceTierBuffers, long deviceQueries, long deviceFamilies, long deviceThresholds,
        long deviceVotingMask, int numDbVectors, int numQueries, int numFamilies, int numWordsPerVector
    ) {
        return CudaNativeBindings.pithos_cuda_launch_voting(
            deviceTierBuffers, deviceQueries, deviceFamilies, deviceThresholds,
            deviceVotingMask, numDbVectors, numQueries, numFamilies, numWordsPerVector
        );
    }
}