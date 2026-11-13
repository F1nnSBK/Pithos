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



    private static final ThreadLocal<long[]> VISITED_SCRATCH = ThreadLocal.withInitial(() -> new long[1024]);
    private static final ThreadLocal<float[]> QUERY_LUT_SCRATCH = new ThreadLocal<>();
    private static final ThreadLocal<float[]> DB_FLOAT_SCRATCH = new ThreadLocal<>();
    private final boolean sidecarInWhtSpace;
    private final boolean allTiers64Aligned;
    private final int totalTierLongs;
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

    /// Returns true if FP8/FP4 sidecars are stored in orthogonally rotated WHT space.
    public boolean isSidecarInWhtSpace() {
        return sidecarInWhtSpace;
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
        this.sidecarInWhtSpace = !this.isSingleFileContainer
                || (userMetadataJson != null && userMetadataJson.contains("\"space\": \"wht\""));
        this.prefixOffsetsSegment = prefixOffsetsSegment;
        this.prefixPostingsSegment = prefixPostingsSegment;
        this.hasPrefixTable = (prefixOffsetsSegment != null && prefixPostingsSegment != null);

        this.tierLongs = new int[numTiers];
        boolean aligned64 = true;
        int prevBoundVal = 0;
        for (int idx = 0; idx < numTiers; idx++) {
            int width = tiers[idx] - prevBoundVal;
            if ((width & 63) != 0) {
                aligned64 = false;
            }
            this.tierLongs[idx] = (width + 63) / 64;
            prevBoundVal = tiers[idx];
        }
        this.allTiers64Aligned = aligned64;

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
        this.totalTierLongs = offset;

        this.transformOperator = new TransformOperator(dimension, tiers);
        this.numWorkers = Runtime.getRuntime().availableProcessors();
    }

    private long[] packQueryByTiers(long[] rawPacked) {
        if (allTiers64Aligned) {
            return rawPacked;
        }
        long[] perTier = new long[totalTierLongs];
        int startDim = 0;
        for (int t = 0; t < numTiers; t++) {
            int width = tierSizes[t];
            int count = tierLongs[t];
            int base = tierOffsets[t];
            for (int l = 0; l < count; l++) {
                perTier[base + l] = TransformOperator.extractTierWord(rawPacked, startDim, width, l);
            }
            startDim = tiers[t];
        }
        return perTier;
    }

    static long readTierWord(MemorySegment seg, long offset, int validBytes) {
        if (validBytes >= 8) {
            return seg.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
        } else if (validBytes == 4) {
            return seg.get(ValueLayout.JAVA_INT_UNALIGNED, offset) & 0xFFFFFFFFL;
        } else {
            long word = 0L;
            for (int b = 0; b < validBytes; b++) {
                word |= ((long) seg.get(ValueLayout.JAVA_BYTE, offset + b) & 0xFFL) << (b * 8);
            }
            return word;
        }
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

        boolean isPithosExt = mainPath.toString().endsWith(".pithos");
        boolean isPithosMagic = PithosContainer.isPithosContainer(mainPath);

        // 1. Check if this is a Single-File .pithos Container (DIOGENES magic or .pithos extension)
        if (isPithosMagic || isPithosExt) {
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

                long fileSize = channel.size();
                MemorySegment containerSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, Arena.global());

                long idsOffset = PithosContainer.align64(PithosContainer.SUPERBLOCK_SIZE);
                long idsLength = totalRecords * 8L;
                if (idsOffset + idsLength > fileSize || idsOffset < 0 || idsLength < 0) {
                    throw new IOException("Corrupt container: IDs segment exceeds file size (offset=" + idsOffset + ", len=" + idsLength + ", file=" + fileSize + ")");
                }
                MemorySegment idsSegment = containerSegment.asSlice(idsOffset, idsLength);

                MemorySegment[] tierSegments = new MemorySegment[numTiers];
                int prevBound = 0;
                long currentOffset = PithosContainer.align64(idsOffset + idsLength);
                for (int k = 0; k < numTiers; k++) {
                    int width = tiers[k] - prevBound;
                    long bytesPerRecord = switch (qMode) {
                        case 1 -> 2L * ((width + 7) / 8);
                        case 2 -> (width * 4L);
                        default -> (width + 7) / 8;
                    };
                    long tierLen = totalRecords * bytesPerRecord;
                    if (currentOffset + tierLen > fileSize || currentOffset < 0 || tierLen < 0) {
                        throw new IOException("Corrupt container: Tier " + k + " segment exceeds file size (offset=" + currentOffset + ", len=" + tierLen + ", file=" + fileSize + ")");
                    }
                    tierSegments[k] = containerSegment.asSlice(currentOffset, tierLen);
                    currentOffset = PithosContainer.align64(currentOffset + tierLen);
                    prevBound = tiers[k];
                }

                MemorySegment fp16Segment = null;
                MemorySegment fp8Segment = null;
                MemorySegment fp4Segment = null;
                if (sidecarMode == VectorDb.SIDECAR_FP16) {
                    long sidecarLen = totalRecords * dimension * 2L;
                    if (currentOffset + sidecarLen > fileSize || currentOffset < 0 || sidecarLen < 0) {
                        throw new IOException("Corrupt container: FP16 sidecar segment exceeds file size");
                    }
                    fp16Segment = containerSegment.asSlice(currentOffset, sidecarLen);
                    currentOffset = PithosContainer.align64(currentOffset + sidecarLen);
                } else if (sidecarMode == VectorDb.SIDECAR_FP8) {
                    long sidecarLen = totalRecords * dimension * 1L;
                    if (currentOffset + sidecarLen > fileSize || currentOffset < 0 || sidecarLen < 0) {
                        throw new IOException("Corrupt container: FP8 sidecar segment exceeds file size");
                    }
                    fp8Segment = containerSegment.asSlice(currentOffset, sidecarLen);
                    currentOffset = PithosContainer.align64(currentOffset + sidecarLen);
                } else if (sidecarMode == VectorDb.SIDECAR_FP4) {
                    int blockSize = 16;
                    int numBlocks = (dimension + blockSize - 1) / blockSize;
                    long sidecarLen = totalRecords * (numBlocks * 9L);
                    if (currentOffset + sidecarLen > fileSize || currentOffset < 0 || sidecarLen < 0) {
                        throw new IOException("Corrupt container: FP4 sidecar segment exceeds file size");
                    }
                    fp4Segment = containerSegment.asSlice(currentOffset, sidecarLen);
                    currentOffset = PithosContainer.align64(currentOffset + sidecarLen);
                }

                PithosContainer.Section prefixSec = PithosContainer.extractPrefixTableSection(tocJson);
                MemorySegment prefixOffsetsSegment = null;
                MemorySegment prefixPostingsSegment = null;
                if (prefixSec.offset() > 0 && prefixSec.length() >= PithosContainer.MIH_OFFSETS_BYTES) {
                    long offSegLen = PithosContainer.MIH_OFFSETS_BYTES;
                    long postSegLen = totalRecords * 4L * PithosContainer.NUM_MIH_CHUNKS;
                    if (prefixSec.offset() + offSegLen + postSegLen > fileSize || prefixSec.offset() < 0) {
                        throw new IOException("Corrupt container: Prefix table segment exceeds file size");
                    }
                    prefixOffsetsSegment = containerSegment.asSlice(prefixSec.offset(), offSegLen);
                    prefixPostingsSegment = containerSegment.asSlice(prefixSec.offset() + offSegLen, postSegLen);
                }

                PithosContainer.Section metaSec = PithosContainer.extractMetadataSection(tocJson);
                MemorySegment metadataPayloadSegment = null;
                if (metaSec.offset() > 0 && metaSec.length() > 0) {
                    if (metaSec.offset() + metaSec.length() > fileSize || metaSec.offset() < 0) {
                        throw new IOException("Corrupt container: Metadata segment exceeds file size");
                    }
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
            } catch (Exception e) {
                if (isPithosExt || isPithosMagic) {
                    throw new IOException("Failed to load Pithos container '" + mainPath + "': " + e.getMessage(), e);
                }
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
            throw new IllegalArgumentException("Invalid file format for '" + mainPath + "': file is neither a valid single-file .pithos container nor a legacy multi-file PLAN index");
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
                case 1 -> 2L * ((width + 7) / 8);
                case 2 -> (width * 4L);
                default -> (width + 7) / 8;
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

        if (hasPrefixTable && size > 2048) {
            return searchWithPrefixRouting(queries, k);
        }

        return searchLinearScanParallel(queries, k);
    }

    private void evaluateCandidateGate2(
            long rowIdx,
            int activeT,
            float[] zQuery,
            long[] bQuery,
            long[] bQueryMask,
            int[] topDists,
            long[] topRowIds,
            int kCandidate) {
        if (metadataSegment != null && (metadataSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8L) & 1L) == 1L) {
            return;
        }

        int currentLimit = topDists[kCandidate - 1];
        int totalDist = 0;

        if (qMode == 1) { // 2-bit QJL
            for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                int width = tierSizes[tierIdx];
                int offset = tierOffsets[tierIdx];
                MemorySegment tierSeg = tierSegments[tierIdx];
                int tierDist = 0;
                if ((width & 63) == 0) {
                    long baseOffset = rowIdx * (numLongs * 16L);
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
                } else {
                    int halfBytes = (width + 7) / 8;
                    long baseOffset = rowIdx * (2L * halfBytes);
                    for (int l = 0; l < numLongs; l++) {
                        int validBytes = Math.min(8, halfBytes - l * 8);
                        int remBits = Math.min(64, width - (l << 6));
                        long bitMask = (remBits == 64) ? -1L : ((1L << remBits) - 1L);
                        long dbSign = readTierWord(tierSeg, baseOffset + (l * 8L), validBytes) & bitMask;
                        long dbMask = readTierWord(tierSeg, baseOffset + halfBytes + (l * 8L), validBytes) & bitMask;
                        long qSign = bQuery[offset + l] & bitMask;
                        long qMask = bQueryMask[offset + l] & bitMask;
                        tierDist += 4 * Long.bitCount(dbMask & qMask & (dbSign ^ qSign)) + Long.bitCount(dbMask ^ qMask);
                    }
                }
                totalDist += tierDist;
                if (totalDist > currentLimit) break;
            }
        } else if (qMode == 2) { // Float-Hybrid
            float[] dbFloat = DB_FLOAT_SCRATCH.get();
            if (dbFloat == null || dbFloat.length < dimension) {
                dbFloat = new float[dimension];
                DB_FLOAT_SCRATCH.set(dbFloat);
            }
            int dimOffset = 0;
            for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                int width = tierSizes[tierIdx];
                long baseOffset = rowIdx * (width * 4L);
                MemorySegment.copy(tierSegments[tierIdx], ValueLayout.JAVA_FLOAT, baseOffset, dbFloat, dimOffset, width);
                dimOffset += width;
            }
            totalDist = (int) (transformOperator.computeL2Float(zQuery, dbFloat) * 1000f);
        } else { // 1-bit unrolled 8x
            for (int tierIdx = 0; tierIdx <= activeT; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                int width = tierSizes[tierIdx];
                int offset = tierOffsets[tierIdx];
                MemorySegment tierSeg = tierSegments[tierIdx];
                int tierDist = 0;
                if ((width & 63) == 0) {
                    long baseOffset = rowIdx * (numLongs * 8L);
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
                } else {
                    int tierBytes = (width + 7) / 8;
                    long baseOffset = rowIdx * tierBytes;
                    for (int l = 0; l < numLongs; l++) {
                        int validBytes = Math.min(8, tierBytes - l * 8);
                        int remBits = Math.min(64, width - (l << 6));
                        long bitMask = (remBits == 64) ? -1L : ((1L << remBits) - 1L);
                        long dbWord = readTierWord(tierSeg, baseOffset + (l * 8L), validBytes) & bitMask;
                        tierDist += Long.bitCount((bQuery[offset + l] ^ dbWord) & bitMask);
                    }
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

    private List<SearchResult> rerankGate3(
            float[] query,
            float[] zQuery,
            int[] topDists,
            long[] topRowIds,
            int validCandidates,
            int k) {
        if (qMode == 2) {
            int limit = Math.min(k, validCandidates);
            List<SearchResult> queryResults = new ArrayList<>(limit);
            for (int i = 0; i < limit; i++) {
                long rowIdx = topRowIds[i];
                long recordId = idsSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8L);
                queryResults.add(new SearchResult(recordId, topDists[i]));
            }
            return queryResults;
        }

        // Gate 3: Precision Sidecar Reranking with Early Distance Cutoff (zero syscalls, zero boxing)
        double[] bestDists = new double[k];
        long[] bestRowIds = new long[k];
        Arrays.fill(bestDists, Double.MAX_VALUE);

        float[] sidecarQuery = sidecarInWhtSpace ? zQuery : query;

        if (fp8Segment != null) {
            int lutSize = dimension * 256;
            float[] queryLut = QUERY_LUT_SCRATCH.get();
            if (queryLut == null || queryLut.length < lutSize) {
                queryLut = new float[lutSize];
                QUERY_LUT_SCRATCH.set(queryLut);
            }
            for (int d = 0; d < dimension; d++) {
                float qVal = sidecarQuery[d];
                int dBase = d << 8;
                for (int b = 0; b < 256; b++) {
                    float diff = qVal - FP8_E4M3_LUT[b];
                    queryLut[dBase | b] = diff * diff;
                }
            }
            for (int cIdx = 0; cIdx < validCandidates; cIdx++) {
                long rowIdx = topRowIds[cIdx];
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
            for (int cIdx = 0; cIdx < validCandidates; cIdx++) {
                long rowIdx = topRowIds[cIdx];
                double currentLimit = bestDists[k - 1];
                double dist = computeExactL2FP4(sidecarQuery, rowIdx, localFp4, numBlocks, currentLimit);
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
            for (int cIdx = 0; cIdx < validCandidates; cIdx++) {
                long rowIdx = topRowIds[cIdx];
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
            for (int cIdx = 0; cIdx < validCandidates; cIdx++) {
                long rowIdx = topRowIds[cIdx];
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
        return queryResults;
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
                bQuery = packQueryByTiers(packed[0]);
                bQueryMask = packQueryByTiers(packed[1]);
            } else if (qMode == 0) { // 1-bit
                bQuery = packQueryByTiers(transformOperator.quantize1Bit(zQuery));
            } else {
                bQuery = null;
            }

            int[] topDists = new int[kCandidate];
            long[] topRowIds = new long[kCandidate];
            Arrays.fill(topDists, Integer.MAX_VALUE);

            for (long rowIdx = 0; rowIdx < size; rowIdx++) {
                evaluateCandidateGate2(rowIdx, activeT, zQuery, bQuery, bQueryMask, topDists, topRowIds, kCandidate);
            }

            int validCandidates = 0;
            while (validCandidates < kCandidate && topDists[validCandidates] != Integer.MAX_VALUE) {
                validCandidates++;
            }

            finalResults[q] = rerankGate3(query, zQuery, topDists, topRowIds, validCandidates, k);
        });

        return finalResults;
    }

    /// Evaluates queries using Gate 0 Multi-Index Hashing (MIH) 4x8 prefix routing with query-directed margin probing.
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
                : Math.max(64, 4 * k));

        IntStream.range(0, numQueries).parallel().forEach(q -> {
            float[] query = queries[q];
            float[] zQuery = transformOperator.preconditionAndRotate(query);

            long[] bQuery;
            long[] bQueryMask = null;
            int[] queryChunkKeys = new int[4];
            if (qMode == 1) { // 2-bit QJL
                float qThreshold = TransformOperator.calculatePercentileThreshold(zQuery, 0.20f);
                long[][] packed = transformOperator.quantize2Bit(zQuery, qThreshold);
                long w0 = packed[0][0];
                queryChunkKeys[0] = (int) (w0 & 0xFFL);
                queryChunkKeys[1] = (int) ((w0 >> 8) & 0xFFL);
                queryChunkKeys[2] = (int) ((w0 >> 16) & 0xFFL);
                queryChunkKeys[3] = (int) ((w0 >> 24) & 0xFFL);
                bQuery = packQueryByTiers(packed[0]);
                bQueryMask = packQueryByTiers(packed[1]);
            } else if (qMode == 0) { // 1-bit
                long[] rawPacked = transformOperator.quantize1Bit(zQuery);
                long w0 = rawPacked[0];
                queryChunkKeys[0] = (int) (w0 & 0xFFL);
                queryChunkKeys[1] = (int) ((w0 >> 8) & 0xFFL);
                queryChunkKeys[2] = (int) ((w0 >> 16) & 0xFFL);
                queryChunkKeys[3] = (int) ((w0 >> 24) & 0xFFL);
                bQuery = packQueryByTiers(rawPacked);
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

            int targetGather = (int) Math.min(size, Math.max(kCandidate * 4, 1024));
            int gatheredCount = 0;

            // Pass 1: Exact matches across all 4 chunks (4 buckets total)
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
                    evaluateCandidateGate2(rowIdxInt & 0xFFFFFFFFL, activeT, zQuery, bQuery, bQueryMask, topDists, topRowIds, kCandidate);
                }
            }

            // Sort bit indices [0..7] within each chunk by ascending margin |zQuery[c*8 + bit]|
            int[][] sortedBits = new int[4][8];
            for (int c = 0; c < 4; c++) {
                for (int b = 0; b < 8; b++) {
                    sortedBits[c][b] = b;
                }
                int baseDim = c * 8;
                for (int i = 1; i < 8; i++) {
                    int keyBit = sortedBits[c][i];
                    float keyMargin = (baseDim + keyBit < dimension) ? Math.abs(zQuery[baseDim + keyBit]) : Float.MAX_VALUE;
                    int j = i - 1;
                    while (j >= 0) {
                        int prevBit = sortedBits[c][j];
                        float prevMargin = (baseDim + prevBit < dimension) ? Math.abs(zQuery[baseDim + prevBit]) : Float.MAX_VALUE;
                        if (prevMargin <= keyMargin) break;
                        sortedBits[c][j + 1] = prevBit;
                        j--;
                    }
                    sortedBits[c][j + 1] = keyBit;
                }
            }

            // Pass 2: Query-directed Hamming-1 neighbor buckets ordered by lowest margin across all 4 chunks
            if (gatheredCount < targetGather) {
                outerPass2:
                for (int rank = 0; rank < 8; rank++) {
                    for (int c = 0; c < 4; c++) {
                        int bit = sortedBits[c][rank];
                        int bKey = queryChunkKeys[c] ^ (1 << bit);
                        long chunkOffBase = c * (256L + 1L) * 4L;
                        long chunkPostBase = (long) c * size * 4L;
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
                            evaluateCandidateGate2(rowIdxInt & 0xFFFFFFFFL, activeT, zQuery, bQuery, bQueryMask, topDists, topRowIds, kCandidate);
                        }
                        if (gatheredCount >= targetGather) break outerPass2;
                    }
                }
            }

            // Pass 3: Query-directed Hamming-2 neighbor buckets across all 4 chunks if still undersaturated
            if (gatheredCount < kCandidate * 2) {
                outerPass3:
                for (int r1 = 0; r1 < 8; r1++) {
                    for (int r2 = r1 + 1; r2 < 8; r2++) {
                        for (int c = 0; c < 4; c++) {
                            int b1 = sortedBits[c][r1];
                            int b2 = sortedBits[c][r2];
                            int bKey = queryChunkKeys[c] ^ (1 << b1) ^ (1 << b2);
                            long chunkOffBase = c * (256L + 1L) * 4L;
                            long chunkPostBase = (long) c * size * 4L;
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
                                evaluateCandidateGate2(rowIdxInt & 0xFFFFFFFFL, activeT, zQuery, bQuery, bQueryMask, topDists, topRowIds, kCandidate);
                            }
                            if (gatheredCount >= kCandidate * 2) break outerPass3;
                        }
                    }
                }
            }

            int validCandidates = 0;
            while (validCandidates < kCandidate && topDists[validCandidates] != Integer.MAX_VALUE) {
                validCandidates++;
            }

            if (validCandidates == 0) {
                int limit = (int) Math.min(size, kCandidate);
                for (int i = 0; i < limit; i++) {
                    topRowIds[i] = i;
                    topDists[i] = 0;
                }
                validCandidates = limit;
            }

            finalResults[q] = rerankGate3(query, zQuery, topDists, topRowIds, validCandidates, k);
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
        double sum = 0.0;
        for (int d = 0; d < dimension; d++) {
            short fp16Val = fp16Segment.get(ValueLayout.JAVA_SHORT, rowOffset + d * 2L);
            float dbVal = Float.float16ToFloat(fp16Val);
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

        if (qMode == 1) { // 2-bit mode
            double sumPositive = 0.0;
            double sumActive = 0.0;
            int tierStartDim = 0;
            for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                int width = tierSizes[tierIdx];
                int halfBytes = (width + 7) / 8;
                long baseOffset = rowIdx * (2L * halfBytes);

                for (int l = 0; l < numLongs; l++) {
                    int validBytes = Math.min(8, halfBytes - l * 8);
                    int limit = Math.min(64, width - (l << 6));
                    long limitMask = limit == 64 ? -1L : (1L << limit) - 1L;
                    long mask = readTierWord(tierSegments[tierIdx], baseOffset + halfBytes + (l * 8L), validBytes) & limitMask;
                    if (mask == 0L)
                        continue;

                    totalMaskPopcount += Long.bitCount(mask);
                    long word = readTierWord(tierSegments[tierIdx], baseOffset + (l * 8L), validBytes) & limitMask;

                    int bitOffset = tierStartDim + (l << 6);
                    long active = mask;
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
                tierStartDim += width;
            }
            return totalMaskPopcount + queryL2Norm - 4.0 * sumPositive + 2.0 * sumActive;
        } else { // 1-bit mode
            double sumPositive = 0.0;
            int tierStartDim = 0;
            for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                int width = tierSizes[tierIdx];
                int tierBytes = (width + 7) / 8;
                long baseOffset = rowIdx * tierBytes;

                for (int l = 0; l < numLongs; l++) {
                    int validBytes = Math.min(8, tierBytes - l * 8);
                    int limit = Math.min(64, width - (l << 6));
                    long limitMask = limit == 64 ? -1L : (1L << limit) - 1L;
                    long word = readTierWord(tierSegments[tierIdx], baseOffset + (l * 8L), validBytes) & limitMask;
                    int bitOffset = tierStartDim + (l << 6);
                    long active = word;
                    while (active != 0) {
                        int bitIdx = Long.numberOfTrailingZeros(active);
                        sumPositive += query[bitOffset + bitIdx];
                        active &= active - 1;
                    }
                }
                tierStartDim += width;
            }
            return query.length + queryL2Norm + 2.0 * querySum - 4.0 * sumPositive;
        }
    }

    @Override
    public long queryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        return queryPlanetaryGrid(queries, families, thresholds, 0L, votingMask);
    }

    @Override
    public long queryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, long rawMaskAddress, MemorySegment votingMask) {
        final MemorySegment finalVotingMask = (votingMask == null && rawMaskAddress != 0L) 
            ? MemorySegment.ofAddress(rawMaskAddress).reinterpret(size) 
            : votingMask;

        if (queries == null || queries.length == 0)
            return 0;
        if (size == 0)
            return 0;

        long currentChunkSize = this.chunkSize;
        long numChunks = (size + currentChunkSize - 1) / currentChunkSize;

        try (Arena arena = Arena.ofShared()) {
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
                    if (finalVotingMask != null) {
                        finalVotingMask.set(ValueLayout.JAVA_BYTE, i, mergedVal);
                    }
                    if (Integer.bitCount(mergedVal & 0xFF) >= 5) {
                        resonantCount++;
                    }
                }
                return resonantCount;
            }).sum();
        }
    }

    private void executeVotingRange(long startIdx, long endIdx, float[][] queries, int[] families, int[] thresholds,
            MemorySegment localMask) {
        int numQueries = queries.length;

        long[][] bQueries = new long[numQueries][];
        long[][] bQueriesMask = new long[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            float[] z = transformOperator.preconditionAndRotate(queries[q]);
            if (qMode == 1) {
                float qThreshold = TransformOperator.calculatePercentileThreshold(z, 0.20f);
                long[][] packed = transformOperator.quantize2Bit(z, qThreshold);
                bQueries[q] = packQueryByTiers(packed[0]);
                bQueriesMask[q] = packQueryByTiers(packed[1]);
            } else {
                bQueries[q] = packQueryByTiers(transformOperator.quantize1Bit(z));
            }
        }

        int T = 0;
        for (int i = 0; i < numTiers; i++) {
            if (cumulativeEnergy[i] >= targetEnergyBudget) {
                T = i;
                break;
            }
        }

        long[] dbWords = new long[totalTierLongs];
        long[] dbMasks = new long[totalTierLongs];

        MemorySegment metaSeg = this.metadataSegment;
        MemorySegment[] localTiers = this.tierSegments;

        for (long i = startIdx; i < endIdx; i++) {
            long metaVal = (metaSeg != null) ? metaSeg.get(ValueLayout.JAVA_LONG, i * 8) : 2L;
            if ((metaVal & 1L) == 1L)
                continue;

            // Load active tiers 0..T
            if (qMode == 1) { // 2-bit mode
                for (int tierIdx = 0; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int width = tierSizes[tierIdx];
                    int offset = tierOffsets[tierIdx];
                    MemorySegment tierSeg = localTiers[tierIdx];
                    if ((width & 63) == 0) {
                        long baseOffset = i * (numLongs * 16L);
                        MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset, dbWords, offset, numLongs);
                        MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L), dbMasks, offset, numLongs);
                    } else {
                        int halfBytes = (width + 7) / 8;
                        long baseOffset = i * (2L * halfBytes);
                        for (int l = 0; l < numLongs; l++) {
                            int validBytes = Math.min(8, halfBytes - l * 8);
                            int remBits = Math.min(64, width - (l << 6));
                            long bitMask = (remBits == 64) ? -1L : ((1L << remBits) - 1L);
                            dbWords[offset + l] = readTierWord(tierSeg, baseOffset + (l * 8L), validBytes) & bitMask;
                            dbMasks[offset + l] = readTierWord(tierSeg, baseOffset + halfBytes + (l * 8L), validBytes) & bitMask;
                        }
                    }
                }
            } else { // 1-bit mode
                for (int tierIdx = 0; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int width = tierSizes[tierIdx];
                    int offset = tierOffsets[tierIdx];
                    MemorySegment tierSeg = localTiers[tierIdx];
                    if ((width & 63) == 0) {
                        long baseOffset = i * (numLongs * 8L);
                        MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset, dbWords, offset, numLongs);
                    } else {
                        int tierBytes = (width + 7) / 8;
                        long baseOffset = i * tierBytes;
                        for (int l = 0; l < numLongs; l++) {
                            int validBytes = Math.min(8, tierBytes - l * 8);
                            int remBits = Math.min(64, width - (l << 6));
                            long bitMask = (remBits == 64) ? -1L : ((1L << remBits) - 1L);
                            dbWords[offset + l] = readTierWord(tierSeg, baseOffset + (l * 8L), validBytes) & bitMask;
                        }
                    }
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
        return cudaQueryPlanetaryGrid(queries, families, thresholds, 0L, votingMask);
    }

    @Override
    public long cudaQueryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, long rawMaskAddress, MemorySegment votingMask) {
        if (queries.length < GPU_BATCH_THRESHOLD || dimension < MIN_DIMENSION_FOR_GPU) {
            return queryPlanetaryGrid(queries, families, thresholds, rawMaskAddress, votingMask);
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
            return queryPlanetaryGrid(queries, families, thresholds, rawMaskAddress, votingMask);
        }

        ByteBuffer votingBuffer = ByteBuffer.allocateDirect(Math.toIntExact(size));
        long votingBufferPtr = CudaMemoryManager.getDirectBufferAddress(votingBuffer);
        CudaMemoryManager.copyFromDevice(votingBufferPtr, deviceVotingMask, size);
        
        final MemorySegment finalVotingMask = (votingMask == null && rawMaskAddress != 0L) 
            ? MemorySegment.ofAddress(rawMaskAddress).reinterpret(size) 
            : votingMask;

        long count = 0;
        for (int i = 0; i < size; i++) {
            byte v = votingBuffer.get(i);
            if (v != 0) {
                count++;
            }
            if (finalVotingMask != null) {
                finalVotingMask.set(ValueLayout.JAVA_BYTE, i, v);
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