package org.pithos;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
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

import com.lmax.disruptor.BlockingWaitStrategy;
import com.lmax.disruptor.ExceptionHandler;
import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.WorkHandler;
import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.dsl.ProducerType;

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

    private final Disruptor<RangeEvent> disruptor;
    private final RingBuffer<RangeEvent> ringBuffer;
    private final int numWorkers;
    private volatile long chunkSize = 20000;

    /// Sets the parallel chunk size for work distribution across Disruptor worker threads.
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

    /// Disruptor range event holding task state for parallel chunks.
    public static class RangeEvent {
        public long startIdx;
        public long endIdx;
        public float[][] queries;
        public int k;
        public int[][][] threadLocalDists;
        public long[][][] threadLocalIds;
        public boolean isVoting;
        public CountDownLatch latch;

        public int[] families;
        public int[] thresholds;
        public MemorySegment[] threadLocalMasks;

        public void setKnn(long startIdx, long endIdx, float[][] queries, int k,
                long[][][] threadLocalIds, int[][][] threadLocalDists, CountDownLatch latch) {
            this.startIdx = startIdx;
            this.endIdx = endIdx;
            this.queries = queries;
            this.k = k;
            this.threadLocalIds = threadLocalIds;
            this.threadLocalDists = threadLocalDists;
            this.isVoting = false;
            this.latch = latch;
        }

        public void setVoting(long startIdx, long endIdx, float[][] queries, int[] families, int[] thresholds,
                MemorySegment[] threadLocalMasks, CountDownLatch latch) {
            this.startIdx = startIdx;
            this.endIdx = endIdx;
            this.queries = queries;
            this.families = families;
            this.thresholds = thresholds;
            this.threadLocalMasks = threadLocalMasks;
            this.isVoting = true;
            this.latch = latch;
        }
    }

    private static class RangeWorkHandler implements WorkHandler<RangeEvent> {
        private final FlatIndex index;
        private final int threadId;

        public RangeWorkHandler(FlatIndex index, int threadId) {
            this.index = index;
            this.threadId = threadId;
        }

        @Override
        public void onEvent(RangeEvent event) throws Exception {
            try {
                if (event.isVoting) {
                    index.executeVotingRange(event.startIdx, event.endIdx, event.queries, event.families,
                            event.thresholds, event.threadLocalMasks[threadId]);
                } else {
                    index.executeKnnRange(event.startIdx, event.endIdx, event.queries, event.k,
                            event.threadLocalIds[threadId], event.threadLocalDists[threadId]);
                }
            } finally {
                event.latch.countDown();
            }
        }
    }

    /// Returns the active sidecar storage format mode.
    public int getSidecarMode() {
        return sidecarMode;
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
        ThreadFactory threadFactory = r -> {
            Thread t = new Thread(r, "pithos-disruptor-worker");
            t.setDaemon(true);
            return t;
        };

        this.disruptor = new Disruptor<>(
                RangeEvent::new,
                1024,
                threadFactory,
                ProducerType.SINGLE,
                new BlockingWaitStrategy());
        this.disruptor.setDefaultExceptionHandler(new ExceptionHandler<RangeEvent>() {
            @Override
            public void handleEventException(Throwable ex, long sequence, RangeEvent event) {
                if (ex instanceof InterruptedException ||
                        ex.getCause() instanceof InterruptedException ||
                        (ex.getMessage() != null && ex.getMessage().contains("InterruptedException"))) {
                    return;
                }
                ex.printStackTrace();
            }

            @Override
            public void handleOnStartException(Throwable ex) {
                ex.printStackTrace();
            }

            @Override
            public void handleOnShutdownException(Throwable ex) {
                if (!(ex instanceof InterruptedException) && !(ex.getCause() instanceof InterruptedException)) {
                    ex.printStackTrace();
                }
            }
        });

        RangeWorkHandler[] handlers = new RangeWorkHandler[numWorkers];
        for (int i = 0; i < numWorkers; i++) {
            handlers[i] = new RangeWorkHandler(this, i);
        }
        this.disruptor.handleEventsWithWorkerPool(handlers);
        this.ringBuffer = this.disruptor.start();
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
        if (!mainPath.toFile().exists()) {
            throw new IOException("Base file path does not exist: " + basePath);
        }

        MemorySegment mappedBase;
        try (FileChannel channel = FileChannel.open(mainPath, StandardOpenOption.READ)) {
            mappedBase = channel.map(FileChannel.MapMode.READ_ONLY, 0, 64, Arena.global());
        }

        byte m0 = mappedBase.get(ValueLayout.JAVA_BYTE, 0);
        byte m1 = mappedBase.get(ValueLayout.JAVA_BYTE, 1);
        byte m2 = mappedBase.get(ValueLayout.JAVA_BYTE, 2);
        byte m3 = mappedBase.get(ValueLayout.JAVA_BYTE, 3);
        if (m0 != 'P' || m1 != 'L' || m2 != 'A' || m3 != 'N') {
            throw new IllegalArgumentException("Invalid file magic: must be PLAN");
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

        int numQueries = queries.length;
        int kCandidate = (int) Math.min(size, (fp16Segment != null || fp8Segment != null || fp4Segment != null) 
                ? Math.max(500, 25 * k) 
                : Math.max(100, 3 * k));

        long[][][] threadLocalIds = new long[numWorkers][numQueries][kCandidate];
        int[][][] threadLocalDists = new int[numWorkers][numQueries][kCandidate];

        for (int w = 0; w < numWorkers; w++) {
            for (int q = 0; q < numQueries; q++) {
                Arrays.fill(threadLocalDists[w][q], Integer.MAX_VALUE);
            }
        }

        long currentChunkSize = this.chunkSize;
        long numChunks = (size + currentChunkSize - 1) / currentChunkSize;
        CountDownLatch latch = new CountDownLatch((int) numChunks);

        for (long c = 0; c < numChunks; c++) {
            long startIdx = c * currentChunkSize;
            long endIdx = Math.min(startIdx + currentChunkSize, size);

            long sequence = ringBuffer.next();
            try {
                RangeEvent event = ringBuffer.get(sequence);
                event.setKnn(startIdx, endIdx, queries, kCandidate, threadLocalIds, threadLocalDists, latch);
            } finally {
                ringBuffer.publish(sequence);
            }
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Search execution was interrupted", e);
        }

        float[][] zQueries = new float[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            zQueries[q] = transformOperator.preconditionAndRotate(queries[q]);
        }

        List<SearchResult>[] finalResults = new List[numQueries];
        IntStream.range(0, numQueries).forEach(q -> {
            List<SearchResult> merged = new ArrayList<>();
            for (int w = 0; w < numWorkers; w++) {
                long[] ids = threadLocalIds[w][q];
                int[] dists = threadLocalDists[w][q];
                for (int i = 0; i < kCandidate; i++) {
                    if (dists[i] != Integer.MAX_VALUE) {
                        merged.add(new SearchResult(ids[i], dists[i]));
                    }
                }
            }

            merged.sort((r1, r2) -> {
                int cmp = Integer.compare(r1.score(), r2.score());
                if (cmp != 0)
                    return cmp;
                return Long.compare(r1.id(), r2.id());
            });

            List<Long> candidates = new ArrayList<>();
            Set<Long> seen = new HashSet<>();
            for (SearchResult r : merged) {
                if (seen.add(r.id())) {
                    candidates.add(r.id());
                    if (candidates.size() >= kCandidate)
                        break;
                }
            }

            double queryL2Norm = 0.0;
            double querySum = 0.0;
            for (float val : zQueries[q]) {
                queryL2Norm += val * val;
                querySum += val;
            }

            if (qMode == 2) {
                List<SearchResult> queryResults = new ArrayList<>();
                int limit = Math.min(k, candidates.size());
                for (int i = 0; i < limit; i++) {
                    long rowIdx = candidates.get(i);
                    long recordId = idsSegment.get(ValueLayout.JAVA_LONG, rowIdx * 8);
                    queryResults.add(new SearchResult(recordId, 0));
                }
                finalResults[q] = queryResults;
                return;
            }

            class RerankedCandidate {
                final long rowIdx;
                final double distance;

                RerankedCandidate(long rowIdx, double distance) {
                    this.rowIdx = rowIdx;
                    this.distance = distance;
                }
            }

            List<RerankedCandidate> reranked = new ArrayList<>();
            if (fp8Segment != null) {
                // Hebel 1: Asymmetric Precomputed Query Lookup-Tables
                float[][] queryLut = new float[dimension][256];
                float[] qRaw = queries[q];
                for (int d = 0; d < dimension; d++) {
                    float qVal = qRaw[d];
                    for (int b = 0; b < 256; b++) {
                        float diff = qVal - FP8_E4M3_LUT[b];
                        queryLut[d][b] = diff * diff;
                    }
                }
                byte[] localFp8 = new byte[dimension];
                for (long rowIdx : candidates) {
                    double dist = computeExactL2FP8_LUT(queryLut, rowIdx, localFp8);
                    reranked.add(new RerankedCandidate(rowIdx, dist));
                }
            } else if (fp4Segment != null) {
                int numBlocks = (dimension + 15) / 16;
                int bytesPerRecord = numBlocks * 9;
                byte[] localFp4 = new byte[bytesPerRecord];
                float[] qRaw = queries[q];
                for (long rowIdx : candidates) {
                    double dist = computeExactL2FP4(qRaw, rowIdx, localFp4, numBlocks);
                    reranked.add(new RerankedCandidate(rowIdx, dist));
                }
            } else if (fp16Segment != null) {
                float[] rawQuery = queries[q];
                short[] localFp16 = new short[dimension];
                for (long rowIdx : candidates) {
                    double dist = computeExactL2FP16(rawQuery, rowIdx, localFp16);
                    reranked.add(new RerankedCandidate(rowIdx, dist));
                }
            } else {
                for (long rowIdx : candidates) {
                    double dist = computeAsymmetricL2DistanceOffHeap(zQueries[q], queryL2Norm, querySum, rowIdx);
                    reranked.add(new RerankedCandidate(rowIdx, dist));
                }
            }

            reranked.sort((c1, c2) -> Double.compare(c1.distance, c2.distance));

            List<SearchResult> queryResults = new ArrayList<>();
            int limit = Math.min(k, reranked.size());
            for (int i = 0; i < limit; i++) {
                RerankedCandidate c = reranked.get(i);
                long recordId = idsSegment.get(ValueLayout.JAVA_LONG, c.rowIdx * 8);
                queryResults.add(new SearchResult(recordId, (int) (c.distance * 1000000.0)));
            }
            finalResults[q] = queryResults;
        });

        return finalResults;
    }

    private void executeKnnRange(long startIdx, long endIdx, float[][] queries, int k, long[][] myIds,
            int[][] myDists) {
        int numQueries = queries.length;

        // Binarize queries
        long[][] bQueries = new long[numQueries][];
        long[][] bQueriesMask = new long[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            if (qMode == 1) {
                float[] z = transformOperator.preconditionAndRotate(queries[q]);
                float qThreshold = TransformOperator.calculatePercentileThreshold(z, 0.20f);
                long[][] packed = transformOperator.quantize2Bit(z, qThreshold);
                bQueries[q] = packed[0];
                bQueriesMask[q] = packed[1];
            } else if (qMode == 0) {
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

        MemorySegment metaSeg = this.metadataSegment;
        MemorySegment[] localTiers = this.tierSegments;
        MemorySegment tier0 = localTiers[0];

        // SIMD Float-L2 dispatch (D <= 32)
        if (dimension <= SIMD_FLOAT_DIM_THRESHOLD && qMode == 0) {
            float[][] rotQueries = new float[numQueries][];
            for (int q = 0; q < numQueries; q++)
                rotQueries[q] = transformOperator.preconditionAndRotate(queries[q]);
            float[] dbFloat = new float[dimension];
            int numLongs0 = tierLongs[0];
            for (long i = startIdx; i < endIdx; i++) {
                if ((metaSeg.get(ValueLayout.JAVA_LONG, i * 8) & 1L) == 1L)
                    continue;
                long baseOff = i * (numLongs0 * 8L);
                long word = tier0.get(ValueLayout.JAVA_LONG, baseOff);
                for (int d = 0; d < dimension; d++) {
                    dbFloat[d] = ((word >>> d) & 1L) != 0L ? 1.0f : -1.0f;
                }
                for (int q = 0; q < numQueries; q++) {
                    int iDist = (int) (transformOperator.computeL2Float(rotQueries[q], dbFloat) * 1000f);
                    int[] dists = myDists[q];
                    long[] ids = myIds[q];
                    if (iDist < dists[k - 1]) {
                        int pos = k - 1;
                        while (pos > 0 && iDist < dists[pos - 1]) {
                            dists[pos] = dists[pos - 1];
                            ids[pos] = ids[pos - 1];
                            pos--;
                        }
                        dists[pos] = iDist;
                        ids[pos] = i;
                    }
                }
            }
            return;
        }

        // QMode 2: Float-Hybrid raw bypass
        if (qMode == 2) {
            float[][] rotQueries = new float[numQueries][];
            for (int q = 0; q < numQueries; q++)
                rotQueries[q] = transformOperator.preconditionAndRotate(queries[q]);
            float[] dbFloat = new float[dimension];
            for (long i = startIdx; i < endIdx; i++) {
                if ((metaSeg.get(ValueLayout.JAVA_LONG, i * 8) & 1L) == 1L)
                    continue;
                int dimOffset = 0;
                for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                    int width = tiers[tierIdx] - (tierIdx == 0 ? 0 : tiers[tierIdx - 1]);
                    long baseOffset = i * (width * 4L);
                    MemorySegment.copy(localTiers[tierIdx], ValueLayout.JAVA_FLOAT, baseOffset, dbFloat, dimOffset, width);
                    dimOffset += width;
                }
                for (int q = 0; q < numQueries; q++) {
                    int iDist = (int) (transformOperator.computeL2Float(rotQueries[q], dbFloat) * 1000f);
                    int[] dists = myDists[q];
                    long[] ids = myIds[q];
                    if (iDist < dists[k - 1]) {
                        int pos = k - 1;
                        while (pos > 0 && iDist < dists[pos - 1]) {
                            dists[pos] = dists[pos - 1];
                            ids[pos] = ids[pos - 1];
                            pos--;
                        }
                        dists[pos] = iDist;
                        ids[pos] = i;
                    }
                }
            }
            return;
        }

        // Standard Hamming Path (QMode 0 & 1): Sequential Loading with Early Exit
        int totalLongs = dimension / 64;
        long[] dbWords = new long[totalLongs];
        long[] dbMasks = new long[totalLongs];

        for (long i = startIdx; i < endIdx; i++) {
            if ((metaSeg.get(ValueLayout.JAVA_LONG, i * 8) & 1L) == 1L)
                continue;

            // Load all active tiers up to T for record i first
            if (qMode == 1) { // 2-bit mode
                for (int tierIdx = 0; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int offset = tierOffsets[tierIdx];
                    MemorySegment tierSeg = localTiers[tierIdx];
                    long baseOffset = i * (numLongs * 16L);
                    MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset, dbWords, offset, numLongs);
                    MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset + (numLongs * 8L), dbMasks, offset, numLongs);
                }
            } else { // 1-bit mode
                for (int tierIdx = 0; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int offset = tierOffsets[tierIdx];
                    MemorySegment tierSeg = localTiers[tierIdx];
                    long baseOffset = i * (numLongs * 8L);
                    MemorySegment.copy(tierSeg, ValueLayout.JAVA_LONG, baseOffset, dbWords, offset, numLongs);
                }
            }

            // Evaluate queries against pre-loaded database words
            for (int q = 0; q < numQueries; q++) {
                int[] dists = myDists[q];
                int currentLimit = dists[k - 1];

                int totalDist = 0;

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

                    // If dynamic threshold is exceeded, break early
                    if (totalDist > currentLimit) {
                        break;
                    }
                }

                if (totalDist < currentLimit) {
                    long[] ids = myIds[q];
                    int pos = k - 1;
                    while (pos > 0 && totalDist < dists[pos - 1]) {
                        dists[pos] = dists[pos - 1];
                        ids[pos] = ids[pos - 1];
                        pos--;
                    }
                    dists[pos] = totalDist;
                    ids[pos] = i; // Save the internal row index
                }
            }
        }
    }

    private double computeExactL2FP8_LUT(float[][] queryLut, long rowIdx, byte[] localFp8) {
        long rowOffset = rowIdx * (long) dimension;
        MemorySegment.copy(fp8Segment, ValueLayout.JAVA_BYTE, rowOffset, localFp8, 0, dimension);
        double sum = 0.0;
        for (int d = 0; d < dimension; d++) {
            sum += queryLut[d][localFp8[d] & 0xFF];
        }
        return sum;
    }

    private double computeExactL2FP4(float[] rawQuery, long rowIdx, byte[] localFp4, int numBlocks) {
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
        }
        return sum;
    }

    private double computeExactL2FP16(float[] rawQuery, long rowIdx, short[] localFp16) {
        long rowOffset = rowIdx * dimension * 2L;
        MemorySegment.copy(fp16Segment, ValueLayout.JAVA_SHORT, rowOffset, localFp16, 0, dimension);
        double sum = 0.0;
        for (int d = 0; d < dimension; d++) {
            float dbVal = Float.float16ToFloat(localFp16[d]);
            double diff = rawQuery[d] - dbVal;
            sum += diff * diff;
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
        CountDownLatch latch = new CountDownLatch((int) numChunks);

        Arena arena = Arena.global();
        MemorySegment[] threadLocalMasks = new MemorySegment[numWorkers];
        for (int w = 0; w < numWorkers; w++) {
            threadLocalMasks[w] = arena.allocate(size);
        }

        for (long c = 0; c < numChunks; c++) {
            long startIdx = c * currentChunkSize;
            long endIdx = Math.min(startIdx + currentChunkSize, size);

            long sequence = ringBuffer.next();
            try {
                RangeEvent event = ringBuffer.get(sequence);
                event.setVoting(startIdx, endIdx, queries, families, thresholds, threadLocalMasks, latch);
            } finally {
                ringBuffer.publish(sequence);
            }
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Voting execution was interrupted", e);
        }

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
            long metaVal = metaSeg.get(ValueLayout.JAVA_LONG, i * 8);
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
        if (disruptor != null) {
            try {
                disruptor.shutdown();
            } catch (Throwable t) {
                // ignore
            }
        }
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