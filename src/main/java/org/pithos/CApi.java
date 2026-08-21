package org.pithos;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CLongPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/// # CApi
///
/// C-API entry points for the Pithos binary vector search engine.
/// Exposes high-performance native methods to C/C++/Python callers via GraalVM `@CEntryPoint`.
///
/// ### Memory Management Guidelines:
/// - **Output Allocation:** The native caller allocates all memory for output pointers (`outIds`, `outDistances`, `votingMask`).
/// - **Zero-Copy / Off-Heap:** Pithos maps datasets off-heap using POSIX virtual memory (`mmap`) via Java Foreign Function & Memory (FFM) API.
/// - **DMA / FPGA Direct Access:** Direct base virtual addresses are retrievable via `vdb_get_tier_address` for zero-copy hardware DMA transfers.
///
/// ### API Return Codes:
/// | Code | Name | Description |
/// | :--- | :--- | :--- |
/// | `0` | `SUCCESS` | Operation completed successfully. |
/// | `-1` | `ERR_DB_NOT_INIT` | Global database coordinator not initialized. Call `vdb_init` first. |
/// | `-2` | `ERR_INDEX_NOT_FOUND` | Requested logical index name is not registered. |
/// | `-3` | `ERR_INVALID_OPERATION` | Invalid parameter or operation not supported. |
/// | `-4` | `ERR_INTERNAL_EXCEPTION` | Unexpected internal exception occurred. Stack trace printed to stderr. |
/// | `-5` | `ERR_FILE_IO` | Could not read or write file(s) on disk. |
/// | `-6` | `ERR_UNSUPPORTED_LAYOUT` | Index layout mismatch. |
public class CApi {
    private static VectorDb db;

    private CApi() {
    }

    /// Initializes the global database coordinator. Must be called once before any database operations.
    ///
    /// @param thread the GraalVM isolate thread context
    /// @return `0` on success, `-4` on internal exception
    @CEntryPoint(name = "vdb_init")
    public static int init(IsolateThread thread) {
        try {
            db = new VectorDb();
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Maps an existing compiled database off-heap into virtual memory without custom weights.
    ///
    /// @param thread the GraalVM isolate thread context
    /// @param name C string specifying the unique logical name
    /// @param path C string specifying the base filepath of the compiled index on disk
    /// @return `0` on success, negative error code on failure
    @CEntryPoint(name = "vdb_load_index")
    public static int loadIndex(IsolateThread thread, CCharPointer name, CCharPointer path) {
        if (db == null) {
            return -1;
        }
        if (name.isNull() || path.isNull()) {
            return -3;
        }
        try {
            String indexName = CTypeConversion.toJavaString(name);
            String filePath = CTypeConversion.toJavaString(path);
            db.loadIndex(indexName, filePath, null, 0);
            return 0;
        } catch (IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Maps an existing compiled database off-heap, supplying frozen projection/LoRA weights W ∈ ℝ^(D × D₀).
    ///
    /// The model weights are used to compute SVD singular values σᵢ, constructing the
    /// Matryoshka cumulative spectral energy distribution Φ(k) to dynamically target a recall energy budget τ.
    ///
    /// @param thread the GraalVM isolate thread context
    /// @param name C string specifying unique logical name
    /// @param path C string specifying base filepath on disk
    /// @param weights C float array pointer storing the weight matrix of size D × D₀
    /// @param loraDim inner bottleneck dimension D₀
    /// @return `0` on success, negative error code on failure
    @CEntryPoint(name = "vdb_load_index_with_weights")
    public static int loadIndexWithWeights(IsolateThread thread, CCharPointer name, CCharPointer path,
            CFloatPointer weights, int loraDim) {
        if (db == null) {
            return -1;
        }
        if (name.isNull() || path.isNull() || weights.isNull() || loraDim <= 0) {
            return -3;
        }
        try {
            String indexName = CTypeConversion.toJavaString(name);
            String filePath = CTypeConversion.toJavaString(path);

            Index tempIdx = FlatIndex.mapFile(filePath, null, 0);
            int dim = tempIdx.getDimension();
            tempIdx.close();

            float[] javaWeights = new float[dim * loraDim];
            for (int i = 0; i < javaWeights.length; i++) {
                javaWeights[i] = weights.read(i);
            }

            db.loadIndex(indexName, filePath, javaWeights, loraDim);
            return 0;
        } catch (IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Retrieves metadata attributes for a loaded index.
    ///
    /// @param thread the GraalVM isolate thread context
    /// @param indexName C string identifying the target loaded index
    /// @param outDimension output pointer for vector dimensionality (D)
    /// @param outSize output pointer for record count (N)
    /// @param outPlanetId output pointer for planet ID code
    /// @param outPlanetRadius output pointer for planet radius in meters
    /// @param outTiersCount output pointer for tier count (T)
    /// @return `0` on success, negative error code on failure
    @CEntryPoint(name = "vdb_get_info")
    public static int getInfo(IsolateThread thread, CCharPointer indexName,
            CIntPointer outDimension, CLongPointer outSize,
            CCharPointer outPlanetId, CLongPointer outPlanetRadius,
            CIntPointer outTiersCount) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull() || outDimension.isNull() || outSize.isNull() || outPlanetId.isNull()
                || outPlanetRadius.isNull() || outTiersCount.isNull()) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }

            outDimension.write(0, index.getDimension());
            outSize.write(0, index.size());
            outPlanetId.write(0, (byte) index.getPlanetId());
            outPlanetRadius.write(0, index.getPlanetRadius());
            outTiersCount.write(0, index.getTierCount());
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Performs a batch k-NN search on raw continuous float query vectors.
    ///
    /// @param thread the GraalVM isolate thread context
    /// @param indexName C string identifying target index
    /// @param queries contiguous C float array pointer of size `numQueries x D`
    /// @param numQueries number of query vectors in the batch
    /// @param k top-k nearest neighbors per query
    /// @param outIds pre-allocated C long array pointer of size `numQueries x k`
    /// @param outDistances pre-allocated C int array pointer of size `numQueries x k` (scaled by 1,000,000)
    /// @return `0` on success, negative error code on failure
    @CEntryPoint(name = "vdb_batch_search")
    public static int batchSearch(IsolateThread thread, CCharPointer indexName, CFloatPointer queries, int numQueries,
            int k,
            CLongPointer outIds, CIntPointer outDistances) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull() || queries.isNull() || outIds.isNull() || outDistances.isNull() || numQueries <= 0 || k <= 0) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }

            int dim = index.getDimension();
            float[][] javaQueries = new float[numQueries][dim];
            for (int q = 0; q < numQueries; q++) {
                int qOffset = q * dim;
                for (int d = 0; d < dim; d++) {
                    javaQueries[q][d] = queries.read(qOffset + d);
                }
            }

            List<Index.SearchResult>[] results = index.batchSearch(javaQueries, k);

            for (int q = 0; q < numQueries; q++) {
                List<Index.SearchResult> queryResults = results[q];
                int count = queryResults.size();
                for (int i = 0; i < k; i++) {
                    long outId = -1;
                    int outDist = Integer.MAX_VALUE;
                    if (i < count) {
                        Index.SearchResult r = queryResults.get(i);
                        outId = r.id();
                        outDist = r.score();
                    }
                    outIds.write(q * k + i, outId);
                    outDistances.write(q * k + i, outDist);
                }
            }
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Performs multi-family resonant voting across scientific criteria, writing bitmasks into `votingMask`.
    ///
    /// @param thread the GraalVM isolate thread context
    /// @param indexName C string identifying the index
    /// @param queries contiguous C float array of shape `numQueries x D`
    /// @param queryFamilies C int array of semantic families (0..7)
    /// @param queryThresholds C int array of Hamming distance cutoff thresholds
    /// @param numQueries number of queries in the batch
    /// @param votingMask pre-allocated C byte array of size N bytes
    /// @return number of resonant candidate records with ≥ 5 votes, or negative error code
    @CEntryPoint(name = "vdb_query_planetary_grid")
    public static long queryPlanetaryGrid(IsolateThread thread, CCharPointer indexName, CFloatPointer queries,
            CIntPointer queryFamilies, CIntPointer queryThresholds, int numQueries,
            CCharPointer votingMask) {

        if (db == null) {
            return -1;
        }
        if (indexName.isNull() || queries.isNull() || queryFamilies.isNull() || queryThresholds.isNull()
                || votingMask.isNull() || numQueries <= 0) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }

            int dim = index.getDimension();
            long totalTiles = index.size();

            float[][] javaQueries = new float[numQueries][dim];
            int[] javaFamilies = new int[numQueries];
            int[] javaThresholds = new int[numQueries];

            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < dim; j++) {
                    javaQueries[q][j] = queries.read(q * dim + j);
                }
                javaFamilies[q] = queryFamilies.read(q);
                javaThresholds[q] = queryThresholds.read(q);
            }

            long rawAddress = votingMask.rawValue();
            MemorySegment maskSegment = MemorySegment.ofAddress(rawAddress).reinterpret(totalTiles);

            return index.queryPlanetaryGrid(javaQueries, javaFamilies, javaThresholds, maskSegment);
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Compiles raw continuous float records into a multi-tier binary database layout.
    @CEntryPoint(name = "vdb_compile_index_file")
    public static int compileIndexFile(IsolateThread thread, CCharPointer path, byte planetId, long planetRadius,
            int dimension, CIntPointer tiers, int numTiers,
            CLongPointer ids, CFloatPointer vectors, int numRecords, int qMode) {
        if (path.isNull() || tiers.isNull() || ids.isNull() || vectors.isNull() || dimension <= 0 || numTiers <= 0 || numRecords <= 0) {
            return -3;
        }
        try {
            String filePath = CTypeConversion.toJavaString(path);

            int[] javaTiers = new int[numTiers];
            for (int i = 0; i < numTiers; i++) {
                javaTiers[i] = tiers.read(i);
            }

            List<VectorRecord> records = new ArrayList<>(numRecords);
            for (int i = 0; i < numRecords; i++) {
                long id = ids.read(i);
                float[] vector = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    vector[j] = vectors.read(i * dimension + j);
                }
                records.add(new VectorRecord(id, vector));
            }

            VectorDb.compileIndexFile(filePath, planetId, planetRadius, dimension, javaTiers, records, qMode);
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Compiles raw continuous float records into a multi-tier binary database layout with configurable sidecar mode.
    @CEntryPoint(name = "vdb_compile_index_file_ext")
    public static int compileIndexFileExt(IsolateThread thread, CCharPointer path, byte planetId, long planetRadius,
            int dimension, CIntPointer tiers, int numTiers,
            CLongPointer ids, CFloatPointer vectors, int numRecords, int qMode, int sidecarMode) {
        if (path.isNull() || tiers.isNull() || ids.isNull() || vectors.isNull() || dimension <= 0 || numTiers <= 0 || numRecords <= 0) {
            return -3;
        }
        try {
            String filePath = CTypeConversion.toJavaString(path);

            int[] javaTiers = new int[numTiers];
            for (int i = 0; i < numTiers; i++) {
                javaTiers[i] = tiers.read(i);
            }

            List<VectorRecord> records = new ArrayList<>(numRecords);
            for (int i = 0; i < numRecords; i++) {
                long id = ids.read(i);
                float[] vector = new float[dimension];
                int offset = i * dimension;
                for (int j = 0; j < dimension; j++) {
                    vector[j] = vectors.read(offset + j);
                }
                records.add(new VectorRecord(id, vector));
            }

            VectorDb.compileIndexFile(filePath, planetId, planetRadius, dimension, javaTiers, records, qMode, sidecarMode);
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Compiles raw continuous float records into a universal schema-agnostic single-file .pithos container (DIOGENES format).
    @CEntryPoint(name = "vdb_compile_container")
    public static int compileContainer(IsolateThread thread, CCharPointer path, int dimension, CIntPointer tiers, int numTiers,
            CLongPointer ids, CFloatPointer vectors, int numRecords, int metricType, int qMode, int sidecarMode,
            CCharPointer metadataPayload, int metadataLen, CCharPointer metadataFormat, CCharPointer userMetadataJson) {
        if (path.isNull() || tiers.isNull() || ids.isNull() || vectors.isNull() || dimension <= 0 || numTiers <= 0 || numRecords <= 0) {
            return -3;
        }
        try {
            String filePath = CTypeConversion.toJavaString(path);
            int[] javaTiers = new int[numTiers];
            for (int i = 0; i < numTiers; i++) {
                javaTiers[i] = tiers.read(i);
            }

            List<VectorRecord> records = new ArrayList<>(numRecords);
            for (int i = 0; i < numRecords; i++) {
                long id = ids.read(i);
                float[] vector = new float[dimension];
                int offset = i * dimension;
                for (int j = 0; j < dimension; j++) {
                    vector[j] = vectors.read(offset + j);
                }
                records.add(new VectorRecord(id, vector));
            }

            byte[] metaBytes = null;
            if (metadataPayload.isNonNull() && metadataLen > 0) {
                metaBytes = new byte[metadataLen];
                for (int i = 0; i < metadataLen; i++) {
                    metaBytes[i] = metadataPayload.read(i);
                }
            }

            String metaFormat = metadataFormat.isNonNull() ? CTypeConversion.toJavaString(metadataFormat) : "raw";
            String userJson = userMetadataJson.isNonNull() ? CTypeConversion.toJavaString(userMetadataJson) : null;

            VectorDb.compileContainer(filePath, dimension, javaTiers, records, metricType, qMode, sidecarMode,
                    metaBytes, metaFormat, userJson);
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Copies user metadata JSON string into buffer. Returns length of JSON string or -1 on error.
    @CEntryPoint(name = "vdb_get_user_metadata")
    public static int getUserMetadata(IsolateThread thread, CCharPointer indexName, CCharPointer outBuf, int maxLen) {
        if (db == null) return -1;
        if (indexName.isNull() || outBuf.isNull() || maxLen <= 0) return -3;
        String idxName = CTypeConversion.toJavaString(indexName);
        String metaJson = db.getUserMetadata(idxName);
        if (metaJson == null) return 0;
        byte[] bytes = metaJson.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int copyLen = Math.min(bytes.length, maxLen - 1);
        for (int i = 0; i < copyLen; i++) {
            outBuf.write(i, (byte) bytes[i]);
        }
        outBuf.write(copyLen, (byte) 0);
        return bytes.length;
    }

    /// Returns the sidecar format mode (0=None, 1=FP16, 2=FP8, 3=FP4) for a loaded index.
    @CEntryPoint(name = "vdb_get_sidecar_mode")
    public static int getSidecarMode(IsolateThread thread, CCharPointer indexName) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull()) {
            return -3;
        }
        String idxName = CTypeConversion.toJavaString(indexName);
        Index index = db.getIndex(idxName);
        if (index == null) {
            return -2;
        }
        return index.getSidecarMode();
    }

    /// Returns the total record count (N) for a loaded index.
    @CEntryPoint(name = "vdb_size")
    public static long size(IsolateThread thread, CCharPointer indexName) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull()) {
            return -3;
        }
        String idxName = CTypeConversion.toJavaString(indexName);
        Index index = db.getIndex(idxName);
        if (index == null) {
            return -2;
        }
        return index.size();
    }

    /// Unmaps and drops an index from memory space, releasing all file mappings.
    @CEntryPoint(name = "vdb_drop_index")
    public static int dropIndex(IsolateThread thread, CCharPointer indexName) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull()) {
            return -3;
        }
        String idxName = CTypeConversion.toJavaString(indexName);
        return db.dropIndex(idxName) ? 0 : -2;
    }

    /// Adjusts the parallel scan chunk size for multi-threaded search dispatching.
    @CEntryPoint(name = "vdb_set_chunk_size")
    public static int setChunkSize(IsolateThread thread, CCharPointer indexName, long chunkSize) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull()) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex flatIndex) {
                flatIndex.setChunkSize(chunkSize);
                return 0;
            }
            return -3;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Sets the dynamic target spectral energy budget threshold τ ∈ (0, 1] for Matryoshka early-exit tier truncation.
    @CEntryPoint(name = "vdb_set_energy_budget")
    public static int setEnergyBudget(IsolateThread thread, CCharPointer indexName, double tau) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull()) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex flatIndex) {
                flatIndex.setTargetEnergyBudget(tau);
                return 0;
            }
            return -3;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// **Hardware Acceleration & DMA Direct Access Endpoint:**
    /// Retrieves the raw off-heap virtual memory address and length of a specific index tier.
    @CEntryPoint(name = "vdb_get_tier_address")
    public static int getTierAddress(IsolateThread thread, CCharPointer indexName, int tierIdx,
            CLongPointer outAddress, CLongPointer outLength) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull() || outAddress.isNull() || outLength.isNull()) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex flatIdx) {
                long addr = flatIdx.getTierAddress(tierIdx);
                long len = flatIdx.getTierByteSize(tierIdx);
                if (addr == 0) {
                    return -3;
                }
                outAddress.write(0, addr);
                outLength.write(0, len);
                return 0;
            }
            return -6;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Retrieves the raw off-heap virtual memory address and length of the metadata column segment.
    @CEntryPoint(name = "vdb_get_metadata_address")
    public static int getMetadataAddress(IsolateThread thread, CCharPointer indexName,
            CLongPointer outAddress, CLongPointer outLength) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull() || outAddress.isNull() || outLength.isNull()) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex flatIdx) {
                long addr = flatIdx.getMetadataAddress();
                long len = flatIdx.getMetadataByteSize();
                outAddress.write(0, addr);
                outLength.write(0, len);
                return 0;
            }
            return -6;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Retrieves the raw off-heap virtual memory address and length of the record ID column segment.
    @CEntryPoint(name = "vdb_get_ids_address")
    public static int getIdsAddress(IsolateThread thread, CCharPointer indexName,
            CLongPointer outAddress, CLongPointer outLength) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull() || outAddress.isNull() || outLength.isNull()) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex flatIdx) {
                long addr = flatIdx.getIdsAddress();
                long len = flatIdx.getIdsByteSize();
                outAddress.write(0, addr);
                outLength.write(0, len);
                return 0;
            }
            return -6;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Binarizes a single float vector using Rademacher signs preconditioning and block-diagonal Hadamard rotation.
    @CEntryPoint(name = "vdb_transform_and_quantize")
    public static int transformAndQuantize(IsolateThread thread, CCharPointer indexName, CFloatPointer inVector,
            CLongPointer outPacked) {
        if (db == null) {
            return -1;
        }
        if (indexName.isNull() || inVector.isNull() || outPacked.isNull()) {
            return -3;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex flatIdx) {
                int dim = flatIdx.getDimension();
                float[] javaVector = new float[dim];
                for (int j = 0; j < dim; j++) {
                    javaVector[j] = inVector.read(j);
                }
                long[] packed = flatIdx.getTransformOperator().transformAndQuantize(javaVector);
                for (int i = 0; i < packed.length; i++) {
                    outPacked.write(i, packed[i]);
                }
                return 0;
            }
            return -6;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Closes the database coordinator and frees all allocations.
    @CEntryPoint(name = "vdb_close")
    public static int closeDb(IsolateThread thread) {
        if (db != null) {
            try {
                db.close();
                db = null;
            } catch (Throwable t) {
                t.printStackTrace();
                return -4;
            }
        }
        return 0;
    }

    /// Triggers explicit garbage collection and memory compaction inside the GraalVM isolate.
    @CEntryPoint(name = "vdb_shrink_to_fit")
    public static int shrinkToFit(IsolateThread thread) {
        try {
            System.gc();
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    // =========================================================================
    // LSM Delta Buffer C-API
    // =========================================================================

    /// Creates an in-memory writable delta buffer for real-time inserts.
    @CEntryPoint(name = "vdb_create_delta_buffer")
    public static int createDeltaBuffer(IsolateThread thread, CCharPointer indexName, int flushThreshold) {
        if (db == null)
            return -1;
        if (indexName.isNull())
            return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            db.createDeltaBuffer(idxName, flushThreshold);
            return 0;
        } catch (IllegalArgumentException e) {
            return -2;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Inserts a single continuous float vector into the delta buffer.
    @CEntryPoint(name = "vdb_insert")
    public static int insert(IsolateThread thread, CCharPointer indexName, long id,
            CFloatPointer vector) {
        if (db == null)
            return -1;
        if (indexName.isNull() || vector.isNull())
            return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null)
                return -2;
            int dim = index.getDimension();
            float[] javaVector = new float[dim];
            for (int i = 0; i < dim; i++) {
                javaVector[i] = vector.read(i);
            }
            boolean ok = db.insertIntoDelta(idxName, id, javaVector);
            return ok ? 0 : -2;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Soft-deletes a record from the delta buffer (tombstone).
    @CEntryPoint(name = "vdb_delete_from_delta")
    public static int deleteFromDelta(IsolateThread thread, CCharPointer indexName, long id) {
        if (db == null)
            return -1;
        if (indexName.isNull())
            return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            if (db.getDeltaBuffer(idxName) == null)
                return -2;
            return db.deleteFromDelta(idxName, id) ? 1 : 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Returns the live record count in the delta buffer.
    @CEntryPoint(name = "vdb_delta_size")
    public static long deltaSize(IsolateThread thread, CCharPointer indexName) {
        if (db == null)
            return -1;
        if (indexName.isNull())
            return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            DeltaBuffer buf = db.getDeltaBuffer(idxName);
            if (buf == null)
                return -2;
            return buf.liveSize();
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Checks if the active record count in the delta buffer exceeds the flush threshold.
    @CEntryPoint(name = "vdb_needs_flush")
    public static int needsFlush(IsolateThread thread, CCharPointer indexName) {
        if (db == null)
            return -1;
        if (indexName.isNull())
            return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            DeltaBuffer buf = db.getDeltaBuffer(idxName);
            if (buf == null)
                return -2;
            return buf.needsFlush() ? 1 : 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Performs a unified search querying both the memory-mapped base index and the delta buffer.
    @CEntryPoint(name = "vdb_search_merged")
    public static int searchMerged(IsolateThread thread, CCharPointer indexName,
            CFloatPointer query, int k,
            CLongPointer outIds, CIntPointer outDistances) {
        if (db == null)
            return -1;
        if (indexName.isNull() || query.isNull() || outIds.isNull() || outDistances.isNull() || k <= 0)
            return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null)
                return -2;

            int dim = index.getDimension();
            float[] javaQuery = new float[dim];
            for (int i = 0; i < dim; i++) {
                javaQuery[i] = query.read(i);
            }

            List<Index.SearchResult> results = db.searchMerged(idxName, javaQuery, k);
            int count = results.size();
            for (int i = 0; i < k; i++) {
                if (i < count) {
                    outIds.write(i, results.get(i).id());
                    outDistances.write(i, results.get(i).score());
                } else {
                    outIds.write(i, -1L);
                    outDistances.write(i, Integer.MAX_VALUE);
                }
            }
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Backs up all live entries from the delta buffer to a binary file.
    @CEntryPoint(name = "vdb_backup_delta")
    public static int backupDelta(IsolateThread thread, CCharPointer indexName, CCharPointer path) {
        if (db == null)
            return -1;
        if (indexName.isNull() || path.isNull())
            return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            String filePath = CTypeConversion.toJavaString(path);
            db.backupDelta(idxName, filePath);
            return 0;
        } catch (IllegalStateException e) {
            return -2;
        } catch (java.io.IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Restores a delta buffer from a previously serialized binary file.
    @CEntryPoint(name = "vdb_restore_delta")
    public static int restoreDelta(IsolateThread thread, CCharPointer indexName,
            CCharPointer path, int flushThreshold) {
        if (db == null)
            return -1;
        if (indexName.isNull() || path.isNull())
            return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            if (db.getIndex(idxName) == null)
                return -2;
            String filePath = CTypeConversion.toJavaString(path);
            db.restoreDelta(idxName, filePath, flushThreshold);
            return 0;
        } catch (java.io.IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    // =========================================================================
    // CUDA Acceleration C-API
    // =========================================================================

    @CEntryPoint(name = "vdb_cuda_init")
    public static int cudaInit(IsolateThread thread, int deviceId) {
        if (db == null) return -1;
        try {
            return db.cudaInit(deviceId);
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    @CEntryPoint(name = "vdb_cuda_shutdown")
    public static int cudaShutdown(IsolateThread thread) {
        if (db == null) return -1;
        try {
            db.cudaShutdown();
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    @CEntryPoint(name = "vdb_cuda_is_available")
    public static int cudaIsAvailable(IsolateThread thread) {
        if (db == null) return 0;
        return db.cudaIsAvailable() ? 1 : 0;
    }

    @CEntryPoint(name = "vdb_cuda_batch_search")
    public static int cudaBatchSearch(IsolateThread thread, CCharPointer indexName, CFloatPointer queries, int numQueries,
            int k, CLongPointer outIds, CIntPointer outDistances) {
        if (db == null) return -1;
        if (indexName.isNull() || queries.isNull() || outIds.isNull() || outDistances.isNull() || numQueries <= 0 || k <= 0) return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) return -2;

            int dim = index.getDimension();
            float[][] javaQueries = new float[numQueries][dim];
            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < dim; j++) {
                    javaQueries[q][j] = queries.read(q * dim + j);
                }
            }

            List<Index.SearchResult>[] results = db.cudaBatchSearch(idxName, javaQueries, k);

            for (int q = 0; q < numQueries; q++) {
                List<Index.SearchResult> queryResults = results[q];
                int count = queryResults.size();
                for (int i = 0; i < k; i++) {
                    long outId = -1;
                    int outDist = Integer.MAX_VALUE;
                    if (i < count) {
                        Index.SearchResult r = queryResults.get(i);
                        outId = r.id();
                        outDist = r.score();
                    }
                    outIds.write(q * k + i, outId);
                    outDistances.write(q * k + i, outDist);
                }
            }
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    @CEntryPoint(name = "vdb_cuda_query_planetary_grid")
    public static long cudaQueryPlanetaryGrid(IsolateThread thread, CCharPointer indexName, CFloatPointer queries,
            CIntPointer queryFamilies, CIntPointer queryThresholds, int numQueries,
            CCharPointer votingMask) {
        if (db == null) return -1;
        if (indexName.isNull() || queries.isNull() || queryFamilies.isNull() || queryThresholds.isNull() || votingMask.isNull() || numQueries <= 0) return -3;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) return -2;

            int dim = index.getDimension();
            long totalTiles = index.size();

            float[][] javaQueries = new float[numQueries][dim];
            int[] javaFamilies = new int[numQueries];
            int[] javaThresholds = new int[numQueries];

            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < dim; j++) {
                    javaQueries[q][j] = queries.read(q * dim + j);
                }
                javaFamilies[q] = queryFamilies.read(q);
                javaThresholds[q] = queryThresholds.read(q);
            }

            long rawAddress = votingMask.rawValue();
            MemorySegment maskSegment = MemorySegment.ofAddress(rawAddress).reinterpret(totalTiles);

            return db.cudaQueryPlanetaryGrid(idxName, javaQueries, javaFamilies, javaThresholds, maskSegment);
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /// Compacts multiple compiled indexes into a single consolidated index.
    @CEntryPoint(name = "vdb_compact_indexes")
    public static int compactIndexes(IsolateThread thread, CCharPointer sourcePathsJoined, CCharPointer targetPath) {
        if (sourcePathsJoined.isNull() || targetPath.isNull()) return -3;
        try {
            String javaSourcePathsJoined = CTypeConversion.toJavaString(sourcePathsJoined);
            String javaTargetPath = CTypeConversion.toJavaString(targetPath);
            VectorDb.compactIndexes(javaSourcePathsJoined, javaTargetPath);
            return 0;
        } catch (IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }
}
