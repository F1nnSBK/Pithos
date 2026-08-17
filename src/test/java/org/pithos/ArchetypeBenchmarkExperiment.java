package org.pithos;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/// # ArchetypeBenchmarkExperiment
///
/// Comprehensive empirical benchmark of Pithos (MIDB) v1.1.0 using:
/// 1. **278 Real Target Archetype Embeddings** (True Positives, DINOv3 ViT-S/16 + LoRA).
/// 2. **150 Real Mined Hard Negatives** (High-confusion distractors).
/// 3. **9,572 Background Surface Distractors**.
///
/// Total Dataset Size: 10,000 High-Dimensional Vectors (D = 384).
public class ArchetypeBenchmarkExperiment {

    private static final int TARGET_COUNT = 278;
    private static final int HARD_NEG_COUNT = 150;
    private static final int DIMENSION = 384;
    private static final int[] TIERS = {64, 128, 256, 384};

    /// Loads 384-dimensional float embeddings from a raw binary file.
    private float[][] loadFloatBinary(String path, int count) throws IOException {
        File binFile = new File(path);
        if (!binFile.exists()) {
            throw new IllegalStateException("Binary file not found at: " + binFile.getAbsolutePath());
        }

        byte[] bytes = new byte[count * DIMENSION * Float.BYTES];
        try (FileInputStream fis = new FileInputStream(binFile)) {
            int read = fis.read(bytes);
            if (read != bytes.length) {
                throw new IOException("Expected " + bytes.length + " bytes, but read " + read);
            }
        }

        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[][] matrix = new float[count][DIMENSION];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < DIMENSION; d++) {
                matrix[i][d] = buf.getFloat();
            }
        }
        return matrix;
    }

    @Test
    void runArchetypeAndHardNegativesBenchmark(@TempDir Path tempDir) throws IOException {
        System.out.println("\n" + "=".repeat(100));
        System.out.println("  PITHOS v1.1.0 BENCHMARK: REAL ARCHETYPES + MINED HARD NEGATIVES");
        System.out.println("=".repeat(100));

        // 1. Load 278 True Archetype Embeddings
        float[][] realArchetypes = loadFloatBinary("src/test/resources/archetypes/archetype_vectors_384.bin", TARGET_COUNT);
        System.out.printf("[Dataset] Loaded %d real True-Positive Target Archetypes (dim=%d).%n",
                realArchetypes.length, DIMENSION);

        // 2. Load 150 Real Mined Hard Negatives
        float[][] hardNegatives = loadFloatBinary("src/test/resources/archetypes/hard_negatives_384.bin", HARD_NEG_COUNT);
        System.out.printf("[Dataset] Loaded %d real Mined Hard Negatives.%n",
                hardNegatives.length);

        // 3. Assemble Dataset of 10,000 Records:
        //    - Records 0 .. 277      : True Targets (ID: 1 .. 278)
        //    - Records 278 .. 427    : Hard Negatives (ID: 279 .. 428)
        //    - Records 428 .. 9,999  : Background Distractors (ID: 429 .. 10,000)
        int totalRecords = 10000;
        int backgroundCount = totalRecords - TARGET_COUNT - HARD_NEG_COUNT;
        Random rand = new Random(42);

        List<VectorRecord> dataset = new ArrayList<>(totalRecords);
        float[][] allVectors = new float[totalRecords][DIMENSION];

        // Add True Targets
        for (int i = 0; i < TARGET_COUNT; i++) {
            long id = i + 1;
            allVectors[i] = Arrays.copyOf(realArchetypes[i], DIMENSION);
            dataset.add(new VectorRecord(id, allVectors[i]));
        }

        // Add Hard Negatives
        for (int i = 0; i < HARD_NEG_COUNT; i++) {
            int rowIdx = TARGET_COUNT + i;
            long id = rowIdx + 1;
            allVectors[rowIdx] = Arrays.copyOf(hardNegatives[i], DIMENSION);
            dataset.add(new VectorRecord(id, allVectors[rowIdx]));
        }

        // Add Background Distractors
        for (int i = 0; i < backgroundCount; i++) {
            int rowIdx = TARGET_COUNT + HARD_NEG_COUNT + i;
            long id = rowIdx + 1;
            float[] bg = new float[DIMENSION];
            float normSq = 0.0f;
            for (int d = 0; d < DIMENSION; d++) {
                bg[d] = (float) rand.nextGaussian();
                normSq += bg[d] * bg[d];
            }
            float norm = (float) Math.sqrt(normSq);
            for (int d = 0; d < DIMENSION; d++) bg[d] /= norm;

            allVectors[rowIdx] = bg;
            dataset.add(new VectorRecord(id, bg));
        }

        System.out.printf("[Index Assembly] Total: %d records (%d True Targets, %d Hard Negatives, %d General Distractors).%n",
                totalRecords, TARGET_COUNT, HARD_NEG_COUNT, backgroundCount);

        // 50 Real Target Archetype Queries
        int numQueries = 50;
        float[][] targetQueries = new float[numQueries][DIMENSION];
        for (int q = 0; q < numQueries; q++) {
            int targetIdx = q * (TARGET_COUNT / numQueries);
            float[] orig = realArchetypes[targetIdx];
            float[] noisy = new float[DIMENSION];
            float normSq = 0.0f;
            for (int d = 0; d < DIMENSION; d++) {
                noisy[d] = orig[d] + (float) (rand.nextGaussian() * 0.04);
                normSq += noisy[d] * noisy[d];
            }
            float norm = (float) Math.sqrt(normSq);
            for (int d = 0; d < DIMENSION; d++) targetQueries[q][d] = noisy[d] / norm;
        }

        // -------------------------------------------------------------------------------------------------
        // Compute Exact Float32 Ground Truth
        // -------------------------------------------------------------------------------------------------
        int k = 10;
        List<Set<Long>> exactGt = new ArrayList<>();
        for (int q = 0; q < numQueries; q++) {
            float[] qVec = targetQueries[q];
            List<long[]> candidates = new ArrayList<>();
            for (int i = 0; i < totalRecords; i++) {
                float dist = 0.0f;
                for (int d = 0; d < DIMENSION; d++) {
                    float diff = qVec[d] - allVectors[i][d];
                    dist += diff * diff;
                }
                candidates.add(new long[]{i + 1, Float.floatToIntBits(dist)});
            }
            candidates.sort(Comparator.comparingDouble(c -> Float.intBitsToFloat((int) c[1])));
            Set<Long> gtIds = new HashSet<>();
            for (int i = 0; i < k; i++) {
                gtIds.add(candidates.get(i)[0]);
            }
            exactGt.add(gtIds);
        }

        // -------------------------------------------------------------------------------------------------
        // EXPERIMENT 1: Top-k Retrieval on True Targets with Hard Negatives Present
        // -------------------------------------------------------------------------------------------------
        int[] modes = {
                VectorDb.SIDECAR_NONE,
                VectorDb.SIDECAR_FP16,
                VectorDb.SIDECAR_FP8,
                VectorDb.SIDECAR_FP4
        };
        String[] labels = {
                "Bit-Only (No Sidecar)",
                "FP16 Sidecar (2.0 B/dim)",
                "FP8 Sidecar (1.0 B/dim)",
                "NVFP4 Sidecar (0.56 B/dim)"
        };

        System.out.println("\n" + "-".repeat(100));
        System.out.printf("%-26s | %-10s | %-10s | %-11s | %-11s | %-12s%n",
                "Format Mode", "Bytes/Vec", "SSD (MB)", "Recall@1", "Recall@10", "Latency (µs)");
        System.out.println("-".repeat(100));

        for (int m = 0; m < modes.length; m++) {
            int mode = modes[m];
            String label = labels[m];
            Path dbPath = tempDir.resolve("bench_hn_idx_" + mode);

            VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1737400L, DIMENSION, TIERS, dataset, 0, mode);

            long totalBytes = calculateDirectoryIndexBytes(dbPath.toString());
            double mbSize = totalBytes / (1024.0 * 1024.0);
            double bPerVec = (double) totalBytes / totalRecords;

            VectorDb db = new VectorDb();
            Index index = db.loadIndex("bench_idx_" + mode, dbPath.toString(), null, 0);

            index.batchSearch(targetQueries, 1);

            long t0 = System.nanoTime();
            List<Index.SearchResult>[] results = index.batchSearch(targetQueries, k);
            long totalTimeNs = System.nanoTime() - t0;
            double latUs = (totalTimeNs / (double) numQueries) / 1000.0;

            double recall1Sum = 0.0;
            double recall10Sum = 0.0;
            for (int q = 0; q < numQueries; q++) {
                Set<Long> gt = exactGt.get(q);
                List<Index.SearchResult> pred = results[q];
                if (!pred.isEmpty() && gt.contains(pred.get(0).id())) {
                    recall1Sum += 1.0;
                }
                int overlap = 0;
                for (Index.SearchResult r : pred) {
                    if (gt.contains(r.id())) overlap++;
                }
                recall10Sum += (overlap / (double) k);
            }

            double r1 = (recall1Sum / numQueries) * 100.0;
            double r10 = (recall10Sum / numQueries) * 100.0;

            System.out.printf("%-26s | %9.1f B | %8.2f MB | %9.2f %% | %9.2f %% | %10.1f µs%n",
                    label, bPerVec, mbSize, r1, r10, latUs);

            db.close();
        }
        System.out.println("-".repeat(100));

        // -------------------------------------------------------------------------------------------------
        // EXPERIMENT 2: Multi-Anchor Resonant Screening (278 Target Anchors)
        // -------------------------------------------------------------------------------------------------
        System.out.println("\n" + "=".repeat(100));
        System.out.println("  EXPERIMENT 2: MULTI-ANCHOR RESONANT SCREENING (278 ANCHORS)");
        System.out.println("=".repeat(100));

        Path resDbPath = tempDir.resolve("bench_res_hn_idx");
        VectorDb.compileIndexFile(resDbPath.toString(), (byte) 1, 1737400L, DIMENSION, TIERS, dataset, 0, VectorDb.SIDECAR_FP8);

        VectorDb resDb = new VectorDb();
        Index resIndex = resDb.loadIndex("bench_res_hn", resDbPath.toString(), null, 0);

        int[] testThresholds = {30, 45, 60, 75, 90};
        System.out.printf("%-15s | %-16s | %-16s | %-16s | %-14s%n",
                "Threshold (bits)", "Sensitivity", "Hard Neg Rejection", "BG Selectivity", "Throughput");
        System.out.println("-".repeat(90));

        for (int thresh : testThresholds) {
            int[] families = new int[TARGET_COUNT];
            int[] thresholds = new int[TARGET_COUNT];
            for (int i = 0; i < TARGET_COUNT; i++) {
                families[i] = i % 8;
                thresholds[i] = thresh;
            }

            MemorySegment votingMask = Arena.ofAuto().allocate(totalRecords);
            long t0Res = System.nanoTime();
            long resonantCount = resIndex.queryPlanetaryGrid(realArchetypes, families, thresholds, votingMask);
            long resElapsedNs = System.nanoTime() - t0Res;
            double vectorsPerSec = (totalRecords / (resElapsedNs / 1e9));

            int trueTargetsFound = 0;
            int hardNegativesPassed = 0;
            int backgroundPassed = 0;

            for (int i = 0; i < totalRecords; i++) {
                byte mask = votingMask.get(ValueLayout.JAVA_BYTE, i);
                if (mask != 0) {
                    if (i < TARGET_COUNT) {
                        trueTargetsFound++;
                    } else if (i < TARGET_COUNT + HARD_NEG_COUNT) {
                        hardNegativesPassed++;
                    } else {
                        backgroundPassed++;
                    }
                }
            }

            double sensitivity = (trueTargetsFound / (double) TARGET_COUNT) * 100.0;
            double hardNegRejection = 100.0 - ((hardNegativesPassed / (double) HARD_NEG_COUNT) * 100.0);
            double bgSelectivity = 100.0 - ((backgroundPassed / (double) backgroundCount) * 100.0);

            System.out.printf("  %-13d | %13.2f %% | %15.2f %% | %13.2f %% | %,10.0f V/s%n",
                    thresh, sensitivity, hardNegRejection, bgSelectivity, vectorsPerSec);
        }
        System.out.println("=".repeat(100) + "\n");
        resDb.close();
    }

    private static long calculateDirectoryIndexBytes(String basePath) {
        long size = 0;
        String[] exts = {"", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin", "_tier_2.bin", "_tier_3.bin", "_fp16.bin", "_fp8.bin", "_fp4.bin"};
        for (String ext : exts) {
            File f = new File(basePath + ext);
            if (f.exists()) size += f.length();
        }
        return size;
    }
}
