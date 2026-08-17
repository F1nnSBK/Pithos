package org.pithos;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class Experiment {

    @Test
    void runRecallBenchmark(@TempDir Path tempDir) throws IOException {
        System.out.println("================================================================================");
        System.out.println("  PITHOS JAVA NATIVE RECALL & ACCURACY BENCHMARK (N=10,000, D=128)");
        System.out.println("================================================================================");

        int N = 10000;
        int D = 384;
        int[] tiers = {64, 128, 256, 384};
        int k = 10;
        int numQueries = 50;

        Random rand = new Random(42);
        List<VectorRecord> records = new ArrayList<>(N);
        float[][] vectors = new float[N][D];

        for (int i = 0; i < N; i++) {
            float[] vec = new float[D];
            float normSq = 0.0f;
            for (int j = 0; j < D; j++) {
                vec[j] = (float) rand.nextGaussian();
                normSq += vec[j] * vec[j];
            }
            float norm = (float) Math.sqrt(normSq);
            for (int j = 0; j < D; j++) {
                vec[j] /= norm;
                vectors[i][j] = vec[j];
            }
            records.add(new VectorRecord(i + 1, vec));
        }

        // Generate normalized queries
        float[][] queries = new float[numQueries][D];
        for (int q = 0; q < numQueries; q++) {
            float normSq = 0.0f;
            for (int j = 0; j < D; j++) {
                queries[q][j] = (float) rand.nextGaussian();
                normSq += queries[q][j] * queries[q][j];
            }
            float norm = (float) Math.sqrt(normSq);
            for (int j = 0; j < D; j++) {
                queries[q][j] /= norm;
            }
        }

        // Exact Float32 Ground Truth
        System.out.println("[Ground Truth] Computing exact float32 KNN...");
        List<Set<Long>> groundTruth = new ArrayList<>();
        for (int q = 0; q < numQueries; q++) {
            float[] qVec = queries[q];
            List<int[]> topCandidates = new ArrayList<>();
            for (int i = 0; i < N; i++) {
                float dist = 0.0f;
                for (int d = 0; d < D; d++) {
                    float diff = qVec[d] - vectors[i][d];
                    dist += diff * diff;
                }
                topCandidates.add(new int[]{i + 1, Float.floatToIntBits(dist)});
            }
            topCandidates.sort((a, b) -> Float.compare(Float.intBitsToFloat(a[1]), Float.intBitsToFloat(b[1])));
            Set<Long> gtIds = new HashSet<>();
            for (int i = 0; i < k; i++) {
                gtIds.add((long) topCandidates.get(i)[0]);
            }
            groundTruth.add(gtIds);
        }

        int[] sidecarModes = {VectorDb.SIDECAR_NONE, VectorDb.SIDECAR_FP16, VectorDb.SIDECAR_FP8, VectorDb.SIDECAR_FP4};
        String[] labels = {"Bit-Only (No Sidecar)", "FP16 Sidecar (2.0 B/dim)", "FP8 Sidecar (1.0 B/dim)", "NVFP4 Sidecar (0.56 B/dim)"};

        for (int m = 0; m < sidecarModes.length; m++) {
            int mode = sidecarModes[m];
            String label = labels[m];
            Path dbPath = tempDir.resolve("idx_" + mode);

            VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1737400L, D, tiers, records, 0, mode);

            VectorDb db = new VectorDb();
            Index index = db.loadIndex("idx_" + mode, dbPath.toString(), null, 0);

            long t0 = System.nanoTime();
            List<Index.SearchResult>[] results = index.batchSearch(queries, k);
            long totalTimeNs = System.nanoTime() - t0;
            double latUs = (totalTimeNs / (double) numQueries) / 1000.0;

            double recall10Sum = 0.0;
            for (int q = 0; q < numQueries; q++) {
                Set<Long> gt = groundTruth.get(q);
                int matches = 0;
                for (Index.SearchResult r : results[q]) {
                    if (gt.contains(r.id())) {
                        matches++;
                    }
                }
                recall10Sum += (matches / (double) k);
            }
            double r10 = (recall10Sum / numQueries) * 100.0;

            System.out.printf("  %-28s -> Recall@10: %6.2f%% | Latency: %6.1f µs/query%n", label, r10, latUs);
            db.close();
        }
        System.out.println("================================================================================");
    }
}

