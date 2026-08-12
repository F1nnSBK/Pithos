package org.pithos;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Experiment {

    @Test
    void runRecallExperiment(@TempDir Path tempDir) throws IOException {
        System.out.println("=== RUNNING RECALL EXPERIMENT ===");
        Path dbPath = tempDir.resolve("experiment_db");
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        // Generate 1000 completely random vectors
        Random rand = new Random(42);
        List<VectorRecord> records = new ArrayList<>();
        for (long i = 0; i < 1000; i++) {
            float[] vec = new float[D];
            for (int j = 0; j < D; j++) {
                vec[j] = rand.nextFloat() * 2.0f - 1.0f;
            }
            records.add(new VectorRecord(i, vec));
        }

        // Compile index (qMode = 0 -> 1-bit)
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1000L, D, tiers, records);

        // Load index
        VectorDb db = new VectorDb();
        Index index = db.loadIndex("exp_test", dbPath.toString(), null, 0);

        // Create 1 query vector (also random)
        float[][] queries = new float[1][D];
        for (int j = 0; j < D; j++) {
            queries[0][j] = rand.nextFloat() * 2.0f - 1.0f;
        }

        int[] families = {0};
        int[] thresholds = {64}; // loose threshold

        // Allocate memory segment for voting mask
        java.lang.foreign.MemorySegment votingMask = java.lang.foreign.Arena.ofAuto().allocate(1000);

        // Run Voting Search
        index.queryPlanetaryGrid(queries, families, thresholds, votingMask);
        int candidatesFound = 0;
        for (long i = 0; i < 1000; i++) {
            if (votingMask.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i) != 0) {
                candidatesFound++;
            }
        }
        System.out.println("CANDIDATES FOUND: " + candidatesFound);
        
        db.close();
    }
}
