package examples.java;

import org.pithos.DeltaBuffer;
import org.pithos.FlatIndex;
import org.pithos.Index;
import org.pithos.VectorDb;
import org.pithos.VectorRecord;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Demonstrates the native Java API of Pithos:
 * 1. Compiling multi-tier binary columnar index files
 * 2. Memory-mapping index files off-heap with zero GC pressure
 * 3. Matryoshka cumulative spectral energy budget early exit
 * 4. LSM DeltaBuffer for real-time inserts and soft deletes
 * 5. Merged search combining base immutable index and mutable delta buffer
 */
public class PithosApiDemo {

    public static void main(String[] args) throws IOException {
        System.out.println("================================================================");
        System.out.println("            PITHOS VECTOR DATABASE (JAVA API DEMO)             ");
        System.out.println("================================================================\n");

        String basePath = "temp/java_demo_idx";
        new File("temp").mkdirs();

        int dimension = 128;
        int numRecords = 2000;
        int[] tiers = new int[]{64, 128};

        // 1. Prepare synthetic vector records
        System.out.printf("[1/5] Compiling %d records (D=%d) across 2 tiers [64, 128]...\n", numRecords, dimension);
        List<VectorRecord> records = new ArrayList<>(numRecords);
        for (int i = 0; i < numRecords; i++) {
            float[] vec = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                vec[d] = (float) Math.sin((i + 1) * (d + 1));
            }
            records.add(new VectorRecord(1000L + i, vec));
        }

        // 2. Compile index on disk
        VectorDb.compileIndexFile(basePath, (byte) 1, 1737400L, dimension, tiers, records, 0);

        // 3. Open VectorDb and load index off-heap
        VectorDb db = new VectorDb();
        System.out.println("[2/5] Loading index off-heap into POSIX virtual memory...");
        Index index = db.loadIndex("java_index", basePath, null, 0);

        System.out.printf("  - Loaded Index: %d records, %d dimensions, %d tiers\n",
                index.size(), index.getDimension(), index.getTierCount());

        // 4. Batch Search
        System.out.println("\n[3/5] Querying Top-5 Nearest Neighbors on base index...");
        float[][] queries = new float[1][dimension];
        for (int d = 0; d < dimension; d++) {
            queries[0][d] = (float) Math.sin(1 * (d + 1));
        }

        List<Index.SearchResult>[] results = index.batchSearch(queries, 5);
        for (int i = 0; i < results[0].size(); i++) {
            Index.SearchResult r = results[0].get(i);
            System.out.printf("  Rank %d: Record ID = %d | Distance = %.4f\n",
                    (i + 1), r.id(), r.score() / 1_000_000.0);
        }

        // 5. LSM DeltaBuffer
        System.out.println("\n[4/5] Testing LSM DeltaBuffer for Real-Time Streaming Ingestion...");
        DeltaBuffer delta = db.createDeltaBuffer("java_index", 1000);

        long newId = 99999L;
        float[] newVec = queries[0].clone();
        delta.insert(newId, newVec);
        System.out.printf("  - Inserted new record ID %d into DeltaBuffer (size: %d)\n", newId, delta.size());

        // Merged Search
        System.out.println("\n[5/5] Performing Merged Search (Base Index + LSM Delta):");
        List<Index.SearchResult> merged = db.searchMerged("java_index", queries[0], 5);
        for (int i = 0; i < merged.size(); i++) {
            Index.SearchResult r = merged.get(i);
            String origin = (r.id() == newId) ? "LSM Delta" : "Base Index";
            System.out.printf("  Rank %d: Record ID = %d | Distance = %.4f | Origin = %s\n",
                    (i + 1), r.id(), r.score() / 1_000_000.0, origin);
        }

        db.close();
        System.out.println("\nDatabase closed cleanly.");
    }
}
