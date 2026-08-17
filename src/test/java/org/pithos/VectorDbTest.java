package org.pithos;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VectorDbTest {

    @Test
    void testJacobiSvdAndEnergy() {
        int D = 4;
        int D0 = 4;
        // Simple diagonal weight matrix W
        float[] weights = {
            2.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.5f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f
        };
        float[] phi = TransformOperator.computeCumulativeEnergy(weights, D, D0);
        
        // Sum of squared singular values = 4 + 1 + 0.25 = 5.25
        // Cumulative energies:
        // phi[0] = 4 / 5.25 = 0.7619
        // phi[1] = (4 + 1) / 5.25 = 0.9523
        // phi[2] = (4 + 1 + 0.25) / 5.25 = 1.0
        // phi[3] = 1.0
        
        assertEquals(0.7619f, phi[0], 1e-3f);
        assertEquals(0.9523f, phi[1], 1e-3f);
        assertEquals(1.0f, phi[2], 1e-3f);
        assertEquals(1.0f, phi[3], 1e-3f);
    }

    @Test
    void testTransformOperator() {
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] x = new float[128];
        java.util.Arrays.fill(x, 1.0f);
        long[] packed = transformer.transformAndQuantize(x);
        
        assertNotNull(packed);
        assertEquals(2, packed.length);
    }

    @Test
    void testKroneckerTransformOperator() {
        int D = 12;
        int[] tiers = {12}; // Block size 12 -> non-power of two. Triggers Kronecker fallback.
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] x = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f};
        long[] packed = transformer.transformAndQuantize(x);
        
        assertNotNull(packed);
        assertEquals(1, packed.length);
    }

    @Test
    void testCompileAndQueryIndex(@TempDir Path tempDir) throws IOException {
        Path dbPath = tempDir.resolve("test_pithos");
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        // Define target transformed vectors z with positive MSB (z[127] >= 0.0) to bypass QEG
        // MSB in flat index logic corresponds to bit 63 of the first tier (index 63 of transformed vector z)
        float[] targetZ0 = new float[128];
        java.util.Arrays.fill(targetZ0, -1.0f);
        targetZ0[63] = 1.0f; // Enable QEG bit

        float[] targetZ1 = new float[128];
        java.util.Arrays.fill(targetZ1, 1.0f); // All positive -> closest to query

        float[] targetZ2 = new float[128];
        java.util.Arrays.fill(targetZ2, -0.2f);
        targetZ2[63] = 1.0f; // Enable QEG bit

        // Back-project target transform vectors to raw input vectors x
        float[] vec0 = transformer.backProject(targetZ0);
        float[] vec1 = transformer.backProject(targetZ1);
        float[] vec2 = transformer.backProject(targetZ2);

        List<VectorRecord> records = List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1),
            new VectorRecord(2, vec2)
        );

        // Compile index
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1737400L, D, tiers, records);

        // Load index without weights (equal energy distribution)
        VectorDb db = new VectorDb();
        Index index = db.loadIndex("pithos_test", dbPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(3, index.size());
        assertEquals(128, index.getDimension());
        assertEquals(1, index.getPlanetId());
        assertEquals(1737400L, index.getPlanetRadius());
        assertEquals(2, index.getTierCount());

        // Perform float-based KNN search using a query that is all-positive in transformed domain
        float[] targetQueryZ = new float[128];
        java.util.Arrays.fill(targetQueryZ, 0.9f);
        float[] query = transformer.backProject(targetQueryZ);

        List<Index.SearchResult> results = index.search(query, 2);
        
        assertEquals(2, results.size());
        // Closest should be vec1 (ID 1) because targetZ1 has all positive entries (aligned with query)
        assertEquals(1, results.get(0).id());

        // Clean up
        db.close();
    }

    @Test
    void testCompileAndQuery2BitIndex(@TempDir Path tempDir) throws IOException {
        Path dbPath = tempDir.resolve("test_pithos_2bit");
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        // We want to verify that 2-bit mode successfully indexes and filters query noise
        float[] vec0 = new float[128];
        java.util.Arrays.fill(vec0, 0.5f);
        // Ensure MSB is positive for QEG
        float[] targetZ0 = transformer.preconditionAndRotate(vec0);
        targetZ0[63] = 1.0f;
        vec0 = transformer.backProject(targetZ0);

        float[] vec1 = new float[128];
        java.util.Arrays.fill(vec1, -0.5f);
        float[] targetZ1 = transformer.preconditionAndRotate(vec1);
        targetZ1[63] = 1.0f;
        vec1 = transformer.backProject(targetZ1);

        List<VectorRecord> records = List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1)
        );

        // Compile index with qMode = 1 (2-bit)
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1737400L, D, tiers, records, 1);

        // Load index
        VectorDb db = new VectorDb();
        Index index = db.loadIndex("pithos_2bit_test", dbPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(2, index.size());

        // Search with query vec0
        List<Index.SearchResult> results = index.search(vec0, 2);
        assertEquals(2, results.size());
        assertEquals(0, results.get(0).id()); // Closest should be ID 0

        db.close();
    }

    @Test
    void testCompaction(@TempDir Path tempDir) throws IOException {
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] vec0 = new float[D];
        java.util.Arrays.fill(vec0, 0.5f);
        float[] targetZ0 = transformer.preconditionAndRotate(vec0);
        targetZ0[63] = 1.0f;
        vec0 = transformer.backProject(targetZ0);

        float[] vec1 = new float[D];
        java.util.Arrays.fill(vec1, -0.5f);
        float[] targetZ1 = transformer.preconditionAndRotate(vec1);
        targetZ1[63] = 1.0f;
        vec1 = transformer.backProject(targetZ1);

        Path dbPath1 = tempDir.resolve("db1");
        VectorDb.compileIndexFile(dbPath1.toString(), (byte) 1, 1000L, D, tiers, List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1)
        ));

        float[] vec2 = new float[D];
        java.util.Arrays.fill(vec2, 0.2f);
        float[] targetZ2 = transformer.preconditionAndRotate(vec2);
        targetZ2[63] = 1.0f;
        vec2 = transformer.backProject(targetZ2);

        float[] vec3 = new float[D];
        java.util.Arrays.fill(vec3, -0.2f);
        float[] targetZ3 = transformer.preconditionAndRotate(vec3);
        targetZ3[63] = 1.0f;
        vec3 = transformer.backProject(targetZ3);

        Path dbPath2 = tempDir.resolve("db2");
        VectorDb.compileIndexFile(dbPath2.toString(), (byte) 1, 1000L, D, tiers, List.of(
            new VectorRecord(2, vec2),
            new VectorRecord(3, vec3)
        ));

        Path dbCompactedPath = tempDir.resolve("db_compacted");
        String sourcePaths = dbPath1.toString() + ";" + dbPath2.toString();
        VectorDb.compactIndexes(sourcePaths, dbCompactedPath.toString());

        VectorDb db = new VectorDb();
        Index index = db.loadIndex("compacted", dbCompactedPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(4, index.size());
        assertEquals(D, index.getDimension());
        assertEquals(2, index.getTierCount());

        List<Index.SearchResult> results = index.search(vec0, 4);
        assertEquals(4, results.size());
        assertEquals(0, results.get(0).id());

        db.close();
    }

    @Test
    void testWalRecovery(@TempDir Path tempDir) throws IOException {
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] vec0 = new float[D];
        java.util.Arrays.fill(vec0, 0.5f);
        float[] vec1 = new float[D];
        java.util.Arrays.fill(vec1, -0.5f);

        Path dbPath = tempDir.resolve("db_wal");
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1000L, D, tiers, List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1)
        ));

        // 1. Initialize and write to delta buffer (which writes to WAL)
        VectorDb db = new VectorDb();
        db.loadIndex("idx_wal", dbPath.toString(), null, 0);
        DeltaBuffer delta = db.createDeltaBuffer("idx_wal", 100);

        float[] insertVec0 = new float[D];
        insertVec0[0] = 99.0f;
        float[] insertVec1 = new float[D];
        insertVec1[0] = 88.0f;
        float[] insertVec2 = new float[D];
        insertVec2[0] = 77.0f;

        delta.insert(100, insertVec0);
        delta.insert(200, insertVec1);
        delta.insert(300, insertVec2);

        // Delete one entry
        delta.delete(200);

        assertEquals(2, delta.liveSize());
        assertEquals(3, delta.totalSize());

        Path walFile = tempDir.resolve("db_wal_wal.bin");
        assertTrue(Files.exists(walFile));
        assertTrue(Files.size(walFile) > 0);

        db.close(); // Closes delta and releases WAL file lock

        // 2. Re-load index and verify WAL recovery
        VectorDb db2 = new VectorDb();
        db2.loadIndex("idx_wal", dbPath.toString(), null, 0);
        DeltaBuffer delta2 = db2.createDeltaBuffer("idx_wal", 100);

        assertEquals(2, delta2.liveSize());
        assertEquals(3, delta2.totalSize());

        // Drain the delta buffer (should truncate WAL)
        List<VectorRecord> drained = delta2.drainLiveEntries();
        assertEquals(2, drained.size());
        assertEquals(0, delta2.liveSize());

        // Verify WAL is truncated to size 0
        assertEquals(0, Files.size(walFile));

        db2.close();
    }

    @Test
    void testOptionalFp16(@TempDir Path tempDir) throws IOException {
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] vec0 = new float[D];
        java.util.Arrays.fill(vec0, 0.5f);
        float[] targetZ0 = transformer.preconditionAndRotate(vec0);
        targetZ0[63] = 1.0f;
        vec0 = transformer.backProject(targetZ0);

        float[] vec1 = new float[D];
        java.util.Arrays.fill(vec1, -0.5f);
        float[] targetZ1 = transformer.preconditionAndRotate(vec1);
        targetZ1[63] = 1.0f;
        vec1 = transformer.backProject(targetZ1);

        Path dbPath = tempDir.resolve("db_no_fp16");
        
        // Compile without writing FP16 sidecar
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1000L, D, tiers, List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1)
        ), 0, false);

        // Verify FP16 sidecar file does NOT exist
        Path fp16Path = tempDir.resolve("db_no_fp16_fp16.bin");
        assertFalse(Files.exists(fp16Path));

        // Load index without FP16 sidecar
        VectorDb db = new VectorDb();
        Index index = db.loadIndex("no_fp16", dbPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(2, index.size());

        // Perform search
        List<Index.SearchResult> results = index.search(vec0, 2);
        assertEquals(2, results.size());
        assertEquals(0, results.get(0).id()); // Closest should still be ID 0 (asymmetric L2 fallback works!)

        db.close();
    }

    @Test
    void testFP8EncodingDecoding() {
        float[] testValues = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.001953125f, 448.0f, -448.0f, 500.0f, -600.0f};
        for (float val : testValues) {
            byte encoded = VectorDb.encodeFP8_E4M3(val);
            float decoded = VectorDb.decodeFP8_E4M3(encoded);
            if (val == 0.0f) {
                assertEquals(0.0f, decoded, 1e-6f);
            } else if (Math.abs(val) > 448.0f) {
                assertEquals(Math.signum(val) * 448.0f, decoded, 1e-3f);
            } else {
                float relErr = Math.abs(val - decoded) / Math.abs(val);
                assertTrue(relErr < 0.15f, "Rel error too high for " + val + ": decoded=" + decoded);
            }
        }
    }

    @Test
    void testFP8LookupTableExactness() {
        for (int i = 0; i < 256; i++) {
            byte b = (byte) i;
            float fromMethod = VectorDb.decodeFP8_E4M3(b);
            float fromLut = FlatIndex.FP8_E4M3_LUT[i];
            assertEquals(fromMethod, fromLut, 1e-6f);
        }
    }

    @Test
    void testFP4EncodingDecoding() {
        for (int nibble = 0; nibble < 16; nibble++) {
            float decoded = VectorDb.decodeFP4_E2M1_Nibble(nibble);
            byte reEncoded = VectorDb.encodeFP4_E2M1_Nibble(decoded);
            assertEquals(nibble, reEncoded & 0x0F);
        }
    }

    @Test
    void testGeodeticMortonEncoding() {
        long mortonEquator = VectorDb.encodeGeodeticMorton(0.0, 0.0);
        long mortonNorthPole = VectorDb.encodeGeodeticMorton(90.0, 0.0);
        long mortonSouthPole = VectorDb.encodeGeodeticMorton(-90.0, 0.0);

        assertTrue(mortonEquator > 0L);
        assertTrue(mortonNorthPole > 0L);
        assertNotEquals(mortonEquator, mortonNorthPole);
        assertNotEquals(mortonEquator, mortonSouthPole);
    }

    @Test
    void testCompileAndQueryFP8SidecarIndex(@TempDir Path tempDir) throws IOException {
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] vec0 = new float[D];
        java.util.Arrays.fill(vec0, 0.5f);
        float[] targetZ0 = transformer.preconditionAndRotate(vec0);
        targetZ0[63] = 1.0f;
        vec0 = transformer.backProject(targetZ0);

        float[] vec1 = new float[D];
        java.util.Arrays.fill(vec1, -0.5f);
        float[] targetZ1 = transformer.preconditionAndRotate(vec1);
        targetZ1[63] = 1.0f;
        vec1 = transformer.backProject(targetZ1);

        Path dbPath = tempDir.resolve("db_fp8");
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1000L, D, tiers, List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1)
        ), 0, VectorDb.SIDECAR_FP8);

        // Verify _fp8.bin exists and has size 2 * 128 * 1 = 256 bytes
        Path fp8Path = tempDir.resolve("db_fp8_fp8.bin");
        assertTrue(Files.exists(fp8Path));
        assertEquals(2L * D * 1L, Files.size(fp8Path));

        VectorDb db = new VectorDb();
        Index index = db.loadIndex("fp8_idx", dbPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(2, index.size());
        assertEquals(VectorDb.SIDECAR_FP8, index.getSidecarMode());

        List<Index.SearchResult> results = index.search(vec0, 2);
        assertEquals(2, results.size());
        assertEquals(0, results.get(0).id()); // Nearest neighbor is ID 0

        db.close();
    }

    @Test
    void testCompileAndQueryFP4SidecarIndex(@TempDir Path tempDir) throws IOException {
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] vec0 = new float[D];
        java.util.Arrays.fill(vec0, 0.5f);
        float[] targetZ0 = transformer.preconditionAndRotate(vec0);
        targetZ0[63] = 1.0f;
        vec0 = transformer.backProject(targetZ0);

        float[] vec1 = new float[D];
        java.util.Arrays.fill(vec1, -0.5f);
        float[] targetZ1 = transformer.preconditionAndRotate(vec1);
        targetZ1[63] = 1.0f;
        vec1 = transformer.backProject(targetZ1);

        Path dbPath = tempDir.resolve("db_fp4");
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1000L, D, tiers, List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1)
        ), 0, VectorDb.SIDECAR_FP4);

        // Verify _fp4.bin exists and has size 2 * (128/16 * 9) = 2 * 72 = 144 bytes
        Path fp4Path = tempDir.resolve("db_fp4_fp4.bin");
        assertTrue(Files.exists(fp4Path));
        int numBlocks = (D + 15) / 16;
        assertEquals(2L * numBlocks * 9L, Files.size(fp4Path));

        VectorDb db = new VectorDb();
        Index index = db.loadIndex("fp4_idx", dbPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(2, index.size());
        assertEquals(VectorDb.SIDECAR_FP4, index.getSidecarMode());

        List<Index.SearchResult> results = index.search(vec0, 2);
        assertEquals(2, results.size());
        assertEquals(0, results.get(0).id());

        db.close();
    }

    @Test
    void testFP8Compaction(@TempDir Path tempDir) throws IOException {
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] vec0 = new float[D];
        java.util.Arrays.fill(vec0, 0.5f);
        float[] targetZ0 = transformer.preconditionAndRotate(vec0);
        targetZ0[63] = 1.0f;
        vec0 = transformer.backProject(targetZ0);

        float[] vec1 = new float[D];
        java.util.Arrays.fill(vec1, -0.5f);
        float[] targetZ1 = transformer.preconditionAndRotate(vec1);
        targetZ1[63] = 1.0f;
        vec1 = transformer.backProject(targetZ1);

        Path dbPath1 = tempDir.resolve("fp8_db1");
        VectorDb.compileIndexFile(dbPath1.toString(), (byte) 1, 1000L, D, tiers, List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1)
        ), 0, VectorDb.SIDECAR_FP8);

        float[] vec2 = new float[D];
        java.util.Arrays.fill(vec2, 0.2f);
        float[] targetZ2 = transformer.preconditionAndRotate(vec2);
        targetZ2[63] = 1.0f;
        vec2 = transformer.backProject(targetZ2);

        float[] vec3 = new float[D];
        java.util.Arrays.fill(vec3, -0.2f);
        float[] targetZ3 = transformer.preconditionAndRotate(vec3);
        targetZ3[63] = 1.0f;
        vec3 = transformer.backProject(targetZ3);

        Path dbPath2 = tempDir.resolve("fp8_db2");
        VectorDb.compileIndexFile(dbPath2.toString(), (byte) 1, 1000L, D, tiers, List.of(
            new VectorRecord(2, vec2),
            new VectorRecord(3, vec3)
        ), 0, VectorDb.SIDECAR_FP8);

        Path dbCompactedPath = tempDir.resolve("fp8_db_compacted");
        String sourcePaths = dbPath1.toString() + ";" + dbPath2.toString();
        VectorDb.compactIndexes(sourcePaths, dbCompactedPath.toString());

        Path fp8CompactedPath = tempDir.resolve("fp8_db_compacted_fp8.bin");
        assertTrue(Files.exists(fp8CompactedPath));
        assertEquals(4L * D * 1L, Files.size(fp8CompactedPath));

        VectorDb db = new VectorDb();
        Index index = db.loadIndex("fp8_compacted", dbCompactedPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(4, index.size());
        assertEquals(VectorDb.SIDECAR_FP8, index.getSidecarMode());

        List<Index.SearchResult> results = index.search(vec0, 4);
        assertEquals(4, results.size());
        assertEquals(0, results.get(0).id());

        db.close();
    }
}
