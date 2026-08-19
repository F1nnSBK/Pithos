package org.pithos;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class PithosContainerTest {

    @Test
    void testSingleFileContainerRoundtrip(@TempDir Path tempDir) throws IOException {
        Path containerPath = tempDir.resolve("test_collection.pithos");
        int dimension = 128;
        int[] tiers = { 64, 128 };
        int numRecords = 500;

        Random rng = new Random(42);
        List<VectorRecord> records = new ArrayList<>(numRecords);
        for (int i = 0; i < numRecords; i++) {
            float[] vec = new float[dimension];
            float norm = 0.0f;
            for (int d = 0; d < dimension; d++) {
                vec[d] = rng.nextFloat() * 2.0f - 1.0f;
                norm += vec[d] * vec[d];
            }
            norm = (float) Math.sqrt(norm);
            for (int d = 0; d < dimension; d++) {
                vec[d] /= norm;
            }
            records.add(new VectorRecord(1000L + i, vec));
        }

        String userMetadata = "{\"collection\": \"cats_vs_dogs\", \"encoder\": \"dinov3\", \"version\": 1}";
        byte[] payload = "arbitrary_binary_metadata_payload_12345".getBytes();

        // 1. Write single-file container
        PithosContainer.writeContainer(
                containerPath,
                dimension,
                tiers,
                records,
                PithosContainer.METRIC_COSINE,
                0, // 1-bit
                VectorDb.SIDECAR_FP8,
                payload,
                "raw",
                userMetadata);

        assertTrue(Files.exists(containerPath));
        assertTrue(PithosContainer.isPithosContainer(containerPath));

        // 2. Load via FlatIndex.mapFile
        FlatIndex index = FlatIndex.mapFile(containerPath.toString(), null, 0);
        assertNotNull(index);
        assertEquals(dimension, index.getDimension());
        assertEquals(numRecords, index.size());
        assertTrue(index.isSingleFileContainer());
        assertNotNull(index.getUserMetadataJson());
        assertTrue(index.getUserMetadataJson().contains("cats_vs_dogs"));
        assertNotNull(index.getMetadataPayloadSegment());
        assertEquals(payload.length, index.getMetadataPayloadSegment().byteSize());

        // 3. Search single-file container
        float[] query = records.get(0).vector();
        Index.SearchResult result = index.search(query, 5).get(0);
        assertNotNull(result);
        assertEquals(1000L, result.id()); // Exact match for top 1

        index.close();
    }

    @Test
    void testSingleFileContainerFP16And2BitQJL(@TempDir Path tempDir) throws IOException {
        Path containerPath = tempDir.resolve("qjl_fp16.pithos");
        int dimension = 64;
        int[] tiers = { 64 };
        int numRecords = 100;

        Random rng = new Random(1337);
        List<VectorRecord> records = new ArrayList<>(numRecords);
        for (int i = 0; i < numRecords; i++) {
            float[] vec = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                vec[d] = (float) rng.nextGaussian();
            }
            records.add(new VectorRecord(5000L + i, vec));
        }

        PithosContainer.writeContainer(
                containerPath,
                dimension,
                tiers,
                records,
                PithosContainer.METRIC_L2,
                1, // 2-bit QJL
                VectorDb.SIDECAR_FP16,
                null,
                null,
                "{\"metric\": \"l2\"}");

        assertTrue(PithosContainer.isPithosContainer(containerPath));

        FlatIndex index = FlatIndex.mapFile(containerPath.toString(), null, 0);
        assertNotNull(index);
        assertEquals(numRecords, index.size());
        assertEquals(VectorDb.SIDECAR_FP16, index.getSidecarMode());

        float[] query = records.get(10).vector();
        Index.SearchResult top = index.search(query, 1).get(0);
        assertEquals(5010L, top.id());

        index.close();
    }

    @Test
    void testGate0PrefixRoutingVerification(@TempDir Path tempDir) throws IOException {
        Path containerPath = tempDir.resolve("prefix_routing_test.pithos");
        int dimension = 128;
        int[] tiers = { 64, 128 };
        int numRecords = 2000;

        Random rng = new Random(999);
        List<VectorRecord> records = new ArrayList<>(numRecords);
        for (int i = 0; i < numRecords; i++) {
            float[] vec = new float[dimension];
            float norm = 0.0f;
            for (int d = 0; d < dimension; d++) {
                vec[d] = rng.nextFloat() * 2.0f - 1.0f;
                norm += vec[d] * vec[d];
            }
            norm = (float) Math.sqrt(norm);
            for (int d = 0; d < dimension; d++) {
                vec[d] /= norm;
            }
            records.add(new VectorRecord(10000L + i, vec));
        }

        // Write single-file container with prefix table
        PithosContainer.writeContainer(
                containerPath,
                dimension,
                tiers,
                records,
                PithosContainer.METRIC_COSINE,
                0, // 1-bit
                VectorDb.SIDECAR_FP8,
                null,
                null,
                "{\"test\": \"gate0_prefix_routing\"}");

        assertTrue(Files.exists(containerPath));

        // Map container
        FlatIndex index = FlatIndex.mapFile(containerPath.toString(), null, 0);
        assertNotNull(index);
        assertTrue(index.hasPrefixTable(), "Prefix table must be active");
        assertNotNull(index.getPrefixOffsetsSegment());
        assertNotNull(index.getPrefixPostingsSegment());
        assertEquals(PithosContainer.MIH_OFFSETS_BYTES, index.getPrefixOffsetsSegment().byteSize());
        assertEquals(numRecords * 4L * PithosContainer.NUM_MIH_CHUNKS, index.getPrefixPostingsSegment().byteSize());

        // Batch search across multiple queries
        float[][] queries = new float[10][dimension];
        for (int q = 0; q < 10; q++) {
            queries[q] = records.get(q * 50).vector();
        }

        List<Index.SearchResult>[] batchResults = index.batchSearch(queries, 5);
        assertEquals(10, batchResults.length);
        for (int q = 0; q < 10; q++) {
            assertNotNull(batchResults[q]);
            assertFalse(batchResults[q].isEmpty());
            assertEquals(10000L + (q * 50), batchResults[q].get(0).id(), "Exact top 1 match must be found");
        }

        index.close();
    }
}
