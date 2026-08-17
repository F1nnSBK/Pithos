#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "pithos.h"

// Helper function to calculate elapsed time in milliseconds
double get_elapsed_ms(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 + (double)(end.tv_nsec - start.tv_nsec) / 1000000.0;
}

// Helper function to calculate elapsed time in microseconds
double get_elapsed_us(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000000.0 + (double)(end.tv_nsec - start.tv_nsec) / 1000.0;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    graal_isolate_t *isolate = NULL;
    graal_isolatethread_t *thread = NULL;
    struct timespec start, end;
    
    double t_isolate_create = 0.0;
    double t_db_init = 0.0;
    double t_compile_index = 0.0;
    double t_load_index = 0.0;
    double t_knn_search = 0.0;
    double t_voting_search = 0.0;
    
    printf("[C Client] Creating GraalVM Isolate...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    if (graal_create_isolate(NULL, &isolate, &thread) != 0) {
        fprintf(stderr, "[C Client] Failed to create GraalVM isolate!\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_isolate_create = get_elapsed_ms(start, end);
    printf("[C Client] Isolate created in %.3f ms\n", t_isolate_create);

    printf("[C Client] Initializing Pithos Database...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    int status = vdb_init(thread);
    if (status != 0) {
        fprintf(stderr, "[C Client] Failed to initialize database (code: %d)!\n", status);
        graal_tear_down_isolate(thread);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_db_init = get_elapsed_ms(start, end);
    printf("[C Client] Database initialized in %.3f ms\n", t_db_init);

    // 1. Compile test database file
    int dimension = 64;
    int num_records = 3;
    long long ids[] = {0, 1, 2};
    int tiers[] = {32, 64};
    int num_tiers = 2;

    float *vectors = (float *)calloc(num_records * dimension, sizeof(float));
    // Record 0: all 0.0
    // Record 1: all 1.0
    for (int i = 0; i < dimension; i++) {
        vectors[1 * dimension + i] = 1.0f;
    }
    // Record 2: first half 1.0, second half 0.0
    for (int i = 0; i < 32; i++) {
        vectors[2 * dimension + i] = 1.0f;
    }

    printf("[C Client] Compiling test binary vector file 'pithos_test.bin'...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Domain ID = 1, Radius = 1737400 meters
    status = vdb_compile_index_file(thread, "pithos_test.bin", (char)1, 1737400LL, dimension, tiers, num_tiers, ids, vectors, num_records, 0);
    if (status != 0) {
        fprintf(stderr, "[C Client] Index compilation failed (code: %d)\n", status);
        free(vectors);
        graal_tear_down_isolate(thread);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_compile_index = get_elapsed_ms(start, end);
    printf("[C Client] Index compiled in %.3f ms\n", t_compile_index);
    free(vectors);

    // 2. Load the compiled index using off-heap memory mapping
    printf("[C Client] Loading memory-mapped index 'test_index' from 'pithos_test.bin'...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    status = vdb_load_index(thread, "test_index", "pithos_test.bin");
    if (status != 0) {
        fprintf(stderr, "[C Client] Failed to load index (code: %d)\n", status);
        graal_tear_down_isolate(thread);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_load_index = get_elapsed_ms(start, end);
    printf("[C Client] Mapped index loaded in %.3f ms\n", t_load_index);

    // Fetch and display database attributes
    int outDimension = 0;
    long long outSize = 0;
    char outPlanetId = 0;
    long long outPlanetRadius = 0;
    int outTiersCount = 0;
    status = vdb_get_info(thread, "test_index", &outDimension, &outSize, &outPlanetId, &outPlanetRadius, &outTiersCount);
    if (status != 0) {
        fprintf(stderr, "[C Client] Failed to get index info (code: %d)\n", status);
    } else {
        printf("\n========================================================================\n");
        printf("[C Client] Pithos Index Metadata Attributes:\n");
        printf("  - Dimension         : %d\n", outDimension);
        printf("  - Size (Records)    : %lld\n", outSize);
        printf("  - Domain ID         : %d\n", (int)outPlanetId);
        printf("  - Reference Radius  : %lld\n", outPlanetRadius);
        printf("  - Tiers Count       : %d\n", outTiersCount);
        printf("========================================================================\n\n");
    }

    // 3. Test Standard Batch Search (KNN)
    printf("[C Client] Running standard KNN batch search...\n");
    float *knn_queries = (float *)calloc(2 * dimension, sizeof(float));
    // Query 1: matches Record 1 (all 1.0f)
    for (int i = 0; i < dimension; i++) {
        knn_queries[0 * dimension + i] = 1.0f;
    }
    // Query 2: matches Record 2 (first half 1.0f, second half 0.0f)
    for (int i = 0; i < 32; i++) {
        knn_queries[1 * dimension + i] = 1.0f;
    }

    int num_queries = 2;
    int k = 2;
    long long out_ids[4];
    int out_distances[4];

    clock_gettime(CLOCK_MONOTONIC, &start);
    status = vdb_batch_search(thread, "test_index", knn_queries, num_queries, k, out_ids, out_distances);
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_knn_search = get_elapsed_us(start, end);
    free(knn_queries);

    if (status != 0) {
        fprintf(stderr, "[C Client] Batch search failed (code: %d)\n", status);
    } else {
        printf("[C Client] KNN search completed in %.3f us. Results:\n", t_knn_search);
        for (int q = 0; q < num_queries; q++) {
            printf("  Query %d results:\n", q + 1);
            for (int i = 0; i < k; i++) {
                int idx = q * k + i;
                printf("    Rank %d: Vector ID = %lld, Hamming Distance = %d\n", i + 1, out_ids[idx], out_distances[idx]);
            }
        }
    }

    // 4. Test Multi-Family Resonant Voting
    printf("\n[C Client] Running Multi-Family Resonant Voting search...\n");
    int num_voting_queries = 8;
    float *voting_queries = (float *)calloc(num_voting_queries * dimension, sizeof(float));
    // All queries align with Record 1 (all 1.0f)
    for (int q = 0; q < num_voting_queries; q++) {
        for (int j = 0; j < dimension; j++) {
            voting_queries[q * dimension + j] = 1.0f;
        }
    }

    int voting_families[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int voting_thresholds[] = {0, 0, 0, 0, 0, 0, 0, 0}; 

    char *voting_mask = (char *)malloc(outSize);
    memset(voting_mask, 0, outSize);

    clock_gettime(CLOCK_MONOTONIC, &start);
    long long resonant_count = vdb_query_planetary_grid(
        thread, "test_index", voting_queries, voting_families, voting_thresholds, num_voting_queries, voting_mask
    );
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_voting_search = get_elapsed_us(start, end);
    free(voting_queries);

    if (resonant_count < 0) {
        fprintf(stderr, "[C Client] query_planetary_grid failed (code: %lld)\n", resonant_count);
    } else {
        printf("[C Client] Resonant voting completed in %.3f us. Found %lld resonant tiles:\n", 
               t_voting_search, resonant_count);
        for (int i = 0; i < outSize; i++) {
            printf("  Tile ID %d: voting_mask = 0x%02X\n", i, (unsigned char)voting_mask[i]);
        }
    }
    free(voting_mask);

    printf("\n[C Client] Closing Pithos Database...\n");
    vdb_close(thread);

    printf("[C Client] Tearing down GraalVM Isolate...\n");
    graal_tear_down_isolate(thread);

    printf("[C Client] Finished successfully!\n");
    return 0;
}
