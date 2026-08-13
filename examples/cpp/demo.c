#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pithos.h"

int main() {
    graal_isolate_t *isolate = NULL;
    graal_isolatethread_t *thread = NULL;

    printf("[Pithos Demo] Creating GraalVM Isolate...\n");
    if (graal_create_isolate(NULL, &isolate, &thread) != 0) {
        fprintf(stderr, "[Error] Failed to create GraalVM isolate!\n");
        return 1;
    }

    printf("[Pithos Demo] Initializing database coordinator...\n");
    if (vdb_init(thread) != 0) {
        fprintf(stderr, "[Error] Failed to initialize Pithos DB engine!\n");
        graal_tear_down_isolate(thread);
        return 1;
    }

    int dimension = 64;
    int num_records = 3;
    long long ids[] = {100, 200, 300};
    int tiers[] = {32, 64};
    int num_tiers = 2;
    const char *index_name = "demo_index";
    const char *index_path = "pithos_demo.bin";

    float *vectors = (float *)calloc(num_records * dimension, sizeof(float));
    for (int i = 0; i < dimension; i++) {
        vectors[1 * dimension + i] = 1.0f;
    }
    for (int i = 0; i < 32; i++) {
        vectors[2 * dimension + i] = 1.0f;
    }

    printf("[Pithos Demo] Compiling database '%s'...\n", index_path);
    int status = vdb_compile_index_file(thread, (char *)index_path, (char)1, 1737400LL, dimension, tiers, num_tiers, ids, vectors, num_records, 0);
    free(vectors);

    if (status != 0) {
        fprintf(stderr, "[Error] Compilation failed: %d\n", status);
        vdb_close(thread);
        graal_tear_down_isolate(thread);
        return 1;
    }

    printf("[Pithos Demo] Loading memory-mapped index '%s'...\n", index_name);
    status = vdb_load_index(thread, (char *)index_name, (char *)index_path);
    if (status != 0) {
        fprintf(stderr, "[Error] Failed to load index: %d\n", status);
        vdb_close(thread);
        graal_tear_down_isolate(thread);
        return 1;
    }

    int outDim = 0;
    long long outSize = 0;
    int8_t outPlanetId = 0;
    long long outPlanetRadius = 0;
    int outTiersCount = 0;
    status = vdb_get_info(thread, (char *)index_name, &outDim, &outSize, &outPlanetId, &outPlanetRadius, &outTiersCount);

    if (status == 0) {
        printf("[Pithos Demo] Index Attributes:\n");
        printf("  - Size      : %lld records\n", outSize);
        printf("  - Dimension : %d\n", outDim);
    }

    float *query = (float *)calloc(dimension, sizeof(float));
    for (int i = 0; i < dimension; i++) {
        query[i] = 1.0f;
    }

    long long out_ids[2];
    int out_distances[2];
    printf("[Pithos Demo] Querying nearest neighbors...\n");
    status = vdb_batch_search(thread, (char *)index_name, query, 1, 2, out_ids, out_distances);
    free(query);

    if (status == 0) {
        printf("[Pithos Demo] Query Results:\n");
        printf("  - Rank 1: ID = %lld, Dist = %d\n", out_ids[0], out_distances[0]);
        printf("  - Rank 2: ID = %lld, Dist = %d\n", out_ids[1], out_distances[1]);
    } else {
        fprintf(stderr, "[Error] Search failed: %d\n", status);
    }

    printf("[Pithos Demo] Shutting down database...\n");
    vdb_close(thread);
    graal_tear_down_isolate(thread);

    return 0;
}
