/**
 * @file fpga_dma_demo.c
 * @brief Demonstrates C/C++ direct off-heap virtual memory access and FPGA DMA co-design with Pithos.
 *
 * This example shows how hardware accelerators (PCIe FPGA, OpenCL, Custom ASIC)
 * can stream raw memory-mapped columnar bit vectors directly via DMA without Java GC or CPU copies.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../../include/pithos.h"

int main() {
    graal_isolate_t *isolate = NULL;
    graal_isolatethread_t *thread = NULL;

    printf("================================================================\n");
    printf("         PITHOS C/C++ FPGA & DMA CO-DESIGN DEMO\n");
    printf("================================================================\n\n");

    printf("[1/5] Initializing GraalVM Isolate and Database Coordinator...\n");
    if (graal_create_isolate(NULL, &isolate, &thread) != 0) {
        fprintf(stderr, "Error: Failed to create GraalVM isolate.\n");
        return 1;
    }
    if (vdb_init(thread) != 0) {
        fprintf(stderr, "Error: Failed to initialize Pithos database engine.\n");
        graal_tear_down_isolate(thread);
        return 1;
    }

    // Prepare sample dataset for compilation
    int dimension = 128;
    int num_records = 1000;
    int tiers[] = {64, 128};
    int num_tiers = 2;
    const char *index_path = "fpga_demo_idx";
    const char *index_name = "fpga_index";

    int64_t *ids = (int64_t *)malloc(num_records * sizeof(int64_t));
    float *vectors = (float *)calloc(num_records * dimension, sizeof(float));

    for (int i = 0; i < num_records; i++) {
        ids[i] = 10000 + i;
        for (int d = 0; d < dimension; d++) {
            vectors[i * dimension + d] = (float)((i + d) % 7 - 3);
        }
    }

    printf("[2/5] Compiling 2-Tier Multi-Dimensional Index (%d records, D=%d)...\n", num_records, dimension);
    int status = vdb_compile_index_file(thread, index_path, (int8_t)2, 3389500LL, dimension, tiers, num_tiers, ids, vectors, num_records, PITHOS_QMODE_1BIT);
    free(ids);
    free(vectors);

    if (status != 0) {
        fprintf(stderr, "Error: Compilation failed with code %d\n", status);
        vdb_close(thread);
        graal_tear_down_isolate(thread);
        return 1;
    }

    printf("[3/5] Memory-Mapping Index '%s' Off-Heap...\n", index_name);
    status = vdb_load_index(thread, index_name, index_path);
    if (status != 0) {
        fprintf(stderr, "Error: Failed to map index: %d\n", status);
        vdb_close(thread);
        graal_tear_down_isolate(thread);
        return 1;
    }

    // Retrieve off-heap memory addresses for zero-copy PCIe DMA transfer
    printf("\n[4/5] Inspecting Physical/Virtual Off-Heap Addresses for FPGA DMA Engine:\n");
    uintptr_t tier0_addr = 0, meta_addr = 0, ids_addr = 0;
    int64_t tier0_len = 0, meta_len = 0, ids_len = 0;

    vdb_get_tier_address(thread, index_name, 0, &tier0_addr, &tier0_len);
    vdb_get_metadata_address(thread, index_name, &meta_addr, &meta_len);
    vdb_get_ids_address(thread, index_name, &ids_addr, &ids_len);

    printf("  - Tier 0 Buffer Address   : 0x%016llx (Length: %lld bytes)\n", (unsigned long long)tier0_addr, (long long)tier0_len);
    printf("  - Metadata Sidecar Address: 0x%016llx (Length: %lld bytes)\n", (unsigned long long)meta_addr, (long long)meta_len);
    printf("  - Record IDs Address      : 0x%016llx (Length: %lld bytes)\n", (unsigned long long)ids_addr, (long long)ids_len);

    // Host CPU Query Preconditioning & Binarization
    printf("\n[5/5] Host CPU Query Preconditioning (Rademacher Flipping + Fast Walsh-Hadamard)...\n");
    float *query = (float *)calloc(dimension, sizeof(float));
    for (int d = 0; d < dimension; d++) {
        query[d] = 1.0f;
    }

    int words_count = (dimension + 63) / 64;
    uint64_t *packed_bits = (uint64_t *)malloc(words_count * sizeof(uint64_t));

    status = vdb_transform_and_quantize(thread, index_name, query, packed_bits);
    if (status == 0) {
        printf("  - Continuous Query Vector (D=%d) successfully binarized to %d x 64-bit uint64 words:\n", dimension, words_count);
        for (int w = 0; w < words_count; w++) {
            printf("    Word [%d]: 0x%016llx\n", w, (unsigned long long)packed_bits[w]);
        }
        printf("  - Hardware Acceleration Ready: Pass packed_bits and tier0_addr to FPGA DMA register ring!\n");
    } else {
        fprintf(stderr, "Error: Transform & quantize failed with code %d\n", status);
    }

    free(query);
    free(packed_bits);

    printf("\n[Cleanup] Unmapping index and releasing GraalVM isolate...\n");
    vdb_close(thread);
    graal_tear_down_isolate(thread);

    printf("Done! FPGA & DMA co-design workflow executed successfully.\n");
    return 0;
}
