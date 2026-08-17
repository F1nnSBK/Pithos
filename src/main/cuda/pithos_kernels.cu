#include <stdint.h>
#include <stddef.h>
#include "pithos_kernels.h"

__global__ void batch_hamming_distance_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    const int num_db_vectors,
    const int num_queries,
    const int num_tiers,
    const int* tier_offsets,
    const int* tier_sizes,
    const int num_words_per_vector
) {
    const int query_idx = blockIdx.y;
    const int db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (db_idx >= num_db_vectors || query_idx >= num_queries) {
        return;
    }
    
    int total_distance = 0;
    
    for (int tier = 0; tier < num_tiers; tier++) {
        const int offset = tier_offsets[tier];
        const int size = tier_sizes[tier];
        const int words_in_tier = (size + 63) / 64;
        
        for (int w = 0; w < words_in_tier; w++) {
            const int word_idx = offset + w;
            const uint64_t db_word = db_vectors[db_idx * num_words_per_vector + word_idx];
            const uint64_t query_word = query_vectors[query_idx * num_words_per_vector + word_idx];
            total_distance += __popcll(db_word ^ query_word);
        }
    }
    
    distances[db_idx * num_queries + query_idx] = total_distance;
}

__global__ void batch_hamming_optimized_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    const int num_db_vectors,
    const int num_queries,
    const int num_words_per_vector
) {
    const int query_idx = blockIdx.y;
    const int db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (db_idx >= num_db_vectors || query_idx >= num_queries) {
        return;
    }
    
    extern __shared__ uint64_t shared_query[];
    
    if (threadIdx.x < num_words_per_vector) {
        shared_query[threadIdx.x] = query_vectors[query_idx * num_words_per_vector + threadIdx.x];
    }
    __syncthreads();
    
    int total_distance = 0;
    for (int w = 0; w < num_words_per_vector; w++) {
        const uint64_t db_word = db_vectors[db_idx * num_words_per_vector + w];
        total_distance += __popcll(db_word ^ shared_query[w]);
    }
    
    distances[db_idx * num_queries + query_idx] = total_distance;
}

__global__ void multi_family_voting_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    const int* families,
    const int* thresholds,
    uint8_t* voting_mask,
    const int num_db_vectors,
    const int num_queries,
    const int num_families,
    const int num_words_per_vector
) {
    const int db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (db_idx >= num_db_vectors) {
        return;
    }
    
    uint8_t mask = 0;
    for (int q = 0; q < num_queries; q++) {
        int distance = 0;
        for (int w = 0; w < num_words_per_vector; w++) {
            const uint64_t db_word = db_vectors[db_idx * num_words_per_vector + w];
            const uint64_t query_word = query_vectors[q * num_words_per_vector + w];
            distance += __popcll(db_word ^ query_word);
        }
        
        if (distance <= thresholds[q]) {
            const int family = families[q];
            if (family >= 0 && family < 8) {
                mask |= (1 << family);
            }
        }
    }
    
    voting_mask[db_idx] = mask;
}

__global__ void walsh_hadamard_transform_kernel(
    float* input,
    float* output,
    const int num_vectors,
    const int dimension
) {
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx >= num_vectors) {
        return;
    }
    
    extern __shared__ float shared_data[];
    float* vec = &shared_data[threadIdx.x * dimension];
    
    for (int i = threadIdx.x; i < dimension; i += blockDim.x) {
        vec[i] = input[vec_idx * dimension + i];
    }
    __syncthreads();
    
    const float scale = 0.7071067811865476f;
    
    for (int stride = 1; stride < dimension; stride *= 2) {
        for (int i = threadIdx.x; i < dimension; i += blockDim.x) {
            if (i + stride < dimension) {
                const float a = vec[i];
                const float b = vec[i + stride];
                vec[i] = (a + b) * scale;
                vec[i + stride] = (a - b) * scale;
            }
        }
        __syncthreads();
    }
    
    for (int i = threadIdx.x; i < dimension; i += blockDim.x) {
        output[vec_idx * dimension + i] = vec[i];
    }
}

int launch_batch_hamming_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_tiers,
    const int* tier_offsets,
    const int* tier_sizes,
    cudaStream_t stream
) {
    const int block_size = 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((num_db_vectors + block_size - 1) / block_size, num_queries, 1);
    
    int num_words_per_vector = 0;
    for (int tier = 0; tier < num_tiers; tier++) {
        num_words_per_vector += (tier_sizes[tier] + 63) / 64;
    }
    
    batch_hamming_distance_kernel<<<grid, block, 0, stream>>>(
        db_vectors, query_vectors, distances,
        num_db_vectors, num_queries, num_tiers,
        tier_offsets, tier_sizes, num_words_per_vector
    );
    
    return cudaGetLastError();
}

int launch_batch_hamming_optimized_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_words_per_vector,
    cudaStream_t stream
) {
    const int block_size = 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((num_db_vectors + block_size - 1) / block_size, num_queries, 1);
    
    size_t shared_mem_size = num_words_per_vector * sizeof(uint64_t);
    
    batch_hamming_optimized_kernel<<<grid, block, shared_mem_size, stream>>>(
        db_vectors, query_vectors, distances,
        num_db_vectors, num_queries, num_words_per_vector
    );
    
    return cudaGetLastError();
}

int launch_multi_family_voting_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    const int* families,
    const int* thresholds,
    uint8_t* voting_mask,
    int num_db_vectors,
    int num_queries,
    int num_families,
    int num_words_per_vector,
    cudaStream_t stream
) {
    const int block_size = 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((num_db_vectors + block_size - 1) / block_size, 1, 1);
    
    multi_family_voting_kernel<<<grid, block, 0, stream>>>(
        db_vectors, query_vectors, families, thresholds, voting_mask,
        num_db_vectors, num_queries, num_families, num_words_per_vector
    );
    
    return cudaGetLastError();
}

int launch_walsh_hadamard_kernel(
    float* input,
    float* output,
    int num_vectors,
    int dimension,
    cudaStream_t stream
) {
    const int block_size = 128;
    dim3 block(block_size, 1, 1);
    dim3 grid((num_vectors + block_size - 1) / block_size, 1, 1);
    
    size_t shared_mem_size = block_size * dimension * sizeof(float);
    
    walsh_hadamard_transform_kernel<<<grid, block, shared_mem_size, stream>>>(
        input, output, num_vectors, dimension
    );
    
    return cudaGetLastError();
}

__device__ __forceinline__ float decode_fp8_e4m3_device(uint8_t val) {
    int sign = (val >> 7) & 1;
    int exp = (val >> 3) & 0xF;
    int mant = val & 0x7;
    float sign_mult = (sign == 1) ? -1.0f : 1.0f;
    if (exp == 0) {
        return sign_mult * ((float)mant / 512.0f);
    } else {
        float scale = 1.0f;
        if (exp >= 7) {
            scale = (float)(1 << (exp - 7));
        } else {
            scale = 1.0f / (float)(1 << (7 - exp));
        }
        return sign_mult * scale * (1.0f + (float)mant / 8.0f);
    }
}

__global__ void batch_rerank_fp8_kernel(
    const uint8_t* db_fp8_vectors,
    const float* query_vectors,
    float* distances,
    const int num_db_vectors,
    const int num_queries,
    const int dimension
) {
    const int query_idx = blockIdx.y;
    const int db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (db_idx >= num_db_vectors || query_idx >= num_queries) {
        return;
    }
    
    float total_dist = 0.0f;
    const uint8_t* db_vec = &db_fp8_vectors[db_idx * dimension];
    const float* q_vec = &query_vectors[query_idx * dimension];
    
    for (int d = 0; d < dimension; d++) {
        float db_val = decode_fp8_e4m3_device(db_vec[d]);
        float diff = q_vec[d] - db_val;
        total_dist += diff * diff;
    }
    
    distances[db_idx * num_queries + query_idx] = total_dist;
}

int launch_batch_rerank_fp8_kernel(
    const uint8_t* db_fp8_vectors,
    const float* query_vectors,
    float* out_distances,
    int num_db_vectors,
    int num_queries,
    int dimension,
    cudaStream_t stream
) {
    const int block_size = 256;
    dim3 block(block_size, 1, 1);
    dim3 grid((num_db_vectors + block_size - 1) / block_size, num_queries, 1);
    
    batch_rerank_fp8_kernel<<<grid, block, 0, stream>>>(
        db_fp8_vectors, query_vectors, out_distances,
        num_db_vectors, num_queries, dimension
    );
    
    return cudaGetLastError();
}
