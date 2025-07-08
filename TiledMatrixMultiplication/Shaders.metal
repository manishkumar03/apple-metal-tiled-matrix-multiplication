//
//  Shaders.metal
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-06.
//

#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 16;

struct Params {
    uint M;
    uint K;
    uint N;
};

// Naive
kernel void matmul_naive(device const float* A [[ buffer(0) ]],
                         device const float* B [[ buffer(1) ]],
                         device float* C [[ buffer(2) ]],
                         constant Params& params [[ buffer(3) ]],
                         uint2 threadIdx [[ thread_position_in_threadgroup ]],
                         uint2 blockIdx [[ threadgroup_position_in_grid ]],
                         uint2 globalIdx [[ thread_position_in_grid ]]) {
    const int M = params.M;
    const int K = params.K;
    const int N = params.N;
    int outputRow = int(globalIdx.y); // Row of C being computed by this thread
    int outputCol = int(globalIdx.x); // Col of C being computed by this thread

    if (outputRow >= M || outputCol >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[outputRow * K + k] * B[k * N + outputCol];
    }

    C[outputRow * N + outputCol] = sum;
}

// Tiled Shared Memory
kernel void matmul_tiled_square(device const float* A [[ buffer(0) ]],
                                device const float* B [[ buffer(1) ]],
                                device float* C [[ buffer(2) ]],
                                constant Params& params [[ buffer(3) ]],
                                uint2 threadIdx [[ thread_position_in_threadgroup ]],
                                uint2 blockIdx [[ threadgroup_position_in_grid ]],
                                uint2 globalIdx [[ thread_position_in_grid ]]) {
    const int N = params.N;
    threadgroup float Asub[TILE_SIZE][TILE_SIZE];
    threadgroup float Bsub[TILE_SIZE][TILE_SIZE];

    int outputRow = int(globalIdx.y); // Row of C being computed by this thread
    int outputCol = int(globalIdx.x); // Col of C being computed by this thread

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Phase 1 --- Cooperative tile loading ---

        // (aCol, bRow) determine which piece of global memory each thread is responsible for loading into the shared tile.
        int aCol = t * TILE_SIZE + threadIdx.x; // The column index in matrix A that this thread will load
        int bRow = t * TILE_SIZE + threadIdx.y; // The row index in matrix B that this thread will load

        Asub[threadIdx.y][threadIdx.x] = (outputRow < N && aCol < N) ? A[outputRow * N + aCol] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = (bRow < N && outputCol < N) ? B[bRow * N + outputCol] : 0.0f;

        // Wait for all threads to finish loading into shared memory
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2 --- Compute partial dot product from this tile ---
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        // Wait for all threads before reusing shared memory in next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store the result in C
    if (outputRow < N && outputCol < N) {
        C[outputRow * N + outputCol] = sum;
    }
}

// Tiled Shared Memory
kernel void matmul_tiled(device const float* A [[ buffer(0) ]],
                         device const float* B [[ buffer(1) ]],
                         device float* C [[ buffer(2) ]],
                         constant Params& params [[ buffer(3) ]],
                         uint2 threadIdx [[ thread_position_in_threadgroup ]],
                         uint2 blockIdx [[ threadgroup_position_in_grid ]],
                         uint2 globalIdx [[ thread_position_in_grid ]]) {
    const int M = params.M;
    const int K = params.K;
    const int N = params.N;
    threadgroup float Asub[TILE_SIZE][TILE_SIZE];
    threadgroup float Bsub[TILE_SIZE][TILE_SIZE];

    int outputRow = int(globalIdx.y); // Row of C being computed by this thread
    int outputCol = int(globalIdx.x); // Col of C being computed by this thread

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Phase 1 --- Cooperative tile loading ---

        // (aCol, bRow) determine which piece of global memory each thread is responsible for loading into the shared tile.
        int aCol = t * TILE_SIZE + threadIdx.x; // The column index in matrix A that this thread will load
        int bRow = t * TILE_SIZE + threadIdx.y; // The row index in matrix B that this thread will load

        Asub[threadIdx.y][threadIdx.x] = (outputRow < M && aCol < K) ? A[outputRow * K + aCol] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = (bRow < K && outputCol < N) ? B[bRow * N + outputCol] : 0.0f;

        // Wait for all threads to finish loading into shared memory
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2 --- Compute partial dot product from this tile ---
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        // Wait for all threads before reusing shared memory in next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store the result in C
    if (outputRow < M && outputCol < N) {
        C[outputRow * N + outputCol] = sum;
    }
}
