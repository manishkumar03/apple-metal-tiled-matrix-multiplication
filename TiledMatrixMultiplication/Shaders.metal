//
//  Shaders.metal
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-06.
//

#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 16;

// Naive
kernel void matmul_naive(device const float* A [[ buffer(0) ]],
                         device const float* B [[ buffer(1) ]],
                         device float* C [[ buffer(2) ]],
                         constant uint& N [[ buffer(3) ]],
                         uint2 threadIdx [[ thread_position_in_threadgroup ]],
                         uint2 blockIdx [[ threadgroup_position_in_grid ]],
                         uint2 globalIdx [[ thread_position_in_grid ]]) {

    int outputRow = int(globalIdx.y); // Row of C being computed by this thread
    int outputCol = int(globalIdx.x); // Col of C being computed by this thread

    if (outputRow >= N || outputCol >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[outputRow * N + k] * B[k * N + outputCol];
    }

    C[outputRow * N + outputCol] = sum;
}

// Tiled Shared Memory
kernel void matmul_tiled(device const float* A [[ buffer(0) ]],
                         device const float* B [[ buffer(1) ]],
                         device float* C [[ buffer(2) ]],
                         constant uint& N [[ buffer(3) ]],
                         uint2 threadIdx [[ thread_position_in_threadgroup ]],
                         uint2 blockIdx [[ threadgroup_position_in_grid ]],
                         uint2 globalIdx [[ thread_position_in_grid ]]) {

    threadgroup float Asub[TILE_SIZE][TILE_SIZE];
    threadgroup float Bsub[TILE_SIZE][TILE_SIZE];

    int outputRow = int(globalIdx.y); // Row of C being computed by this thread
    int outputCol = int(globalIdx.x); // Col of C being computed by this thread

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Phase 1 --- Cooperative tile loading ---
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        Asub[threadIdx.y][threadIdx.x] = (outputRow < N && aCol < N) ? A[outputRow * N + aCol] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = (bRow < N && outputCol < N) ? B[bRow * N + outputCol] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2 --- Compute partial dot product from this tile ---
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store the result in C
    if (outputRow < N && outputCol < N) {
        C[outputRow * N + outputCol] = sum;
    }
}


