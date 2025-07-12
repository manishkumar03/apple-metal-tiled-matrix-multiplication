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

/**
 * Naive Matrix Multiplication Kernel Without Any Optimizations
 *
 * Computes C = A × B where A is M×K, B is K×N, and C is M×N.
 *
 * Algorithm Overview:
 * This kernel implements a naive matrix multiplication algorithm
 */
kernel void matmul_naive(device const float* A [[ buffer(0) ]],
                         device const float* B [[ buffer(1) ]],
                         device float* C [[ buffer(2) ]],
                         constant Params& params [[ buffer(3) ]],
                         uint2 threadIdx [[ thread_position_in_threadgroup ]],
                         uint2 blockIdx [[ threadgroup_position_in_grid ]],
                         uint2 globalIdx [[ thread_position_in_grid ]]) {
    // Matrix dimensions: C[M×N] = A[M×K] × B[K×N]
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

/**
 * Tiled Matrix Multiplication Kernel with Shared Memory Optimization
 *
 * Computes C = A × B where A is M×K, B is K×N, and C is M×N.
 *
 * Algorithm Overview:
 * This kernel implements a tiled matrix multiplication algorithm that dramatically
 * reduces global memory bandwidth requirements by leveraging fast shared memory.
 * Instead of each thread loading all K elements needed for its dot product
 * (resulting in K×M×N total loads), threads cooperatively load small tiles into
 * shared memory that are reused by all threads in the threadgroup.
 *
 * Key Design Principles:
 * 1. Work Distribution: Each thread computes exactly one element of C
 * 2. Tiling Strategy: The K dimension is divided into TILE_SIZE chunks
 * 3. Cooperative Loading: All threads in a threadgroup load one tile from A and B
 * 4. Memory Access Pattern: Optimized for coalesced reads
 *
 * Performance Characteristics:
 * - Global memory reads: Reduced by factor of TILE_SIZE (typically 16x reduction)
 */
kernel void matmul_tiled(device const float* A [[ buffer(0) ]],
                         device const float* B [[ buffer(1) ]],
                         device float* C [[ buffer(2) ]],
                         constant Params& params [[ buffer(3) ]],
                         uint2 threadIdx [[ thread_position_in_threadgroup ]],
                         uint2 blockIdx [[ threadgroup_position_in_grid ]],
                         uint2 globalIdx [[ thread_position_in_grid ]]) {
    // Matrix dimensions: C[M×N] = A[M×K] × B[K×N]
    const int M = params.M;
    const int K = params.K;
    const int N = params.N;

    // Declare shared memory tiles that will be co-operatively loaded by all threads in the threadgroup.
    // Each tile holds a TILE_SIZE×TILE_SIZE submatrix. These will be reloaded and reused multiple times as we slide
    // along the K dimension.
    threadgroup float Asub[TILE_SIZE][TILE_SIZE];
    threadgroup float Bsub[TILE_SIZE][TILE_SIZE];

    // Determine which element of C this thread is responsible for computing.
    // Each thread computes exactly one element of the output matrix C.
    int outputRow = int(globalIdx.y); // Row of C being computed by this thread (ranges from 0 to M-1)
    int outputCol = int(globalIdx.x); // Col of C being computed by this thread (ranges from 0 to N-1)

    // Initialize accumulator for this thread's output element which will accumulate partial products from all tiles.
    float sum = 0.0f;

    // Calculate total number of *complete* tiles needed to cover the K dimension. If the matrix size is not an exact multiple of
    // TILE_SIZE, some of the tiles will have extra threads. We account for these by checking the bounds before loading the tile.
    // Example: if K=1000 and TILE_SIZE=16, we need 63 tiles (ceiling division)
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Main loop: iterate through all tiles along the K dimension.
    // Each iteration processes one TILE_SIZE×TILE_SIZE block from A and B.
    for (int t = 0; t < numTiles; ++t) {
        // ===== PHASE 1: COOPERATIVE LOADING =====
        // All threads in the threadgroup work together to load one tile from A and one from B.

        // (aCol, bRow) determine which piece of global memory each thread is responsible for loading into the shared tile.
        //
        // Calculate the column index for matrix A this thread should load.
        // We need elements from row 'outputRow' of A but the column index advances by TILE_SIZE with each tile iteration.
        // E.g. Thread with threadIdx.x=0 loads from columns 0, 16, 32, ...
        // Thread with threadIdx.x=1 loads from columns 1, 17, 33, ... until all tiles are exhausted.
        int aCol = t * TILE_SIZE + threadIdx.x;

        // Calculate the row index for matrix B this thread should load.
        // We need elements from column 'outputCol' of B but the row index advances by TILE_SIZE with each tile iteration.
        // E.g. Thread with threadIdx.y=0 loads from rows 0, 16, 32, ...
        // Thread with threadIdx.y=1 loads from rows 1, 17, 33, ...
        int bRow = t * TILE_SIZE + threadIdx.y;

        // The intuition for shared tile loading is that each output element C[outputRow][outputCol] is the result of
        // combining row `outputRow` from matrix A with column `outputCol` from matrix B.
        // - The row comes from A (fixed outputRow, iterate over k)
        // - The column comes from B (fixed outputCol, iterate over k)
        // So: C[outputRow][outputCol] = sum_over_k( A[outputRow][k] * B[k][outputCol] )
        // That's why in this tiled matmul kernel, each thread calculates one output element at (outputRow,outputCol)
        // and loads data from both a row of A and a column of B.
        //
        // Load one element from A into shared memory.
        // A is stored in row-major order, so element A[row][col] is at A[row * K + col].
        // For tiles which have extra threads (tiles extend beyond matrix bounds), assign zero to the tile elements.
        Asub[threadIdx.y][threadIdx.x] = (outputRow < M && aCol < K) ? A[outputRow * K + aCol] : 0.0f;

        // Load one element from B into shared memory.
        // B is stored in row-major order, so element B[row][col] is at B[row * N + col].
        // For tiles which have extra threads (tiles extend beyond matrix bounds), assign zero to the tile elements.
        Bsub[threadIdx.y][threadIdx.x] = (bRow < K && outputCol < N) ? B[bRow * N + outputCol] : 0.0f;

        // SYNCHRONIZATION POINT 1: Wait for all threads to complete loading
        // This barrier ensures that no thread starts the computation before all threads have finished loading the tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== PHASE 2: COMPUTATION =====
        // Compute partial dot product from this tile.
        // Each thread computes just one element of the output matrix C[outputRow][outputCol]. It does this by:
        //  - walking across one row of the tile of A (fixed threadIdx.y)
        //  - and one column of the tile of B (fixed threadIdx.x)
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        // SYNCHRONIZATION POINT 2: Wait before moving to next tile
        // This barrier prevents any thread from overwriting shared memory while other threads are still computing
        // with current tile data. Need to wait for all threads to finish their computation before reusing shared
        // memory in the next iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== PHASE 3: WRITE RESULT =====
    // After processing all tiles, write the final result to global memory.
    // Check bounds to make sure this threads maps to a valid element in C. This handles cases where matrix dimensions
    // aren't an exact multiples of TILE_SIZE.
    if (outputRow < M && outputCol < N) {
        C[outputRow * N + outputCol] = sum;
    }
}

/**
 * Tiled Matrix Multiplication Kernel with Shared Memory Optimization
 *
 * Computes C = A × B where A is M×K, B is K×N, and C is M×N.
 *
 * Algorithm Overview:
 * This kernel implements a tiled matrix multiplication algorithm that dramatically
 * reduces global memory bandwidth requirements by leveraging fast shared memory.
 * Instead of each thread loading all K elements needed for its dot product
 * (resulting in K×M×N total loads), threads cooperatively load small tiles into
 * shared memory that are reused by all threads in the threadgroup.
 *
 * Key Design Principles:
 * 1. Work Distribution: Each thread computes exactly one element of C
 * 2. Tiling Strategy: The K dimension is divided into TILE_SIZE chunks
 * 3. Cooperative Loading: All threads in a threadgroup load one tile from A and B
 * 4. Memory Access Pattern: Optimized for coalesced reads
 *
 * Performance Characteristics:
 * - Global memory reads: Reduced by factor of TILE_SIZE (typically 16x reduction)
 */
kernel void matmul_tiled_overloaded(device const float* A [[ buffer(0) ]],
                                    device const float* B [[ buffer(1) ]],
                                    device float* C [[ buffer(2) ]],
                                    constant Params& params [[ buffer(3) ]],
                                    uint2 threadIdx [[ thread_position_in_threadgroup ]],
                                    uint2 blockIdx [[ threadgroup_position_in_grid ]],
                                    uint2 globalIdx [[ thread_position_in_grid ]]) {
    // Matrix dimensions: C[M×N] = A[M×K] × B[K×N]
    const int M = params.M;
    const int K = params.K;
    const int N = params.N;
    const int WORK_PER_THREAD = 2;

    // Declare shared memory tiles that will be co-operatively loaded by all threads in the threadgroup.
    // Each tile holds a TILE_SIZE×TILE_SIZE submatrix. These will be reloaded and reused multiple times as we slide
    // along the K dimension.
    threadgroup float Asub[TILE_SIZE][TILE_SIZE];
    threadgroup float Bsub[TILE_SIZE][TILE_SIZE];

    // Determine which element of C this thread is responsible for computing.
    // Each thread computes exactly one element of the output matrix C.
    //    int outputRow = int(globalIdx.y); // Row of C being computed by this thread (ranges from 0 to M-1)
    //    int outputCol = int(globalIdx.x); // Col of C being computed by this thread (ranges from 0 to N-1)

    // Initialize accumulator for this thread's output element which will accumulate partial products from all tiles.
    float sum[2][2] = {{0.0f,0.0f}, {0.0f, 0.0f}};

    // Calculate total number of *complete* tiles needed to cover the K dimension. If the matrix size is not an exact multiple of
    // TILE_SIZE, some of the tiles will have extra threads. We account for these by checking the bounds before loading the tile.
    // Example: if K=1000 and TILE_SIZE=16, we need 63 tiles (ceiling division)
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Main loop: iterate through all tiles along the K dimension.
    // Each iteration processes one TILE_SIZE×TILE_SIZE block from A and B.

    uint2 blockOrigin = blockIdx * WORK_PER_THREAD;

    for (int t = 0; t < numTiles; ++t) {
        int tileCol;
        int tileRow;

        for (int i = 0; i < WORK_PER_THREAD; i++) {         // Go across rows to the right
            for (int j = 0; j < WORK_PER_THREAD; j++) {     // Go down columns
                tileRow = threadIdx.y * WORK_PER_THREAD + i;
                tileCol = threadIdx.x * WORK_PER_THREAD + j;

                // Row in A
                int globalRowInA = blockIdx.y * TILE_SIZE + tileRow;
                int globalColInA = t * TILE_SIZE + tileCol;

                int globalRowInB = t * TILE_SIZE + tileRow;
                int globalColInB = blockIdx.x * TILE_SIZE + tileCol;

                if (globalRowInA < M && globalColInA < K) {
                    Asub[tileRow][tileCol] = A[(globalRowInA * K) + globalColInA];
                } else {
                    Asub[tileRow][tileCol] = 0.0f;
                }

                if (globalRowInB < K && globalColInB < N) {
                    Bsub[tileRow][tileCol] = B[(globalRowInB * N) + globalColInB];
                } else {
                    Bsub[tileRow][tileCol] = 0.0f;
                }
            }
        }

        // SYNCHRONIZATION POINT 1: Wait for all threads to complete loading
        // This barrier ensures that no thread starts the computation before all threads have finished loading the tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ===== PHASE 2: COMPUTATION =====
        // Compute partial dot product from this tile.
        // Each thread computes just one element of the output matrix C[outputRow][outputCol]. It does this by:
        //  - walking across one row of the tile of A (fixed threadIdx.y)
        //  - and one column of the tile of B (fixed threadIdx.x)

        for (int i = 0; i < WORK_PER_THREAD; ++i) {
            for (int j = 0; j < WORK_PER_THREAD; ++j) {
                for (int k = 0; k < TILE_SIZE; ++k) {
                    sum[i][j] += Asub[threadIdx.y * WORK_PER_THREAD + i][k] * Bsub[k][threadIdx.x * WORK_PER_THREAD + j];
                }
            }
        }

        // SYNCHRONIZATION POINT 2: Wait before moving to next tile
        // This barrier prevents any thread from overwriting shared memory while other threads are still computing
        // with current tile data. Need to wait for all threads to finish their computation before reusing shared
        // memory in the next iteration.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== PHASE 3: WRITE RESULT =====
    // After processing all tiles, write the final result to global memory.
    // Check bounds to make sure this threads maps to a valid element in C. This handles cases where matrix dimensions
    // aren't an exact multiples of TILE_SIZE.
    
    int outputRow = (blockIdx.y * TILE_SIZE) + (threadIdx.y * WORK_PER_THREAD);
    int outputCol = (blockIdx.x * TILE_SIZE) + (threadIdx.x * WORK_PER_THREAD);
    for (int i = 0; i < WORK_PER_THREAD; i++) {         // Go across rows to the right
        for (int j = 0; j < WORK_PER_THREAD; j++) {     // Go down columns
            int outCol = outputCol + j;
            int outRow = outputRow + i;

            if (outRow < M && outCol < N) {
                C[outRow * N + outCol] = sum[i][j];
            }
        }
    }
}

