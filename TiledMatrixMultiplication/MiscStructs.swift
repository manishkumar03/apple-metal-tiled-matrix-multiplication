//
//  MiscStructs.swift
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-08.
//

/// We use this `Params` struct to pass matrix dimensions (M, K, N) to the Metal compute kernel.
///
/// Metal kernel buffers are limited to a fixed number of slots (usually 31), and each parameter
/// like `M`, `K`, or `N` would otherwise take up a separate slot. By grouping them into a single struct, we:
///
/// 1. Save buffer slots (only one constant buffer instead of three separate values)
/// 2. Ensure data layout matches exactly between Swift and Metal
/// 3. Make kernel argument passing cleaner and more scalable (e.g., we can add more config later)
///
/// Note: The layout of this struct must exactly match the version declared in the kernel code
/// (same order, types, and alignment) to avoid undefined behavior.
struct Params {
    var M: UInt32
    var K: UInt32
    var N: UInt32
}

/// This struct contains performance and accuracy metrics for a single Metal kernel run:
/// - `name`: The name of the kernel function (e.g. `"matmul_naive"`).
/// - `duration`: Time taken to execute the kernel in milliseconds.
/// - `maxDiff`: Maximum absolute difference between GPU result and CPU reference (used to verify correctness).
/// - `speedup`: How much faster the GPU kernel was compared to the CPU baseline.
struct BenchmarkResult {
    let name: String
    let duration: Double
    let maxDiff: Float
    let speedup: Double
}
