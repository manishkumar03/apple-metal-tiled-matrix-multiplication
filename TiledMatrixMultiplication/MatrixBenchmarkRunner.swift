//
//  MatrixBenchmarkRunner.swift
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-06.
//

import UIKit
import Metal

/// Coordinates the benchmarking of matrix multiplication on CPU and GPU.
///
/// This class is responsible for:
/// - Generating random input matrices A and B
/// - Computing the reference output matrix C on the CPU
/// - Running multiple Metal kernels (e.g. naive, tiled) for matrix multiplication
/// - Comparing GPU outputs against the CPU result for correctness
/// - Measuring and reporting execution time and speedup for each kernel
///
/// It provides methods like:
/// - `multiplyOnCPU()` to compute the CPU baseline
/// - `runKernelBenchmark(name:)` to run a Metal kernel and verify correctness
/// - `runAllKernels()` to benchmark and compare all configured GPU kernels
///
/// The results are used to validate correctness and evaluate performance improvements of different GPU implementations.
class MatrixBenchmarkRunner {
    let M: Int
    let N: Int
    let K: Int

    // Declaring matrices to be flat (1D) instead of [[Float]] is a deliberate choice as it provides two main benefits:
    //  - Metal expects flat float* buffers
    //  - We can use memory coalescing and access contiguous memory
    // Also, in the nested array of arrays, each inner array is a separate object on the heap which means we cannot use
    // pointer arithmetic as in Metal. And the access is slower as well due to pointer indirection.
    var A: [Float] = []
    var B: [Float] = []
    var C_cpu: [Float] = []
    var helper: MetalKernelHelper!

    // Time taken by the CPU to execute the baseline implementation
    var cpuTime: Double = 0
    var cpuTimeMS: Double { cpuTime * 1000.0 }

    init(M: Int, K: Int, N: Int) {
        self.M = M
        self.K = K
        self.N = N
        self.helper = MetalKernelHelper()
    }

    /// Generates random input matrices A and B with values in the range [-1, 1].
    ///
    /// - A is an M×K matrix stored as a flat array (row-major order)
    /// - B is a K×N matrix stored as a flat array (row-major order)
    ///
    /// These matrices are used as inputs for both CPU and GPU matrix multiplication benchmarks.
    func generateRandomMatrices() {
        A = (0..<M*K).map { _ in Float.random(in: -1...1) }
        B = (0..<K*N).map { _ in Float.random(in: -1...1) }
    }

    /// Perform matrix multiplication on the CPU: C = A × B
    ///
    /// The results from CPU-based matrix multiplication serve two purposes:
    ///  - Timing benchmark: it provides a baseline for performance comparison.
    ///  - Accuracy check: results from Metal GPU kernels are compared against this
    ///    to verify correctness (using max difference or error threshold).
    func multiplyOnCPU() {
        C_cpu = [Float](repeating: 0, count: M * N)
        let start = CACurrentMediaTime()
        for m in 0..<M {
            for n in 0..<N {
                var sum: Float = 0
                for k in 0..<K {
                    sum += A[m * K + k] * B[k * N + n]
                }
                C_cpu[m * N + n] = sum
            }
        }
        cpuTime = CACurrentMediaTime() - start
    }

    /// Runs a specified Metal compute kernel for matrix multiplication and benchmarks its performance.
    ///
    /// - Parameter kernelName: The name of the Metal kernel function to run.
    /// - Returns: A `BenchmarkResult` containing:
    ///   - `name`: the kernel function name,
    ///   - `duration`: time in milliseconds,
    ///   - `maxDiff`: maximum absolute difference with the CPU result,
    ///   - `speedup`: ratio of CPU time to GPU time.
    func runKernelBenchmark(name kernelName: String) -> BenchmarkResult? {
        guard let pipeline = helper.makePipelineFromFunction(kernelName) else { return nil }

        let bufferA = helper.makeBuffer(from: A)
        let bufferB = helper.makeBuffer(from: B)
        let bufferC = helper.makeBuffer(length: M * N * MemoryLayout<Float>.size)
        let constants = helper.makeConstant(from: Params(M: UInt32(M), K: UInt32(K), N: UInt32(N)))

        let start = CACurrentMediaTime()
        helper.dispatchThreadgroups(pipeline: pipeline,
                                    buffers: [bufferA, bufferB, bufferC],
                                    constants: [constants],
                                    M: M, K: K, N: N)
        let duration = CACurrentMediaTime() - start

        var C_gpu = [Float](repeating: 0, count: N * N)
        memcpy(&C_gpu, bufferC.contents(), C_gpu.count * MemoryLayout<Float>.size)

        let maxDiff = zip(C_cpu, C_gpu).map { abs($0 - $1) }.max() ?? 0
        let speedup = cpuTime / duration

        return BenchmarkResult(name: kernelName, duration: duration * 1000.0, maxDiff: maxDiff, speedup: speedup)
    }

    /// Runs all configured Metal matrix multiplication kernels, benchmarks their performance, and verifies correctness.
    ///
    /// - Returns: A summary `String` containing execution times and speedups for each kernel.
    ///            Also prints max difference for each result.
    func runAllKernels() async -> String {
        var log: String = ""
        self.generateRandomMatrices()
        self.multiplyOnCPU()

        let header = self.formatResultHeader()
        log += header + "\n"
        log += String(repeating: "-", count: 37) + "\n"

        let cpuResult = BenchmarkResult(name: "CPU", duration: self.cpuTimeMS, maxDiff: 0, speedup: 1)
        let line = self.formatResultLine(for: cpuResult)
        log += line + "\n"

        let kernelNames = ["matmul_naive",  "matmul_tiled"]
        for kernelName in kernelNames {
            if let result = self.runKernelBenchmark(name: kernelName) {
                let line = self.formatResultLine(for: result)
                log += line + "\n"

                print("Max diff: \(result.maxDiff) for \(kernelName)")
                if result.maxDiff > 1e-2 {
                    print("Results are not matching for \(kernelName)")
                    fatalError(String(format: "Max diff: %.6f", result.maxDiff))
                }
            } else {
                log += "\(kernelName) failed to run\n"
            }
        }

        return log
    }

    private func formatResultHeader() -> String {
        let header = [
            "Kernel".padding(toLength: 20, withPad: " ", startingAt: 0),
            "Time (ms)".padding(toLength: 10, withPad: " ", startingAt: 0),
            "Speedup".padding(toLength: 10, withPad: " ", startingAt: 0)
        ].joined()

        return header
    }

    private func formatResultLine(for result: BenchmarkResult) -> String {
        let line = [
            result.name.padding(toLength: 20, withPad: " ", startingAt: 0),
            String(format: "%.2f", result.duration).padding(toLength: 10, withPad: " ", startingAt: 0),
            String(format: "%.2fx", result.speedup).padding(toLength: 10, withPad: " ", startingAt: 0)
        ].joined()

        return line
    }
}
