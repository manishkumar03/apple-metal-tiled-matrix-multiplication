//
//  MatrixBenchmarkRunner.swift
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-06.
//


import UIKit
import Metal

// This struct will also exist on the Metal side.
struct Params {
    var M: UInt32
    var K: UInt32
    var N: UInt32
}

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

    var cpuTime: Double = 0
    var helper: MetalKernelHelper!

    init(M: Int, K: Int, N: Int) {
        self.M = M
        self.K = K
        self.N = N
        self.helper = MetalKernelHelper()
    }

    var cpuTimeMS: Double { cpuTime * 1000.0 }

    func generateRandomMatrices() {
        A = (0..<M*K).map { _ in Float.random(in: -1...1) }
        B = (0..<K*N).map { _ in Float.random(in: -1...1) }
    }

    // The results from CPU multiplication will serve as the benchmark and will also be used for checking the accuracy
    // of multiplication through Metal kernels.
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

    func runAllKernels() async -> String {
        var log: String = ""
        self.generateRandomMatrices()
        self.multiplyOnCPU()

        log += String(format: "✅ CPU time: %.2f ms\n", self.cpuTimeMS)

        let kernelNames = ["matmul_naive",  "matmul_tiled"]
        for kernelName in kernelNames {
            if let result = self.runKernelBenchmark(name: kernelName) {
                log += String(format: "✅ %@: %.2f ms ", result.name, result.duration)
                log += String(format: "(speedup: %.2fx)\n", result.speedup)
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
}
