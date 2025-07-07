//
//  MatrixBenchmarkRunner.swift
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-06.
//


import UIKit
import Metal

class MatrixBenchmarkRunner {
    let N: Int

    // Declaring matrices to be flat (1D) instead of [[Float]] is a deliberate choice as it provides two main benefits:
    //  - Metal expects flat float* buffers
    //  - We can use memory coalescing and access contiguous memory
    // Also, in the nested array of arrays, each inner array is a separate object on the heap which means we cannot use
    // pointer arithmetic as in Metal. And the access is slower as well due to pointer indirection.
    var A: [Float] = []
    var B: [Float] = []
    var C_cpu: [Float] = []

    var cpuTime: Double = 0

    var device: MTLDevice!
    var queue: MTLCommandQueue!
    var helper: MetalKernelHelper!
    
    init(size: Int) {
        self.N = size
    }

    var cpuTimeMS: Double { cpuTime * 1000.0 }

    func prepareMetal() -> Bool {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return false }
        self.device = device
        self.queue = queue
        self.helper = MetalKernelHelper(device: device)
        return true
    }

    func generateRandomMatrices() {
        A = (0..<N*N).map { _ in Float.random(in: -1...1) }
        B = (0..<N*N).map { _ in Float.random(in: -1...1) }
    }

    // The results from CPU multiplication will serve as the benchmark and will also be used for checking the accuracy
    // of multiplication through Metal kernels.
    func multiplyOnCPU() {
        C_cpu = [Float](repeating: 0, count: N * N)
        let start = CACurrentMediaTime()
        for i in 0..<N {
            for j in 0..<N {
                var sum: Float = 0
                for k in 0..<N {
                    sum += A[i * N + k] * B[k * N + j]
                }
                C_cpu[i * N + j] = sum
            }
        }
        cpuTime = CACurrentMediaTime() - start
    }

    func runKernelBenchmark(name kernelName: String) -> BenchmarkResult? {
        guard let pipeline = helper.makeFunction(kernelName) else { return nil }

        let bufferA = helper.makeBuffer(from: A)
        let bufferB = helper.makeBuffer(from: B)
        let bufferC = helper.makeBuffer(length: N * N * MemoryLayout<Float>.size)
        let constants = [UInt32(N)]

        let start = CACurrentMediaTime()
        helper.dispatchThreadgroups(pipeline: pipeline,
                                    buffers: [bufferA, bufferB, bufferC],
                                    constants: constants,
                                    matrixWidth: N, matrixHeight: N)
        let duration = CACurrentMediaTime() - start

        var C_gpu = [Float](repeating: 0, count: N * N)
        memcpy(&C_gpu, bufferC.contents(), C_gpu.count * MemoryLayout<Float>.size)

        let maxDiff = zip(C_cpu, C_gpu).map { abs($0 - $1) }.max() ?? 0
        let speedup = cpuTime / duration

        return BenchmarkResult(name: kernelName, duration: duration * 1000.0, maxDiff: maxDiff, speedup: speedup)
    }
}
