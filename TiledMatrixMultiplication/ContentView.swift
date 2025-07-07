//
//  ContentView.swift
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-06.
//

import SwiftUI
import Metal

struct BenchmarkResult {
    let name: String
    let duration: Double
    let maxDiff: Float
    let speedup: Double
}

struct ContentView: View {
    @State private var output = ""
    @State private var isProcessing = false

    let N = 128

    var body: some View {
        VStack {
            Button("Run Matrix Multiplication Benchmarks") {
                Task { await runAll() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isProcessing)
            .padding()

            Text("Results for \(N) x \(N) Matrix:")
                .font(.system(.headline, design: .monospaced))
                .padding(.horizontal)
                .foregroundColor(.secondary)

            ScrollView {
                Text(output)
                    .padding(.horizontal)
                    .font(.system(.caption, design: .monospaced))
            }
        }
    }

    func runAll() async {
        isProcessing = true
        defer { isProcessing = false }

        var log: String = ""
        let runner = MatrixBenchmarkRunner(size: N)
        runner.generateRandomMatrices()
        runner.multiplyOnCPU()

        log += String(format: "✅ CPU time: %.2f ms\n", runner.cpuTimeMS)

        let kernelNames = ["matmul_naive", "matmul_tiled"]
        for kernelName in kernelNames {
            if let result = runner.runKernelBenchmark(name: kernelName) {
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

        output = log
    }
}
