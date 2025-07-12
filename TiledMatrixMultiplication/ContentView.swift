//
//  ContentView.swift
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-06.
//

import SwiftUI
import Metal

/**
 * SwiftUI View for Matrix Multiplication Benchmark Interface
 *
 * This view provides a user interface for running and comparing different matrix
 * multiplication kernel implementations on Apple Silicon GPUs.

 * Expected Output Format:
 * The results typically show metrics like:
 * - Execution time per kernel
 * - Relative speedup compared to baseline CPU implementation
 */
struct ContentView: View {
    @State private var output = ""
    @State private var isProcessing = false

    // Matrix dimensions: C[M×N] = A[M×K] × B[K×N]
    let M = 128
    let K = 128
    let N = 128

    var body: some View {
        VStack {
            Button("Run Matrix Multiplication Benchmarks") {
                Task { await runAll() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isProcessing)
            .padding()

            Text("Results for \(M)x\(K) X \(K)x\(N) Matrix")
                .font(.system(.subheadline, design: .monospaced))
                .foregroundColor(.secondary)

            ScrollView {
                Text(output)
                    .font(.system(.caption, design: .monospaced))
            }
        }
    }

    /**
     * Execute Kernel Benchmarks
     *
     * This function orchestrates the execution of all matrix multiplication kernel
     * benchmarks and updates the UI with results.
     */
    func runAll() async {
        isProcessing = true
        defer { isProcessing = false }

        let runner = MatrixBenchmarkRunner(M: M, K: K, N: N)
        self.output = await runner.runAllKernels()
    }
}
