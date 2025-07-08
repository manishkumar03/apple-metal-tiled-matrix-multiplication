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

        let runner = MatrixBenchmarkRunner(M: M, K: K, N: N)
        output = await runner.runAllKernels()
    }
}
