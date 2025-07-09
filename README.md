# Apple Metal - Tiled Matrix Multiplication

A SwiftUI + Apple Metal iOS app which benchmarks different matrix multiplication strategies using Apple Silicon GPUs. It also implements a naive CPU-based matrix multiplication which serves as a benchmark to compare the performance of various metal kernels and is also used to verify their accuracy.

This repository is an educational example that demonstrates how to use Apple's Metal framework for GPU programming in Swift. It's designed to help you understand the full end-to-end pipeline of compute programming with Metal in a real Swift app. It shows how to prepare data on the CPU, send it to the GPU, run a Metal kernel, and read back the results.

The code includes **extensive inline comments** explaining the functioning of each step and the design choices made.

## Core Functionality

The project implements three matrix multiplication techniques and compares their performance:

- Naive implementation using nested for-loops on the CPU
- Naive implementation using a Metal kernel (`matmul_naive`)
- Optimized implementation using shared-memory tiles using a Metal kernel (`matmul_tiled`)


## Topics Covered

The projet shows how to:

- Set up a complete Metal compute pipeline in Swift
- Write and compile a Metal kernel (`.metal` file)
- Transfer data between CPU and GPU using Metal buffers
- Dispatch GPU workloads with `dispatchThreadGroups`
- Get results back from the GPU

## Run Requirements
* 	This is an iOS project so you'd need an Apple Mac to run it
* 	Xcode with Metal support

### License

MIT License — free to learn from and build on. A star ⭐️ would be awesome if this helped you!


### Author

Created by Manish Kumar

Questions welcome!

