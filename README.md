# Apple Metal - Tiled Matrix Multiplication

A SwiftUI + Apple Metal iOS app which benchmarks different matrix multiplication strategies using Apple Silicon GPUs. It also implements a naive CPU-based matrix multiplication which serves as a benchmark to compare the performance of various metal kernels and is also used to verify their accuracy.

This repository is an educational example that demonstrates how to use Apple's Metal framework for GPU programming in Swift. It's designed to help you understand the full end-to-end pipeline of compute programming with Metal in a real Swift app. It shows how to prepare data on the CPU, send it to the GPU, run a Metal kernel, and read back the results.

The code includes **extensive inline comments** explaining the functioning of each step and the design choices made.

# Why Naive Matrix Multiplication Is Inefficient And How Tiling Helps

Matrix multiplication might be the most-studied algorithm in the AI literature. It literally is the foundation of every major process in modern machine learning and especially deep learning. So anytime we can speed it up or make it more efficient, even slightly, we can save a lot in terms of computing power, energy and cost.

To understand the need for tiled matrix multiplication, let’s first look at the naive implementation and try to understand why it’s slow.

Suppose we’re multiplying two matrices, **A** and **B**, to produce the output matrix **C**. To compute the first row of matrix C, we multiply the first row of A with each column of B. This means we need to load the entire matrix B from global memory into local (register/shared) memory for just this one row of A.

Next, to compute the second row of C, we take the second row of A and we multiply it with all columns of B, which means we reload matrix B from global memory again. And this goes on for all the rows of A. So if matrix A has 1024 rows, we’ll be loading matrix B 1024 times from global memory. And matrix A also gets loaded from global memory same number of times. This is obviously quite wasteful and hopefully we can do something about it. 


#### But you might ask: Isn’t Global Memory Fast? Why do we care if matrix B is being fetched multiple times?

It’s true that global memory (on the GPU) is just RAM; it’s not like we are reading from a hard disk. But in GPU terms, it’s still slow.

Here is a table showing the memory access speed of various types of GPU memory:

| Memory Type      | Access Speed     | Scope        | Size     | Programmer Control |
|------------------|------------------|--------------|----------|---------------------|
| Registers         | 1 cycle          | Per thread   | ~32 KB   | Automatic           |
| Shared Memory     | ~5 cycles        | Per block    | ~48 KB   | Manual              |
| L1 Cache          | ~5–20 cycles     | Per SM       | ~64 KB   | Automatic           |
| Constant Memory   | ~5–20 cycles     | Global       | ~64 KB   | Manual              |
| L2 Cache          | ~100 cycles      | All SMs      | ~6 MB    | Automatic           |
| Global Memory     | ~500 cycles      | All threads  | ~32 GB   | Manual              |

Reading a single value from global memory takes around **500 clock cycles**. Whereas accessing shared memory (local to a threadgroup/block) takes around **5 clock cycles**. That’s a **100× difference**.

So if we can reduce the number of global memory accesses, even slightly, we can drastically improve performance. This is where **tiling** comes in.


## The Idea Behind Tiled Matrix Multiplication

Instead of reading the entire matrix B for every row of A, we divide matrix B into smaller **tiles**. Each tile can be loaded once into shared memory and then reused by multiple threads.

Taking our previous example of A and B - both the first and the second rows of A need the first column of B. So instead of each one of them fetching it independently, what if that first column of B could be fetched once and then shared by multiple rows of A? This approach cuts down the number of global memory loads dramatically and is the main intuition behind using tiles in shared memory. 

There are two **tricks** that make this algorithm work:

1. The size of a tile is same as that of a threadgroup/block. Since GPUs execute one complete threadgroup/block at a time, each thread in the threadgroup can load one element in the shared tile.
2. The shared tiles are declared using the keyword `threadgroup` which means that all the threads in the threadblock can access it.


#### You might wonder: Why not copy the entire matrices A and B into shared memory once and reuse them everywhere?

**Answer:** Shared memory is tiny. It’s measured in kilobytes (e.g., 48 KB on many GPUs) whereas input matrices might be a few megabytes in size. So there’s no way to fit the whole matrix into shared memory.

That’s why **tiling** is the best compromise. We load just a small chunk (tile) of A and B at a time into shared memory, compute the partial output, and move on to the next tile.


## Performance Improvement

Let’s say you’re multiplying:

A = 1024×1024
B = 1024×1024
Tile size = 32

Then:

- Instead of `1024 × 1024 = 1,048,576` reads from B (one for each row of A),  
- We only need `(1024 / 32) × 1024 = 32,768` reads from B using tiles.

That’s a **32× reduction in global memory reads**, and the key reason tiled matrix multiplication is **orders of magnitude faster** than the naive version.

## Screenshot
![Tiled Matrix Multiplication Results](https://github.com/user-attachments/assets/4f304ccf-5c6a-47a6-9846-0c55e6e176e0)


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

