//
//  MetalKernelHelper.swift
//  TiledMatrixMultiplication
//
//  Created by Manish Kumar on 2025-07-06.
//

import Metal

class MetalKernelHelper {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    let TILE_SIZE: Int = 16
    var WORK_PER_THREAD = 1 // How many output tiles will one thread produce

    init() {
        // Get a handle to the GPU. There is only one GPU on iPhones and Apple Silicon Macs but
        // there might be more than one on older Intel-based Macs.
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError( "Metal device not available" )
        }

        // A Metal library is a compiled collection of `*.metal` kernels which are embedded into the
        // app bundle by Xcode during the build process. This function can return nil if the kernels did not compile correctly.
        guard let library = try? device.makeDefaultLibrary(bundle: .main) else {
            fatalError( "Could not load default library" )
        }

        // Try to create a `MTLCommandQueue` from a `MTLDevice`. A command queue is a GPU instruction scheduler.
        // It's how you submit work (compute commands) to the GPU. See the function `dispatchThreadgroups()` for details.
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError( "Could not create command queue" )
        }

        self.device = device
        self.commandQueue = commandQueue
        self.library = library
    }

    /// Compile the `kernelFuction` into a compute pipeline state of type `MTLComputePipelineState` that the
    /// GPU can execute. Basically, `makeComputePipelineState()` tells Metal to convert this kernel function
    /// into something which the GPU can execute efficiently.
    func makePipelineFromFunction(_ name: String) -> MTLComputePipelineState? {
        guard let kernelfunction = library.makeFunction(name: name) else {
            fatalError( "Could not find kernel function \(name)" )
        }

        return try? self.device.makeComputePipelineState(function: kernelfunction)
    }

    /// Creates a Metal buffer from a Swift array of any type `T`.
    /// The buffer is created on the current Metal device and contains a copy of the array's data.
    /// This is a generic utility function used to pass data to the GPU from CPU.
    func makeBuffer<T>(from array: [T]) -> MTLBuffer {
        // Use Swift's withUnsafeBytes to access the raw memory of the array.
        return array.withUnsafeBytes { rawBuffer in
            // Create a Metal buffer using the raw memory pointer and the byte count.
            // options: [] means no special storage or caching options are applied.
            // The '!' assumes buffer creation succeeds — should be safe if device is valid and memory is sufficient.
            device.makeBuffer(bytes: rawBuffer.baseAddress!,
                              length: rawBuffer.count,
                              options: [])!
        }
    }

    func makeBuffer(length: Int) -> MTLBuffer {
        device.makeBuffer(length: length, options: [])!
    }

    /// Creates a Metal buffer containing a single constant value of type `T`.
    /// This is typically used to pass small uniform values to a Metal kernel.
    func makeConstant<T>(from value: T) -> MTLBuffer {
        var copy = value
        // Use withUnsafeBytes to get a raw pointer to the constant value.
        return withUnsafeBytes(of: &copy) { rawBuffer in
            // Create a Metal buffer that holds the value.
            // The contents are copied from the raw buffer into GPU memory.
            device.makeBuffer(bytes: rawBuffer.baseAddress!,
                              length: rawBuffer.count,
                              options: [])!
        }
    }

    /// Dispatches a compute kernel using the precompiled pipeline state, binding the provided buffers and constants,
    /// and launching GPU threads with specified configuration.
    ///
    /// - Parameters:
    ///   - buffers: Input/output `MTLBuffer`s to be bound to the kernel.
    ///   - constants: Optional constant `MTLBuffer`s, such as uniform or metadata inputs (default is empty).
    ///   - M, K, N: Matrix sizes.
    func dispatchThreadgroups(kernelName: String,
                              buffers: [MTLBuffer],
                              constants: [MTLBuffer],
                              M: Int, K: Int, N: Int) {
        guard let pipeline = self.makePipelineFromFunction(kernelName) else {
            fatalError( "Could not create compute pipeline state" )
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError( "Could not create command buffer or encoder" )
        }

        encoder.setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            encoder.setBuffer(buf, offset: 0, index: i)
        }

        for (j, constant) in constants.enumerated() {
            encoder.setBuffer(constant, offset: 0, index: buffers.count + j)
        }

        if kernelName == "matmul_tiled_overloaded" {
            WORK_PER_THREAD = 2
        } else {
            WORK_PER_THREAD = 1
        }
        
        let threadgroupSize = MTLSize(width: TILE_SIZE / WORK_PER_THREAD, height: TILE_SIZE / WORK_PER_THREAD, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (N + TILE_SIZE - 1) / TILE_SIZE,
                                          height: (M + TILE_SIZE - 1) / TILE_SIZE,
                                          depth: 1)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
