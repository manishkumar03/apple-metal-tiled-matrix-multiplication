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

    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.library = try! device.makeDefaultLibrary(bundle: .main)
    }

    func makeFunction(_ name: String) -> MTLComputePipelineState? {
        try? device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    }

    func makeBuffer(from array: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: array,
                          length: array.count * MemoryLayout<Float>.size,
                          options: [])!
    }

    func makeBuffer(length: Int) -> MTLBuffer {
        device.makeBuffer(length: length, options: [])!
    }

    func dispatchThreadgroups(pipeline: MTLComputePipelineState,
                              buffers: [MTLBuffer],
                              constants: [UInt32],
                              matrixWidth: Int,
                              matrixHeight: Int) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(pipeline)
        for (i, buf) in buffers.enumerated() {
            encoder.setBuffer(buf, offset: 0, index: i)
        }

        encoder.setBytes(constants, length: MemoryLayout<UInt32>.stride * constants.count, index: buffers.count)

        let threadgroupSize = MTLSize(width: TILE_SIZE, height: TILE_SIZE, depth: 1)
        let threadgroups = MTLSize(width: (matrixWidth + 15) / TILE_SIZE,
                                   height: (matrixHeight + 15) / TILE_SIZE,
                                   depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
