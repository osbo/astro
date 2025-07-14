//
//  MetalKernelsRadixSort.swift
//  astro
//
//  Adapted from MetalKernels by Audulus LLC
//

import Metal
import Foundation

// Swift equivalents of the Metal enums
enum ScanBufferIndex: Int {
    case input = 0
    case output = 1
    case aux = 2
    case length = 3
    case zeroff = 4
    case indirectArguments = 5
    case lengths = 6
}

enum SplitBufferIndex: Int {
    case input = 0
    case inputIndices = 1
    case output = 2
    case outputIndices = 3
    case count = 4
    case bit = 5
    case e = 6
    case f = 7
}

class MetalKernelsRadixSort {
    private let device: MTLDevice
    private let scanKernel: ScanKernel
    private let splitKernel: SplitKernel
    
    init(device: MTLDevice) {
        self.device = device
        self.scanKernel = ScanKernel(device: device)
        self.splitKernel = SplitKernel(device: device)
    }
    
    func sort(commandBuffer: MTLCommandBuffer,
              input: MTLBuffer,
              inputIndices: MTLBuffer,
              output: MTLBuffer,
              outputIndices: MTLBuffer,
              length: UInt32) {
        
        // Validate buffer sizes
        let requiredSize = Int(length) * MemoryLayout<UInt64>.size
        assert(input.length >= requiredSize, "Input buffer too small")
        assert(inputIndices.length >= Int(length) * MemoryLayout<UInt32>.size, "Input indices buffer too small")
        assert(output.length >= requiredSize, "Output buffer too small")
        assert(outputIndices.length >= Int(length) * MemoryLayout<UInt32>.size, "Output indices buffer too small")
        
        // Ensure buffers are large enough
        splitKernel.setMaxLength(length)
        
        var keysIn = input
        var keysOut = output
        var indicesIn = inputIndices
        var indicesOut = outputIndices
        
        // Sort 64 bits, one bit at a time
        for i in 0..<64 {
            splitKernel.encodeSplitTo(commandBuffer,
                                     input: keysIn,
                                     inputIndices: indicesIn,
                                     output: keysOut,
                                     outputIndices: indicesOut,
                                     bit: UInt32(i),
                                     length: length)
            
            // Add memory barrier to ensure all writes are complete before next iteration
            if let compute = commandBuffer.makeComputeCommandEncoder() {
                compute.memoryBarrier(resources: [keysOut, indicesOut])
                compute.endEncoding()
            }
            
            // Swap buffers for next iteration
            swap(&keysIn, &keysOut)
            swap(&indicesIn, &indicesOut)
        }
        
        // Copy final result back to original output buffers if needed
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: keysIn, sourceOffset: 0, to: output, destinationOffset: 0, size: Int(length) * MemoryLayout<UInt64>.size)
            blit.copy(from: indicesIn, sourceOffset: 0, to: outputIndices, destinationOffset: 0, size: Int(length) * MemoryLayout<UInt32>.size)
            blit.endEncoding()
        }
    }
}

// MARK: - Scan Kernel

class ScanKernel {
    private let device: MTLDevice
    private let prefixSumPipeline: MTLComputePipelineState
    private let prefixFixupPipeline: MTLComputePipelineState
    private let scanThreadgroupsPipeline: MTLComputePipelineState
    
    init(device: MTLDevice) {
        self.device = device
        
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to get default Metal library")
        }
        
        do {
            self.prefixSumPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "prefixSum")!)
            self.prefixFixupPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "prefixFixup")!)
            self.scanThreadgroupsPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "scan_threadgroups")!)
        } catch {
            fatalError("Failed to create scan pipelines: \(error)")
        }
    }
    
    func encodeScanTo(_ commandBuffer: MTLCommandBuffer,
                      input: MTLBuffer,
                      output: MTLBuffer,
                      length: UInt32) {

        guard length > 0 else { return }

        let threadgroupSize = 512
        let numThreadgroups = (Int(length) + (threadgroupSize * 2) - 1) / (threadgroupSize * 2)

        var lengthVar = length
        var zeroff: UInt32 = 1

        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.label = "Scan Operation"

            if numThreadgroups <= 1 {
                // A single pass is enough.
                // Still need an aux buffer to satisfy the function signature, even if it's not used for fixup.
                let auxBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModePrivate)!
                auxBuffer.label = "Scan Aux Buffer (dummy)"

                compute.setComputePipelineState(prefixSumPipeline)
                compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
                compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
                compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
                compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
                compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))

            } else {
                // Multiple threadgroups: requires multi-pass scan
                let auxBuffer = device.makeBuffer(length: numThreadgroups * MemoryLayout<UInt32>.size, options: .storageModePrivate)!
                auxBuffer.label = "Scan Aux Buffer"

                // 1. First pass: block-wise scan
                compute.setComputePipelineState(prefixSumPipeline)
                compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
                compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
                compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
                compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
                compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))

                // 2. Second pass: scan the auxiliary buffer. This scan itself could be multi-pass, but we assume it fits in one.
                var auxLength = UInt32(numThreadgroups)
                // We need a dummy aux buffer for this second scan pass to satisfy the function signature.
                let aux2Buffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModePrivate)!
                aux2Buffer.label = "Scan Aux^2 Buffer (dummy)"

                compute.setComputePipelineState(prefixSumPipeline)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.input.rawValue)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.output.rawValue) // In-place
                compute.setBuffer(aux2Buffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
                compute.setBytes(&auxLength, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
                compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
                compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))

                // 3. Third pass: fix up the main buffer
                compute.setComputePipelineState(prefixFixupPipeline)
                compute.setBuffer(output, offset: 0, index: ScanBufferIndex.input.rawValue)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
                compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
                compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            }
            compute.endEncoding()
        }
    }
}

// MARK: - Split Kernel

class SplitKernel {
    private let device: MTLDevice
    private let prepPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let scanKernel: ScanKernel
    private var eBuffer: MTLBuffer
    private var fBuffer: MTLBuffer
    
    init(device: MTLDevice) {
        self.device = device
        self.scanKernel = ScanKernel(device: device)
        
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to get default Metal library")
        }
        
        do {
            self.prepPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "split_prep")!)
            self.scatterPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "split_scatter")!)
        } catch {
            fatalError("Failed to create split pipelines: \(error)")
        }
        
        // Initialize with default size
        self.eBuffer = device.makeBuffer(length: 1024 * MemoryLayout<UInt32>.size, options: .storageModePrivate)!
        self.fBuffer = device.makeBuffer(length: 1024 * MemoryLayout<UInt32>.size, options: .storageModePrivate)!
    }
    
    var maxLength: UInt32 {
        return UInt32(eBuffer.length / MemoryLayout<UInt32>.size)
    }
    
    func setMaxLength(_ maxLength: UInt32) {
        let requiredSize = Int(maxLength) * MemoryLayout<UInt32>.size
        if eBuffer.length != requiredSize {
            eBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)!
            fBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)!
        }
    }
    
    func encodeSplitTo(_ commandBuffer: MTLCommandBuffer,
                       input: MTLBuffer,
                       inputIndices: MTLBuffer,
                       output: MTLBuffer,
                       outputIndices: MTLBuffer,
                       bit: UInt32,
                       length: UInt32) {
        
        assert(length <= maxLength, "Length exceeds maximum supported length")
        
        var bitVar = bit
        var lengthVar = length
        
        // Step 1: Prepare split data
        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.setComputePipelineState(prepPipeline)
            compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
            compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
            compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.count.rawValue)
            
            let threadgroupSize = MTLSize(width: 512, height: 1, depth: 1)
            let gridSize = MTLSize(width: Int(length), height: 1, depth: 1)
            compute.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            compute.endEncoding()
        }
        
        // Step 2: Scan the split data
        scanKernel.encodeScanTo(commandBuffer, input: eBuffer, output: fBuffer, length: length)
        
        // Step 3: Scatter based on split
        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.setComputePipelineState(scatterPipeline)
            compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
            compute.setBuffer(inputIndices, offset: 0, index: SplitBufferIndex.inputIndices.rawValue)
            compute.setBuffer(output, offset: 0, index: SplitBufferIndex.output.rawValue)
            compute.setBuffer(outputIndices, offset: 0, index: SplitBufferIndex.outputIndices.rawValue)
            compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
            compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
            compute.setBuffer(fBuffer, offset: 0, index: SplitBufferIndex.f.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.count.rawValue)
            
            let threadgroupSize = MTLSize(width: 512, height: 1, depth: 1)
            let gridSize = MTLSize(width: Int(length), height: 1, depth: 1)
            compute.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            compute.endEncoding()
        }
    }
}
