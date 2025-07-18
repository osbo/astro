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
        commandBuffer.pushDebugGroup("RadixSort")
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
            if length > 0 {
                blit.copy(from: keysIn, sourceOffset: 0, to: output, destinationOffset: 0, size: Int(length) * MemoryLayout<UInt64>.size)
                blit.copy(from: indicesIn, sourceOffset: 0, to: outputIndices, destinationOffset: 0, size: Int(length) * MemoryLayout<UInt32>.size)
            }
            blit.endEncoding()
        }
        commandBuffer.popDebugGroup()
    }
    
    func sortWithBufferLength(commandBuffer: MTLCommandBuffer,
              input: MTLBuffer,
              inputIndices: MTLBuffer,
              output: MTLBuffer,
              outputIndices: MTLBuffer,
              lengthBuffer: MTLBuffer) {
        // Validate buffer sizes (assume max possible size)
        let maxLength = maxLengthFromBuffers(input: input, inputIndices: inputIndices, output: output, outputIndices: outputIndices)
        splitKernel.setMaxLength(maxLength)
        var keysIn = input
        var keysOut = output
        var indicesIn = inputIndices
        var indicesOut = outputIndices
        for i in 0..<64 {
            splitKernel.encodeSplitToWithBufferLength(commandBuffer,
                                     input: keysIn,
                                     inputIndices: indicesIn,
                                     output: keysOut,
                                     outputIndices: indicesOut,
                                     bit: UInt32(i),
                                     lengthBuffer: lengthBuffer)
            if let compute = commandBuffer.makeComputeCommandEncoder() {
                compute.memoryBarrier(resources: [keysOut, indicesOut])
                compute.endEncoding()
            }
            swap(&keysIn, &keysOut)
            swap(&indicesIn, &indicesOut)
        }
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            let length = lengthFromBuffer(lengthBuffer)
            if length > 0 {
                blit.copy(from: keysIn, sourceOffset: 0, to: output, destinationOffset: 0, size: Int(length) * MemoryLayout<UInt64>.size)
                blit.copy(from: indicesIn, sourceOffset: 0, to: outputIndices, destinationOffset: 0, size: Int(length) * MemoryLayout<UInt32>.size)
            }
            blit.endEncoding()
        }
        commandBuffer.popDebugGroup()
    }
    private func maxLengthFromBuffers(input: MTLBuffer, inputIndices: MTLBuffer, output: MTLBuffer, outputIndices: MTLBuffer) -> UInt32 {
        return UInt32(min(input.length / MemoryLayout<UInt64>.size, inputIndices.length / MemoryLayout<UInt32>.size, output.length / MemoryLayout<UInt64>.size, outputIndices.length / MemoryLayout<UInt32>.size))
    }
    private func lengthFromBuffer(_ buffer: MTLBuffer) -> UInt32 {
        // This is only safe if the buffer is .shared and the value is up to date
        return buffer.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
    }
}

// MARK: - Scan Kernel

class ScanKernel {
    private let device: MTLDevice
    private let prefixSumPipeline: MTLComputePipelineState
    private let prefixFixupPipeline: MTLComputePipelineState
    private let scanThreadgroupsPipeline: MTLComputePipelineState
    private var auxBuffer: MTLBuffer
    private var aux2Buffer: MTLBuffer
    private var auxBufferSize: Int = 0
    private var aux2BufferSize: Int = 0
    
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
        // Pre-allocate with a reasonable default size
        self.auxBuffer = device.makeBuffer(length: 4096 * MemoryLayout<UInt32>.size, options: .storageModePrivate)!
        self.aux2Buffer = device.makeBuffer(length: 4096 * MemoryLayout<UInt32>.size, options: .storageModePrivate)!
        self.auxBufferSize = 4096 * MemoryLayout<UInt32>.size
        self.aux2BufferSize = 4096 * MemoryLayout<UInt32>.size
    }
    
    func setMaxLength(_ maxLength: UInt32) {
        let threadgroupSize = 512
        let numThreadgroups = max(1, (Int(maxLength) + (threadgroupSize * 2) - 1) / (threadgroupSize * 2))
        let requiredAuxSize = numThreadgroups * MemoryLayout<UInt32>.size
        if auxBuffer.length < requiredAuxSize {
            auxBuffer = device.makeBuffer(length: requiredAuxSize, options: .storageModePrivate)!
            auxBufferSize = requiredAuxSize
        }
        if aux2Buffer.length < requiredAuxSize {
            aux2Buffer = device.makeBuffer(length: requiredAuxSize, options: .storageModePrivate)!
            aux2BufferSize = requiredAuxSize
        }
    }
    
    func encodeScanTo(_ commandBuffer: MTLCommandBuffer,
                      input: MTLBuffer,
                      output: MTLBuffer,
                      length: UInt32) {
        guard length > 0 else { return }
        let threadgroupSize = 512
        var numThreadgroups = (Int(length) + (threadgroupSize * 2) - 1) / (threadgroupSize * 2)
        if numThreadgroups == 0 {
            numThreadgroups = 1
        }
        var lengthVar = length
        var zeroff: UInt32 = 1
        if numThreadgroups <= 1 {
            if let compute = commandBuffer.makeComputeCommandEncoder() {
                compute.label = "Scan Operation"
                compute.setComputePipelineState(prefixSumPipeline)
                compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
                compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
                compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
                compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
                compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
                compute.endEncoding()
            }
        } else {
            // 1. First pass: block-wise scan
            if let compute = commandBuffer.makeComputeCommandEncoder() {
                compute.label = "Scan Operation"
                compute.setComputePipelineState(prefixSumPipeline)
                compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
                compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
                compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
                compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
                compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
                compute.endEncoding()
            }
            // 2. Second pass: scan the auxiliary buffer
            var auxLength = UInt32(numThreadgroups)
            if let compute = commandBuffer.makeComputeCommandEncoder() {
                compute.label = "Scan Operation Aux"
                compute.setComputePipelineState(prefixSumPipeline)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.input.rawValue)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.output.rawValue)
                compute.setBuffer(aux2Buffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
                compute.setBytes(&auxLength, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
                compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
                compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
                compute.endEncoding()
            }
            // 3. Third pass: fix up the main buffer
            if let compute = commandBuffer.makeComputeCommandEncoder() {
                compute.label = "Scan Operation Fixup"
                compute.setComputePipelineState(prefixFixupPipeline)
                compute.setBuffer(output, offset: 0, index: ScanBufferIndex.input.rawValue)
                compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
                compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
                compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
                compute.endEncoding()
            }
        }
    }
}

// MARK: - Split Kernel

class SplitKernel {
    private let device: MTLDevice
    private let prepPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let prepPipelineDynamic: MTLComputePipelineState
    private let scatterPipelineDynamic: MTLComputePipelineState
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
            self.prepPipelineDynamic = try device.makeComputePipelineState(function: library.makeFunction(name: "split_prep_dynamic")!)
            self.scatterPipelineDynamic = try device.makeComputePipelineState(function: library.makeFunction(name: "split_scatter_dynamic")!)
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
    
    func encodeSplitToWithBufferLength(_ commandBuffer: MTLCommandBuffer,
                       input: MTLBuffer,
                       inputIndices: MTLBuffer,
                       output: MTLBuffer,
                       outputIndices: MTLBuffer,
                       bit: UInt32,
                       lengthBuffer: MTLBuffer) {
        var bitVar = bit
        // Step 1: Prepare split data (dynamic count)
        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.setComputePipelineState(prepPipelineDynamic)
            compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
            compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
            compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
            compute.setBuffer(lengthBuffer, offset: 0, index: SplitBufferIndex.count.rawValue)
            let maxLength = self.maxLength
            let threadgroupSize = MTLSize(width: 512, height: 1, depth: 1)
            let gridSize = MTLSize(width: Int(maxLength), height: 1, depth: 1)
            compute.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            compute.endEncoding()
        }
        scanKernel.encodeScanTo(commandBuffer, input: eBuffer, output: fBuffer, length: self.maxLength)
        // Step 3: Scatter based on split (dynamic count)
        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.setComputePipelineState(scatterPipelineDynamic)
            compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
            compute.setBuffer(inputIndices, offset: 0, index: SplitBufferIndex.inputIndices.rawValue)
            compute.setBuffer(output, offset: 0, index: SplitBufferIndex.output.rawValue)
            compute.setBuffer(outputIndices, offset: 0, index: SplitBufferIndex.outputIndices.rawValue)
            compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
            compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
            compute.setBuffer(fBuffer, offset: 0, index: SplitBufferIndex.f.rawValue)
            compute.setBuffer(lengthBuffer, offset: 0, index: SplitBufferIndex.count.rawValue)
            let maxLength = self.maxLength
            let threadgroupSize = MTLSize(width: 512, height: 1, depth: 1)
            let gridSize = MTLSize(width: Int(maxLength), height: 1, depth: 1)
            compute.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            compute.endEncoding()
        }
    }
}
