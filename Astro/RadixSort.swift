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
    
    func sort(computeEncoder: MTLComputeCommandEncoder,
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
            computeEncoder.pushDebugGroup("RadixSort Bit \(i)")
            splitKernel.encodeSplitTo(computeEncoder,
                                     input: keysIn,
                                     inputIndices: indicesIn,
                                     output: keysOut,
                                     outputIndices: indicesOut,
                                     bit: UInt32(i),
                                     length: length)
            
            computeEncoder.memoryBarrier(resources: [keysOut, indicesOut])
            
            // Swap buffers for next iteration
            swap(&keysIn, &keysOut)
            swap(&indicesIn, &indicesOut)
            computeEncoder.popDebugGroup()
        }
    }
    
    func sortWithBufferLength(computeEncoder: MTLComputeCommandEncoder,
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
            computeEncoder.pushDebugGroup("RadixSort Bit \(i)")
            splitKernel.encodeSplitToWithBufferLength(computeEncoder,
                                     input: keysIn,
                                     inputIndices: indicesIn,
                                     output: keysOut,
                                     outputIndices: indicesOut,
                                     bit: UInt32(i),
                                     lengthBuffer: lengthBuffer)

            computeEncoder.memoryBarrier(resources: [keysOut, indicesOut])

            swap(&keysIn, &keysOut)
            swap(&indicesIn, &indicesOut)
            computeEncoder.popDebugGroup()
        }
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
    
    func encodeScanTo(_ compute: MTLComputeCommandEncoder,
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
            compute.setComputePipelineState(prefixSumPipeline)
            compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
            compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        } else {
            // 1. First pass: block-wise scan
            compute.setComputePipelineState(prefixSumPipeline)
            compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
            compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            
            compute.memoryBarrier(resources: [auxBuffer])

            // 2. Second pass: scan the auxiliary buffer
            var auxLength = UInt32(numThreadgroups)
            compute.setComputePipelineState(prefixSumPipeline)
            compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.output.rawValue)
            compute.setBuffer(aux2Buffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&auxLength, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))

            compute.memoryBarrier(resources: [auxBuffer, aux2Buffer])

            // 3. Third pass: fix up the main buffer
            compute.setComputePipelineState(prefixFixupPipeline)
            compute.setBuffer(output, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
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
    
    func encodeSplitTo(_ compute: MTLComputeCommandEncoder,
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
        compute.setComputePipelineState(prepPipeline)
        compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
        compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
        compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
        compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.count.rawValue)
        
        let threadgroupSize = MTLSize(width: 512, height: 1, depth: 1)
        let gridSize = MTLSize(width: Int(length), height: 1, depth: 1)
        compute.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        compute.memoryBarrier(resources: [eBuffer])
        
        // Step 2: Scan the split data
        scanKernel.encodeScanTo(compute, input: eBuffer, output: fBuffer, length: length)

        compute.memoryBarrier(resources: [fBuffer])
        
        // Step 3: Scatter based on split
        compute.setComputePipelineState(scatterPipeline)
        compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
        compute.setBuffer(inputIndices, offset: 0, index: SplitBufferIndex.inputIndices.rawValue)
        compute.setBuffer(output, offset: 0, index: SplitBufferIndex.output.rawValue)
        compute.setBuffer(outputIndices, offset: 0, index: SplitBufferIndex.outputIndices.rawValue)
        compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
        compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
        compute.setBuffer(fBuffer, offset: 0, index: SplitBufferIndex.f.rawValue)
        compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.count.rawValue)
        
        compute.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
    
    func encodeSplitToWithBufferLength(_ compute: MTLComputeCommandEncoder,
                       input: MTLBuffer,
                       inputIndices: MTLBuffer,
                       output: MTLBuffer,
                       outputIndices: MTLBuffer,
                       bit: UInt32,
                       lengthBuffer: MTLBuffer) {
        var bitVar = bit
        // Step 1: Prepare split data (dynamic count)
        compute.setComputePipelineState(prepPipelineDynamic)
        compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
        compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
        compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
        compute.setBuffer(lengthBuffer, offset: 0, index: SplitBufferIndex.count.rawValue)
        let maxLength = self.maxLength
        let threadgroupSize = MTLSize(width: 512, height: 1, depth: 1)
        let gridSize = MTLSize(width: Int(maxLength), height: 1, depth: 1)
        compute.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        compute.memoryBarrier(resources: [eBuffer])

        scanKernel.encodeScanTo(compute, input: eBuffer, output: fBuffer, length: self.maxLength)

        compute.memoryBarrier(resources: [fBuffer])

        // Step 3: Scatter based on split (dynamic count)
        compute.setComputePipelineState(scatterPipelineDynamic)
        compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
        compute.setBuffer(inputIndices, offset: 0, index: SplitBufferIndex.inputIndices.rawValue)
        compute.setBuffer(output, offset: 0, index: SplitBufferIndex.output.rawValue)
        compute.setBuffer(outputIndices, offset: 0, index: SplitBufferIndex.outputIndices.rawValue)
        compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
        compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
        compute.setBuffer(fBuffer, offset: 0, index: SplitBufferIndex.f.rawValue)
        compute.setBuffer(lengthBuffer, offset: 0, index: SplitBufferIndex.count.rawValue)
        compute.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
}
