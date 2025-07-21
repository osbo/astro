import Metal
import Foundation

// Swift equivalents of the Metal enums
fileprivate enum ScanBufferIndex: Int {
    case input = 0
    case output = 1
    case aux = 2
    case length = 3
    case zeroff = 4
}

fileprivate enum SplitBufferIndex: Int {
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

    // Intermediate buffers for ping-ponging during the sort
    private var tempKeys: MTLBuffer!
    private var tempIndices: MTLBuffer!
    private var copyKeysPipeline: MTLComputePipelineState!
    private var copyIndicesPipeline: MTLComputePipelineState!
    private var clearBuffer64Pipeline: MTLComputePipelineState!
    private var clearBuffer32Pipeline: MTLComputePipelineState!

    init(device: MTLDevice) {
        self.device = device
        self.scanKernel = ScanKernel(device: device)
        self.splitKernel = SplitKernel(device: device)
        // Create copy pipelines
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to get default Metal library for copy pipelines")
        }
        do {
            self.copyKeysPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "copyBuffer")!)
            self.copyIndicesPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "copyIndices")!)
            self.clearBuffer64Pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "clearBuffer64")!)
            self.clearBuffer32Pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "clearBuffer")!)
        } catch {
            fatalError("Failed to create copy/clear pipelines: \(error)")
        }
    }

    private func clearInternalBuffers(commandBuffer: MTLCommandBuffer) {
        // Clear tempKeys (UInt64)
        if let tempKeys = tempKeys {
            let count = tempKeys.length / MemoryLayout<UInt64>.stride
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Clear TempKeys Buffer"
                encoder.setComputePipelineState(clearBuffer64Pipeline)
                encoder.setBuffer(tempKeys, offset: 0, index: 0)
                let tgSize = MTLSize(width: 64, height: 1, depth: 1)
                let tgCount = MTLSize(width: (count + 63) / 64, height: 1, depth: 1)
                encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                encoder.endEncoding()
            }
        }
        // Clear tempIndices (UInt32)
        if let tempIndices = tempIndices {
            let count = tempIndices.length / MemoryLayout<UInt32>.stride
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Clear TempIndices Buffer"
                encoder.setComputePipelineState(clearBuffer32Pipeline)
                encoder.setBuffer(tempIndices, offset: 0, index: 0)
                let tgSize = MTLSize(width: 64, height: 1, depth: 1)
                let tgCount = MTLSize(width: (count + 63) / 64, height: 1, depth: 1)
                encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                encoder.endEncoding()
            }
        }
    }

    func setMaxLength(_ maxLength: UInt32) {
        let requiredKeySize = Int(maxLength) * MemoryLayout<UInt64>.stride
        let requiredIndexSize = Int(maxLength) * MemoryLayout<UInt32>.stride

        if tempKeys == nil || tempKeys.length < requiredKeySize {
            tempKeys = device.makeBuffer(length: requiredKeySize, options: .storageModePrivate)
            tempKeys.label = "Radix Sort Temp Keys"
        }
        if tempIndices == nil || tempIndices.length < requiredIndexSize {
            tempIndices = device.makeBuffer(length: requiredIndexSize, options: .storageModePrivate)
            tempIndices.label = "Radix Sort Temp Indices"
        }

        splitKernel.setMaxLength(maxLength)
        scanKernel.setMaxLength(maxLength)
    }

    func sort(commandBuffer: MTLCommandBuffer,
              input: MTLBuffer,
              inputIndices: MTLBuffer,
              output: MTLBuffer,
              outputIndices: MTLBuffer,
              estimatedLength: UInt32,
              actualCountBuffer: MTLBuffer) {

        guard estimatedLength > 0 else { return }

        setMaxLength(estimatedLength)

        // --- Clear tempKeys and tempIndices before sort ---
        clearInternalBuffers(commandBuffer: commandBuffer)
        // --- End clear ---

        var keysIn = input
        var keysOut = tempKeys!
        var indicesIn = inputIndices
        var indicesOut = tempIndices!

        guard let compute = commandBuffer.makeComputeCommandEncoder() else { return }
        compute.label = "Radix Sort Pass"

        for i in 0..<64 {
            splitKernel.encodeSplitTo(compute,
                                      input: keysIn,
                                      inputIndices: indicesIn,
                                      output: keysOut,
                                      outputIndices: indicesOut,
                                      bit: UInt32(i),
                                      estimatedLength: estimatedLength,
                                      actualCountBuffer: actualCountBuffer)

            compute.memoryBarrier(resources: [keysOut, indicesOut])

            swap(&keysIn, &keysOut)
            swap(&indicesIn, &indicesOut)
        }

        compute.endEncoding()

        // Copy final results to output buffers using compute kernels
        guard let copyEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        copyEncoder.label = "Radix Sort Final Copy"
        // Copy keys
        copyEncoder.setComputePipelineState(copyKeysPipeline)
        copyEncoder.setBuffer(keysIn, offset: 0, index: 0)
        copyEncoder.setBuffer(output, offset: 0, index: 1)
        copyEncoder.setBuffer(actualCountBuffer, offset: 0, index: 2) // Pass mortonCodeCountBuffer as count buffer
        let threadGroupSize = MTLSize(width: 512, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (Int(estimatedLength) + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
        copyEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        // Copy indices
        copyEncoder.setComputePipelineState(copyIndicesPipeline)
        copyEncoder.setBuffer(indicesIn, offset: 0, index: 0)
        copyEncoder.setBuffer(outputIndices, offset: 0, index: 1)
        copyEncoder.setBuffer(actualCountBuffer, offset: 0, index: 2) // Pass mortonCodeCountBuffer as count buffer
        copyEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        copyEncoder.endEncoding()
    }
}

// MARK: - Scan Kernel

fileprivate class ScanKernel {
    private let device: MTLDevice
    private let prefixSumPipeline: MTLComputePipelineState
    private let prefixFixupPipeline: MTLComputePipelineState

    private var auxBuffer: MTLBuffer!
    private var aux2Buffer: MTLBuffer!
    private var auxSmallBuffer: MTLBuffer!

    init(device: MTLDevice) {
        self.device = device
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to get default Metal library for ScanKernel")
        }

        do {
            self.prefixSumPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "prefixSum")!)
            self.prefixFixupPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "prefixFixup")!)
        } catch {
            fatalError("Failed to create scan pipelines: \(error)")
        }
    }

    func setMaxLength(_ maxLength: UInt32) {
        let threadgroupSize = 512
        // Ensure maxLength is at least 1 to avoid creating a zero-sized buffer
        let adjustedLength = max(1, maxLength)
        let numThreadgroups = (Int(adjustedLength) + (threadgroupSize * 2) - 1) / (threadgroupSize * 2)
        let requiredAuxSize = max(1, numThreadgroups) * MemoryLayout<UInt32>.size

        if auxBuffer == nil || auxBuffer.length < requiredAuxSize {
            auxBuffer = device.makeBuffer(length: requiredAuxSize, options: .storageModePrivate)
            auxBuffer.label = "Scan Aux Buffer"
        }

        // aux2Buffer needs to be large enough to hold the scanned results of auxBuffer
        let requiredAux2Size = requiredAuxSize
        if aux2Buffer == nil || aux2Buffer.length < requiredAux2Size {
            aux2Buffer = device.makeBuffer(length: requiredAux2Size, options: .storageModePrivate)
            aux2Buffer.label = "Scan Aux2 Buffer"
        }

        // auxSmallBuffer for single-threadgroup scans
        let requiredAuxSmallSize = MemoryLayout<UInt32>.size
        if auxSmallBuffer == nil || auxSmallBuffer.length < requiredAuxSmallSize {
            auxSmallBuffer = device.makeBuffer(length: requiredAuxSmallSize, options: .storageModePrivate)
            auxSmallBuffer.label = "Scan Aux Small Buffer"
        }
    }

    func encodeScanTo(_ compute: MTLComputeCommandEncoder,
                      input: MTLBuffer,
                      output: MTLBuffer,
                      length: UInt32) {

        guard length > 0 else { return }

        let threadgroupSize = 512
        let numThreadgroups = (Int(length) + (threadgroupSize * 2) - 1) / (threadgroupSize * 2)

        var lengthVar = length
        var zeroff: UInt32 = 1

        if numThreadgroups <= 1 {
            compute.setComputePipelineState(prefixSumPipeline)
            compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
            compute.setBuffer(self.auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        } else {
            // Pass 1: Block-wise scan
            compute.setComputePipelineState(prefixSumPipeline)
            compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
            compute.setBuffer(self.auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            
            // Pass 2: Scan the auxiliary buffer
            var auxLength = UInt32(numThreadgroups)
            compute.setComputePipelineState(prefixSumPipeline)
            compute.setBuffer(self.auxBuffer, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(self.aux2Buffer, offset: 0, index: ScanBufferIndex.output.rawValue)
            compute.setBuffer(self.auxSmallBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&auxLength, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))

            // Pass 3: Fix up the main buffer
            compute.setComputePipelineState(prefixFixupPipeline)
            compute.setBuffer(output, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(self.aux2Buffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        }
    }
}

// MARK: - Split Kernel

fileprivate class SplitKernel {
    private let device: MTLDevice
    private let prepPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let scanKernel: ScanKernel
    private var eBuffer: MTLBuffer!
    private var fBuffer: MTLBuffer!

    init(device: MTLDevice) {
        self.device = device
        self.scanKernel = ScanKernel(device: device)
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to get default Metal library for SplitKernel")
        }

        do {
            self.prepPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "split_prep")!)
            self.scatterPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "split_scatter")!)
        } catch {
            fatalError("Failed to create split pipelines: \(error)")
        }
    }

    func setMaxLength(_ maxLength: UInt32) {
        let requiredSize = Int(max(1, maxLength)) * MemoryLayout<UInt32>.stride
        if eBuffer == nil || eBuffer.length < requiredSize {
            eBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)
            fBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)
            eBuffer.label = "Radix eBuffer"
            fBuffer.label = "Radix fBuffer"
        }
        
        // Ensure the scan kernel is also initialized with the same max length
        scanKernel.setMaxLength(maxLength)
    }

    func encodeSplitTo(_ compute: MTLComputeCommandEncoder,
                       input: MTLBuffer,
                       inputIndices: MTLBuffer,
                       output: MTLBuffer,
                       outputIndices: MTLBuffer,
                       bit: UInt32,
                       estimatedLength: UInt32,
                       actualCountBuffer: MTLBuffer) {

        var bitVar = bit

        // Step 1: Prepare split data
        compute.setComputePipelineState(prepPipeline)
        compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
        compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
        compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
        compute.setBuffer(actualCountBuffer, offset: 0, index: SplitBufferIndex.count.rawValue)

        let threadgroupSize = MTLSize(width: 512, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (Int(estimatedLength) + threadgroupSize.width - 1) / threadgroupSize.width, height: 1, depth: 1)
        compute.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)

        // Step 2: Scan the e-buffer to get the f-buffer
        scanKernel.encodeScanTo(compute, input: eBuffer, output: fBuffer, length: estimatedLength)

        // Step 3: Scatter based on the split information
        compute.setComputePipelineState(scatterPipeline)
        compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
        compute.setBuffer(inputIndices, offset: 0, index: SplitBufferIndex.inputIndices.rawValue)
        compute.setBuffer(output, offset: 0, index: SplitBufferIndex.output.rawValue)
        compute.setBuffer(outputIndices, offset: 0, index: SplitBufferIndex.outputIndices.rawValue)
        compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
        compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
        compute.setBuffer(fBuffer, offset: 0, index: SplitBufferIndex.f.rawValue)
        compute.setBuffer(actualCountBuffer, offset: 0, index: SplitBufferIndex.count.rawValue)

        compute.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
    }
}
