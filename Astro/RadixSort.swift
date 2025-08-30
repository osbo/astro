import Metal
import Foundation


public enum ScanBufferIndex: Int {
    case input = 0
    case output = 1
    case aux = 2
    case length = 3
    case zeroff = 4
}

public enum SplitBufferIndex: Int {
    case input = 0
    case inputIndices = 1
    case output = 2
    case outputIndices = 3
    case count = 4
    case bit = 5
    case e = 6
    case f = 7
}

public class MetalKernelsRadixSort {
    public let device: MTLDevice
    public let scanKernel: ScanKernel
    public let splitKernel: SplitKernel
    public let maxLength: UInt32


    public var tempKeys: MTLBuffer!
    public var tempIndices: MTLBuffer!
    public var tempKeysB: MTLBuffer!
    public var tempIndicesB: MTLBuffer!
    public var copyKeysPipeline: MTLComputePipelineState!
    public var copyIndicesPipeline: MTLComputePipelineState!
    public var clearBuffer64Pipeline: MTLComputePipelineState!
    public var clearBuffer32Pipeline: MTLComputePipelineState!

    public init(device: MTLDevice, maxLength: UInt32) {
        self.device = device
        self.maxLength = maxLength
        self.scanKernel = ScanKernel(device: device, maxLength: maxLength)
        self.splitKernel = SplitKernel(device: device, maxLength: maxLength)

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
        let requiredKeySize = Int(maxLength) * MemoryLayout<UInt64>.stride
        let requiredIndexSize = Int(maxLength) * MemoryLayout<UInt32>.stride
        tempKeys = device.makeBuffer(length: requiredKeySize, options: .storageModePrivate)
        tempKeys.label = "Radix Sort Temp Keys"
        tempKeysB = device.makeBuffer(length: requiredKeySize, options: .storageModePrivate)
        tempKeysB.label = "Radix Sort Temp Keys B"
        tempIndices = device.makeBuffer(length: requiredIndexSize, options: .storageModePrivate)
        tempIndices.label = "Radix Sort Temp Indices"
        tempIndicesB = device.makeBuffer(length: requiredIndexSize, options: .storageModePrivate)
        tempIndicesB.label = "Radix Sort Temp Indices B"
    }

    public func clearInternalBuffers(commandBuffer: MTLCommandBuffer) {

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

    public func sort(commandBuffer: MTLCommandBuffer,
              input: MTLBuffer,
              inputIndices: MTLBuffer,
              output: MTLBuffer,
              outputIndices: MTLBuffer,
              estimatedLength: UInt32,
              actualCountBuffer: MTLBuffer) {

        guard estimatedLength > 0 else { return }

        let clearBuffers: [(MTLBuffer?, MTLComputePipelineState, Int)] = [
            (tempKeys, clearBuffer64Pipeline, Int(maxLength)),
            (tempKeysB, clearBuffer64Pipeline, Int(maxLength)),
            (tempIndices, clearBuffer32Pipeline, Int(maxLength)),
            (tempIndicesB, clearBuffer32Pipeline, Int(maxLength))
        ]
        for (buffer, pipeline, count) in clearBuffers {
            if let buffer = buffer, let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "Clear Internal Buffer"
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(buffer, offset: 0, index: 0)
                let tgSize = MTLSize(width: 64, height: 1, depth: 1)
                let tgCount = MTLSize(width: (count + 63) / 64, height: 1, depth: 1)
                encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
                encoder.endEncoding()
            }
        }


        if let copyEncoder = commandBuffer.makeComputeCommandEncoder() {
            copyEncoder.label = "Radix Sort Initial Copy"
            copyEncoder.setComputePipelineState(copyKeysPipeline)
            copyEncoder.setBuffer(input, offset: 0, index: 0)
            copyEncoder.setBuffer(tempKeys, offset: 0, index: 1)
            copyEncoder.setBuffer(actualCountBuffer, offset: 0, index: 2)
            let threadGroupSize = MTLSize(width: 512, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (Int(estimatedLength) + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
            copyEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            copyEncoder.setComputePipelineState(copyIndicesPipeline)
            copyEncoder.setBuffer(inputIndices, offset: 0, index: 0)
            copyEncoder.setBuffer(tempIndices, offset: 0, index: 1)
            copyEncoder.setBuffer(actualCountBuffer, offset: 0, index: 2)
            copyEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            copyEncoder.endEncoding()
        }


        var keysIn = tempKeys!
        var keysOut = tempKeysB!
        var indicesIn = tempIndices!
        var indicesOut = tempIndicesB!

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


        if let copyEncoder = commandBuffer.makeComputeCommandEncoder() {
            copyEncoder.label = "Radix Sort Final Copy"
            copyEncoder.setComputePipelineState(copyKeysPipeline)
            copyEncoder.setBuffer(keysIn, offset: 0, index: 0)
            copyEncoder.setBuffer(output, offset: 0, index: 1)
            copyEncoder.setBuffer(actualCountBuffer, offset: 0, index: 2)
            let threadGroupSize = MTLSize(width: 512, height: 1, depth: 1)
            let threadGroups = MTLSize(width: (Int(estimatedLength) + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
            copyEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            copyEncoder.setComputePipelineState(copyIndicesPipeline)
            copyEncoder.setBuffer(indicesIn, offset: 0, index: 0)
            copyEncoder.setBuffer(outputIndices, offset: 0, index: 1)
            copyEncoder.setBuffer(actualCountBuffer, offset: 0, index: 2)
            copyEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            copyEncoder.endEncoding()
        }
    }
}



public class ScanKernel {
    public let device: MTLDevice
    public let prefixSumPipeline: MTLComputePipelineState
    public let prefixFixupPipeline: MTLComputePipelineState

    public var auxBuffer: MTLBuffer!
    public var aux2Buffer: MTLBuffer!
    public var auxSmallBuffer: MTLBuffer!

    public init(device: MTLDevice, maxLength: UInt32) {
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
        let threadgroupSize = 512
        let adjustedLength = max(1, maxLength)
        let numThreadgroups = (Int(adjustedLength) + (threadgroupSize * 2) - 1) / (threadgroupSize * 2)
        let requiredAuxSize = max(1, numThreadgroups) * MemoryLayout<UInt32>.size
        auxBuffer = device.makeBuffer(length: requiredAuxSize, options: .storageModePrivate)
        auxBuffer.label = "Scan Aux Buffer"
        let requiredAux2Size = requiredAuxSize
        aux2Buffer = device.makeBuffer(length: requiredAux2Size, options: .storageModePrivate)
        aux2Buffer.label = "Scan Aux2 Buffer"
        let requiredAuxSmallSize = MemoryLayout<UInt32>.size
        auxSmallBuffer = device.makeBuffer(length: requiredAuxSmallSize, options: .storageModePrivate)
        auxSmallBuffer.label = "Scan Aux Small Buffer"
    }

    public func encodeScanTo(_ compute: MTLComputeCommandEncoder,
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

            compute.setComputePipelineState(prefixSumPipeline)
            compute.setBuffer(input, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(output, offset: 0, index: ScanBufferIndex.output.rawValue)
            compute.setBuffer(self.auxBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            compute.memoryBarrier(resources: [output, self.auxBuffer])
            

            var auxLength = UInt32(numThreadgroups)
            compute.setComputePipelineState(prefixSumPipeline)
            compute.setBuffer(self.auxBuffer, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(self.aux2Buffer, offset: 0, index: ScanBufferIndex.output.rawValue)
            compute.setBuffer(self.auxSmallBuffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&auxLength, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.setBytes(&zeroff, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.zeroff.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
            compute.memoryBarrier(resources: [self.aux2Buffer])


            compute.setComputePipelineState(prefixFixupPipeline)
            compute.setBuffer(output, offset: 0, index: ScanBufferIndex.input.rawValue)
            compute.setBuffer(self.aux2Buffer, offset: 0, index: ScanBufferIndex.aux.rawValue)
            compute.setBytes(&lengthVar, length: MemoryLayout<UInt32>.size, index: ScanBufferIndex.length.rawValue)
            compute.dispatchThreadgroups(MTLSize(width: numThreadgroups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        }
    }
}



public class SplitKernel {
    public let device: MTLDevice
    public let prepPipeline: MTLComputePipelineState
    public let scatterPipeline: MTLComputePipelineState
    public let scanKernel: ScanKernel
    public var eBuffer: MTLBuffer!
    public var fBuffer: MTLBuffer!

    public init(device: MTLDevice, maxLength: UInt32) {
        self.device = device
        self.scanKernel = ScanKernel(device: device, maxLength: maxLength)
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to get default Metal library for SplitKernel")
        }

        do {
            self.prepPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "split_prep")!)
            self.scatterPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "split_scatter")!)
        } catch {
            fatalError("Failed to create split pipelines: \(error)")
        }
        let requiredSize = Int(max(1, maxLength)) * MemoryLayout<UInt32>.stride
        eBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)
        fBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)
        eBuffer.label = "Radix eBuffer"
        fBuffer.label = "Radix fBuffer"
    }

    public func encodeSplitTo(_ compute: MTLComputeCommandEncoder,
                       input: MTLBuffer,
                       inputIndices: MTLBuffer,
                       output: MTLBuffer,
                       outputIndices: MTLBuffer,
                       bit: UInt32,
                       estimatedLength: UInt32,
                       actualCountBuffer: MTLBuffer) {

        var bitVar = bit


        compute.setComputePipelineState(prepPipeline)
        compute.setBuffer(input, offset: 0, index: SplitBufferIndex.input.rawValue)
        compute.setBytes(&bitVar, length: MemoryLayout<UInt32>.size, index: SplitBufferIndex.bit.rawValue)
        compute.setBuffer(eBuffer, offset: 0, index: SplitBufferIndex.e.rawValue)
        compute.setBuffer(actualCountBuffer, offset: 0, index: SplitBufferIndex.count.rawValue)

        let threadgroupSize = MTLSize(width: 512, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (Int(estimatedLength) + threadgroupSize.width - 1) / threadgroupSize.width, height: 1, depth: 1)
        compute.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)


        scanKernel.encodeScanTo(compute, input: eBuffer, output: fBuffer, length: estimatedLength)


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
