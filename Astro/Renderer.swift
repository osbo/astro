import MetalKit
import MetalPerformanceShaders
import simd

class Renderer: NSObject, MTKViewDelegate {
    var parent: ContentView
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var depthStencilState: MTLDepthStencilState!
    var sceneTexture: MTLTexture!
    var depthTexture: MTLTexture! // Add this line
    var currentDrawableSize: CGSize = .zero
    var textureSampler: MTLSamplerState!
    
    var camera = Camera()
    var lastFrameTime = CFAbsoluteTimeGetCurrent()
    var currentTime: Float = 0.0
    
    // --- Simulation Parameters ---
    var usePostProcessing: Bool = false
    var sortThreshold: Int32 = 0
    var currentLayer: Int32 = 0
    var numStars: Int32 = 3
    var numPlanets: Int32 = 100
    var numDust: Int32 = 100000
    
    var numSpheres: Int32 {
        return numStars + numPlanets + numDust
    }
    var sphereCount: Int {
        return Int(numStars) + Int(numPlanets) + Int(numDust)
    }
    
    var starRadius: Float = 50000.0
    var planetRadius: Float = 10000.0
    var dustRadius: Float = 1
    var starMass: Float = 1000000
    var planetMass: Float = 10000
    var dustMass: Float = 100
    var initialVelocityMaximum: Float = 5000
    var spawnBounds: Float = 1000000
    var backgroundColor: MTLClearColor = MTLClearColor(red: 1/255, green: 1/255, blue: 2/255, alpha: 1.0)
    
    // --- Metal Objects ---
    var starSphere: MTKMesh!
    var planetSphere: MTKMesh!
    
    var spherePipelineState: MTLRenderPipelineState!
    var dustPipelineState: MTLRenderPipelineState!
    var postProcessPipelineState: MTLRenderPipelineState!
    
    var globalUniformsBuffer: MTLBuffer!
    var positionMassBuffer: MTLBuffer!
    var velocityRadiusBuffer: MTLBuffer!
    var colorTypeBuffer: MTLBuffer!
    var unsortedMortonCodesBuffer: MTLBuffer!
    var unsortedIndicesBuffer: MTLBuffer!
    var sortedMortonCodesBuffer: MTLBuffer!
    var sortedIndicesBuffer: MTLBuffer!
    var uniqueIndicesBuffer: MTLBuffer!
    var mortonCodeCountBuffer: MTLBuffer!
    var layerCountBuffer: MTLBuffer!
    var octreeNodesBuffer: MTLBuffer!
    
    var pipelinesCreated = false
    var generateMortonCodesPipelineState: MTLComputePipelineState!
    var aggregateLeafNodesPipelineState: MTLComputePipelineState!
    var countUniqueMortonCodesPipelineState: MTLComputePipelineState!
    var aggregateNodesPipelineState: MTLComputePipelineState!
    var fillIndicesPipelineState: MTLComputePipelineState!
    var unsortedLeafPipelineState: MTLComputePipelineState!
    var unsortedInternalPipelineState: MTLComputePipelineState!
    var resetUIntBufferPipelineState: MTLComputePipelineState!
    var clearBufferPipelineState: MTLComputePipelineState!
    var clearBuffer64PipelineState: MTLComputePipelineState!
    var clearOctreeNodeBufferPipelineState: MTLComputePipelineState!
    
    var radixSorter: MetalKernelsRadixSort!
    
    // Floating origin: store world positions as double
    var doublePositions: [SIMD3<Double>] = []
    var positions: [PositionMass] = []
    var velocities: [VelocityRadius] = []
    var colors: [ColorType] = []
    
    init(_ parent: ContentView) {
        self.parent = parent
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal not supported") }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        let allocator = MTKMeshBufferAllocator(device: device)
        let starMDLSphere = MDLMesh(sphereWithExtent: [1, 1, 1], segments: [32, 32], inwardNormals: false, geometryType: .triangles, allocator: allocator)
        do {
            self.starSphere = try MTKMesh(mesh: starMDLSphere, device: device)
        } catch {
            fatalError("Could not create star MTKMesh: \(error)")
        }
        
        let planetMDLSphere = MDLMesh(sphereWithExtent: [1, 1, 1], segments: [16, 16], inwardNormals: false, geometryType: .triangles, allocator: allocator)
        do {
            self.planetSphere = try MTKMesh(mesh: planetMDLSphere, device: device)
        } catch {
            fatalError("Could not create planet MTKMesh: \(error)")
        }
        
        super.init()
        
        setupDepthStencilStates()
        createTextureSampler(device: device)
        radixSorter = MetalKernelsRadixSort(device: device)
        // Defer makeBuffers and setMaxLength until size is known
    }
    
    private func setupDepthStencilStates() {
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .less
        depthDesc.isDepthWriteEnabled = true
        self.depthStencilState = device.makeDepthStencilState(descriptor: depthDesc)!
    }
    
    func resetSimulation() {
        camera = Camera()
        makeSpheres()
        updateGlobalUniforms()
    }
    
    private func updateGlobalUniforms() {
        guard currentDrawableSize.width > 0 && currentDrawableSize.height > 0, let buffer = globalUniformsBuffer else { return }
        let pointer = buffer.contents().bindMemory(to: GlobalUniforms.self, capacity: 1)
        pointer[0].viewMatrix = camera.floatViewMatrix()
        pointer[0].cameraPosition = camera.floatPosition
        let aspect = Float(currentDrawableSize.width) / Float(currentDrawableSize.height)
        pointer[0].projectionMatrix = createProjectionMatrix(fov: .pi/2.5, aspect: aspect, near: 10.0, far: 10000000.0)
    }
    
    private func createTextureSampler(device: MTLDevice) {
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        textureSampler = device.makeSamplerState(descriptor: samplerDescriptor)
    }
    
    private func setupOffscreenTextures(size: CGSize) {
        if usePostProcessing {
            let fullResDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: Int(size.width), height: Int(size.height), mipmapped: false)
            fullResDesc.storageMode = .private
            fullResDesc.usage = [.renderTarget, .shaderRead]
            sceneTexture = device.makeTexture(descriptor: fullResDesc)
            sceneTexture.label = "Scene Texture"
        } else {
            sceneTexture = nil
        }
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: Int(size.width), height: Int(size.height), mipmapped: false)
        depthDesc.storageMode = .memoryless
        depthDesc.usage = [.renderTarget]
        depthTexture = device.makeTexture(descriptor: depthDesc)
        depthTexture.label = "Custom Depth Texture"
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        currentDrawableSize = size
        setupOffscreenTextures(size: size)
        if !pipelinesCreated {
            makePipelines(view: view)
            makeBuffers()
            makeSpheres()
            pipelinesCreated = true
        }
        updateGlobalUniforms()
    }
    
    func makePipelines(view: MTKView) {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not load default Metal library.")
        }
        
        // Your existing pipeline creation logic...
        guard let fragmentFunction = library.makeFunction(name: "fragment_shader") else {
            fatalError("Could not find fragment_shader function.")
        }
        guard let sphereVertexFunction = library.makeFunction(name: "sphere_vertex") else {
            fatalError("Could not find sphere_vertex function.")
        }
        guard let dustVertexFunction = library.makeFunction(name: "dust_point_vertex") else {
            fatalError("Could not find dust_point_vertex function.")
        }
        guard let postProcessVertexFunction = library.makeFunction(name: "postProcessVertex") else {
            fatalError("Could not find postProcessVertex function.")
        }
        guard let postProcessFragmentFunction = library.makeFunction(name: "postProcessFragment") else {
            fatalError("Could not find postProcessFragment function.")
        }
        guard let generateMortonCodesFunction = library.makeFunction(name: "generateMortonCodes") else {
            fatalError("Could not find generateMortonCodes function.")
        }
        guard let aggregateLeafNodesFunction = library.makeFunction(name: "aggregateLeafNodes") else {
            fatalError("Could not find aggregateLeafNodes function in Octree.metal.")
        }
        guard let countUniqueMortonCodesFunction = library.makeFunction(name: "countUniqueMortonCodes") else {
            fatalError("Could not find countUniqueMortonCodes function in Octree.metal.")
        }
        guard let aggregateNodesFunction = library.makeFunction(name: "aggregateNodes") else {
            fatalError("Could not find aggregateNodes function in Octree.metal.")
        }
        guard let fillIndicesFunction = library.makeFunction(name: "fillIndices") else {
            fatalError("Could not find fillIndices function in Octree.metal.")
        }
        guard let unsortedLeafFunction = library.makeFunction(name: "unsortedLeaf") else {
            fatalError("Could not find unsortedLeaf function in Octree.metal.")
        }
        guard let unsortedInternalFunction = library.makeFunction(name: "unsortedInternal") else {
            fatalError("Could not find unsortedInternal function in Octree.metal.")
        }
        guard let resetUIntBufferFunction = library.makeFunction(name: "resetUIntBuffer") else {
            fatalError("Could not find resetUIntBuffer function in Octree.metal.")
        }
        guard let clearBufferFunction = library.makeFunction(name: "clearBuffer") else {
            fatalError("Could not find clearBuffer function in Octree.metal.")
        }
        guard let clearBuffer64Function = library.makeFunction(name: "clearBuffer64") else {
            fatalError("Could not find clearBuffer64 function in Octree.metal.")
        }
        guard let clearOctreeNodeBufferFunction = library.makeFunction(name: "clearOctreeNodeBuffer") else {
            fatalError("Could not find clearOctreeNodeBuffer function in Octree.metal.")
        }
        
        let spherePipelineDescriptor = MTLRenderPipelineDescriptor()
        spherePipelineDescriptor.label = "Sphere Render Pipeline"
        spherePipelineDescriptor.vertexFunction = sphereVertexFunction
        spherePipelineDescriptor.fragmentFunction = fragmentFunction
        spherePipelineDescriptor.vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(starSphere.vertexDescriptor)
        spherePipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        spherePipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        let dustPipelineDescriptor = MTLRenderPipelineDescriptor()
        dustPipelineDescriptor.label = "Dust Render Pipeline"
        dustPipelineDescriptor.vertexFunction = dustVertexFunction
        dustPipelineDescriptor.fragmentFunction = fragmentFunction
        dustPipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        dustPipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        let colorAttachment = dustPipelineDescriptor.colorAttachments[0]!
        colorAttachment.isBlendingEnabled = true
        colorAttachment.rgbBlendOperation = .add
        colorAttachment.alphaBlendOperation = .add
        colorAttachment.sourceRGBBlendFactor = .sourceAlpha
        colorAttachment.sourceAlphaBlendFactor = .sourceAlpha
        colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        let postProcessPipelineDescriptor = MTLRenderPipelineDescriptor()
        postProcessPipelineDescriptor.label = "Post-Process Pipeline"
        postProcessPipelineDescriptor.vertexFunction = postProcessVertexFunction
        postProcessPipelineDescriptor.fragmentFunction = postProcessFragmentFunction
        postProcessPipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        
        do {
            spherePipelineState = try device.makeRenderPipelineState(descriptor: spherePipelineDescriptor)
            dustPipelineState = try device.makeRenderPipelineState(descriptor: dustPipelineDescriptor)
            postProcessPipelineState = try device.makeRenderPipelineState(descriptor: postProcessPipelineDescriptor)
            generateMortonCodesPipelineState = try device.makeComputePipelineState(function: generateMortonCodesFunction)
            aggregateLeafNodesPipelineState = try device.makeComputePipelineState(function: aggregateLeafNodesFunction)
            countUniqueMortonCodesPipelineState = try device.makeComputePipelineState(function: countUniqueMortonCodesFunction)
            aggregateNodesPipelineState = try device.makeComputePipelineState(function: aggregateNodesFunction)
            fillIndicesPipelineState = try device.makeComputePipelineState(function: fillIndicesFunction)
            unsortedLeafPipelineState = try device.makeComputePipelineState(function: unsortedLeafFunction)
            unsortedInternalPipelineState = try device.makeComputePipelineState(function: unsortedInternalFunction)
            resetUIntBufferPipelineState = try device.makeComputePipelineState(function: resetUIntBufferFunction)
            clearBufferPipelineState = try device.makeComputePipelineState(function: clearBufferFunction)
            clearBuffer64PipelineState = try device.makeComputePipelineState(function: clearBuffer64Function)
            clearOctreeNodeBufferPipelineState = try device.makeComputePipelineState(function: clearOctreeNodeBufferFunction)
        } catch {
            fatalError("Failed to create pipeline state: \(error)")
        }
    }
    
    func makeBuffers() {
        guard sphereCount > 0 else { return }
        self.globalUniformsBuffer = device.makeBuffer(length: MemoryLayout<GlobalUniforms>.stride, options: .storageModeShared)
        self.globalUniformsBuffer.label = "Global Uniforms Buffer"
        self.positionMassBuffer = device.makeBuffer(length: MemoryLayout<PositionMass>.stride * sphereCount, options: .storageModeShared)
        self.positionMassBuffer.label = "Position Mass Buffer"
        self.velocityRadiusBuffer = device.makeBuffer(length: MemoryLayout<VelocityRadius>.stride * sphereCount, options: .storageModeShared)
        self.velocityRadiusBuffer.label = "Velocity Radius Buffer"
        self.colorTypeBuffer = device.makeBuffer(length: MemoryLayout<ColorType>.stride * sphereCount, options: .storageModeShared)
        self.colorTypeBuffer.label = "Color Type Buffer"
        self.unsortedMortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * sphereCount, options: .storageModeShared)
        self.unsortedMortonCodesBuffer.label = "Unsorted Morton Codes Buffer"
        self.unsortedIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.unsortedIndicesBuffer.label = "Unsorted Indices Buffer"
        self.sortedMortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * sphereCount, options: .storageModeShared)
        self.sortedMortonCodesBuffer.label = "Sorted Morton Codes Buffer"
        self.sortedIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.sortedIndicesBuffer.label = "Sorted Indices Buffer"
        self.uniqueIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.uniqueIndicesBuffer.label = "Unique Indices Buffer"
        self.mortonCodeCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        self.mortonCodeCountBuffer.label = "Morton Code Count Buffer"
        self.layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        self.layerCountBuffer.label = "Layer Count Buffer"
        self.octreeNodesBuffer = device.makeBuffer(length: MemoryLayout<OctreeNode>.stride * sphereCount * 2, options: .storageModeShared)
        self.octreeNodesBuffer.label = "Octree Nodes Buffer"
        
        radixSorter.setMaxLength(UInt32(sphereCount))
    }
    
    // ... (keep your existing makeSpheres, fillIndices, etc.)
    func fillIndices(commandBuffer: MTLCommandBuffer, buffer: MTLBuffer, startIdx: Int, endIdx: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Fill Indices"
        computeEncoder.setComputePipelineState(fillIndicesPipelineState)
        computeEncoder.setBuffer(buffer, offset: 0, index: 0)
        var startVal = UInt32(startIdx)
        var endVal = UInt32(endIdx)
        computeEncoder.setBytes(&startVal, length: MemoryLayout<UInt32>.size, index: 1)
        computeEncoder.setBytes(&endVal, length: MemoryLayout<UInt32>.size, index: 2)
        let count = endIdx - startIdx
        if count <= 0 { return }
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((count + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }
    
    func clearBuffer<T>(commandBuffer: MTLCommandBuffer, buffer: MTLBuffer, count: Int, dataType: T.Type) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Clear Buffer"
        
        // Choose the appropriate pipeline state based on data type
        if T.self == UInt64.self {
            computeEncoder.setComputePipelineState(clearBuffer64PipelineState)
        } else {
            computeEncoder.setComputePipelineState(clearBufferPipelineState)
        }
        
        computeEncoder.setBuffer(buffer, offset: 0, index: 0)
        
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((count + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }
    
    func clearOctreeNodeBuffer(commandBuffer: MTLCommandBuffer, buffer: MTLBuffer, count: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Clear Octree Node Buffer"
        computeEncoder.setComputePipelineState(clearOctreeNodeBufferPipelineState)
        computeEncoder.setBuffer(buffer, offset: 0, index: 0)
        
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((count + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }
    
    func makeSpheres() {
        positions = []
        velocities = []
        colors = []
        doublePositions = []
        for i in 0..<sphereCount {
            let position = SIMD3<Double>(
                Double.random(in: -Double(spawnBounds)...Double(spawnBounds)),
                Double.random(in: -Double(spawnBounds)...Double(spawnBounds)),
                Double.random(in: -Double(spawnBounds)...Double(spawnBounds))
            )
            doublePositions.append(position)
            let velocity = simd_float3(
                Float.random(in: -initialVelocityMaximum...initialVelocityMaximum),
                Float.random(in: -initialVelocityMaximum...initialVelocityMaximum),
                Float.random(in: -initialVelocityMaximum...initialVelocityMaximum)
            )
            var type: UInt32 = 0
            var mass: Float = 0
            var radius: Float = 0
            var color: simd_float4 = simd_float4(0,0,0,1)
            if i < numStars {
                type = 0
                mass = starMass
                radius = starRadius
                let hue = Float(i) / Float(numStars)
                let oklabColor = simd_float3(0.8, 0.2 * cos(2 * .pi * hue), 0.2 * sin(2 * .pi * hue))
                let srgbColor = oklabToSrgb(oklab: oklabColor)
                color = simd_float4(srgbColor.x, srgbColor.y, srgbColor.z, 1.0)
            } else if i < numStars + numPlanets {
                type = 1
                mass = planetMass
                radius = planetRadius
                color = simd_float4(0, 0, 0, 1)
            } else {
                type = 2
                mass = dustMass
                radius = dustRadius
                color = simd_float4(0, 0, 0, 0)
            }
            positions.append(PositionMass(position: simd_float3(Float(position.x), Float(position.y), Float(position.z)), mass: mass))
            velocities.append(VelocityRadius(velocity: velocity, radius: radius))
            colors.append(ColorType(color: color, type: type))
        }
        positionMassBuffer.contents().copyMemory(from: positions, byteCount: MemoryLayout<PositionMass>.stride * positions.count)
        velocityRadiusBuffer.contents().copyMemory(from: velocities, byteCount: MemoryLayout<VelocityRadius>.stride * velocities.count)
        colorTypeBuffer.contents().copyMemory(from: colors, byteCount: MemoryLayout<ColorType>.stride * colors.count)
    }

    func draw(in view: MTKView) {
        guard pipelinesCreated,
              let drawable = view.currentDrawable,
              let screenRenderPassDescriptor = view.currentRenderPassDescriptor else { return }
        
        let systemTime = CFAbsoluteTimeGetCurrent()
        let deltaTime = min(Float(systemTime - lastFrameTime), 1.0 / 30.0)
        lastFrameTime = systemTime
        currentTime += deltaTime
        camera.update(deltaTime: deltaTime)
        updateGlobalUniforms()
        currentLayer = 0
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Main Command Buffer"
        
        if sphereCount > 0 {
            clearBuffer(commandBuffer: commandBuffer, buffer: sortedMortonCodesBuffer, count: sphereCount, dataType: UInt64.self)
            clearBuffer(commandBuffer: commandBuffer, buffer: sortedIndicesBuffer, count: sphereCount, dataType: UInt32.self)
            clearOctreeNodeBuffer(commandBuffer: commandBuffer, buffer: octreeNodesBuffer, count: sphereCount * 2)
            
            generateMortonCodes(commandBuffer: commandBuffer)
            
            // Always initialize the morton code count and layer count
            unsortedLeafMortonCodes(commandBuffer: commandBuffer)
            
            if currentLayer >= sortThreshold {
                fillIndices(commandBuffer: commandBuffer, buffer: unsortedIndicesBuffer, startIdx: 0, endIdx: sphereCount)
                
                // Use the stable radix sorter
                radixSorter.sort(commandBuffer: commandBuffer,
                                 input: unsortedMortonCodesBuffer,
                                 inputIndices: unsortedIndicesBuffer,
                                 output: sortedMortonCodesBuffer,
                                 outputIndices: sortedIndicesBuffer,
                                 estimatedLength: UInt32(sphereCount),
                                 actualCountBuffer: mortonCodeCountBuffer)
                
                 countUniqueMortonCodes(commandBuffer: commandBuffer, aggregate: false)
                 aggregateLeafNodes(commandBuffer: commandBuffer)
                 buildInternalNodes(commandBuffer: commandBuffer)
                
                // The final encoder is ended by the caller (buildInternalNodes)
            } else {
                aggregateLeafNodes(commandBuffer: commandBuffer)
                buildInternalNodes(commandBuffer: commandBuffer)
            }
        }
        
        if usePostProcessing {
            renderScene(commandBuffer: commandBuffer)
            postProcess(commandBuffer: commandBuffer, source: sceneTexture!, screenPassDescriptor: screenRenderPassDescriptor)
        } else {
            renderSceneDirect(commandBuffer: commandBuffer, screenPassDescriptor: screenRenderPassDescriptor)
        }
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    // ... (keep generateMortonCodes, countUnique, aggregateLeafNodes, etc.)
    func generateMortonCodes(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Generate Morton Codes"
        computeEncoder.setComputePipelineState(generateMortonCodesPipelineState)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(unsortedMortonCodesBuffer, offset: 0, index: 1)
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((sphereCount + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        currentLayer = 0
    }

    func countUniqueMortonCodes(commandBuffer: MTLCommandBuffer, aggregate: Bool) {
        clearBuffer(commandBuffer: commandBuffer, buffer: uniqueIndicesBuffer, count: sphereCount, dataType: UInt32.self)
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Count Unique Morton Codes"
        resetUIntBuffer(computeEncoder: computeEncoder, buffer: mortonCodeCountBuffer)
        computeEncoder.setComputePipelineState(countUniqueMortonCodesPipelineState)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(mortonCodeCountBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(uniqueIndicesBuffer, offset: 0, index: 2)
        var numSpheresVal = UInt32(numSpheres)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 3)
        var aggregateVal: UInt32 = aggregate ? 1 : 0
        computeEncoder.setBytes(&aggregateVal, length: MemoryLayout<UInt32>.size, index: 4)
        computeEncoder.setBuffer(layerCountBuffer, offset: 0, index: 5)
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((sphereCount + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        currentLayer += 1
    }

    func unsortedLeafMortonCodes(commandBuffer: MTLCommandBuffer) {
        fillIndices(commandBuffer: commandBuffer, buffer: uniqueIndicesBuffer, startIdx: 0, endIdx: sphereCount)
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Unsorted Leaf Morton Codes"
        computeEncoder.setComputePipelineState(unsortedLeafPipelineState)
        var numSpheresVal = UInt32(numSpheres)
        computeEncoder.setBuffer(mortonCodeCountBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(layerCountBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 2)
        computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        computeEncoder.endEncoding()
        currentLayer += 1
    }

    func unsortedInternalMortonCodes(commandBuffer: MTLCommandBuffer, layer: Int) {
        let count = max(1, Int(ceil(Double(sphereCount) / pow(8.0, Double(layer)))))
        fillIndices(commandBuffer: commandBuffer, buffer: uniqueIndicesBuffer, startIdx: 0, endIdx: count)
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Unsorted Internal Morton Codes Layer \(layer)"
        computeEncoder.setComputePipelineState(unsortedInternalPipelineState)
        computeEncoder.setBuffer(mortonCodeCountBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(layerCountBuffer, offset: 0, index: 1)
        computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        computeEncoder.endEncoding()
        currentLayer += 1
    }
    
    func aggregateLeafNodes(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Aggregate Leaf Nodes"
        computeEncoder.setComputePipelineState(aggregateLeafNodesPipelineState)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(uniqueIndicesBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(colorTypeBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(mortonCodeCountBuffer, offset: 0, index: 5)
        computeEncoder.setBuffer(octreeNodesBuffer, offset: 0, index: 6)
        computeEncoder.setBuffer(unsortedMortonCodesBuffer, offset: 0, index: 7)
        computeEncoder.setBuffer(unsortedIndicesBuffer, offset: 0, index: 8)
        let numNodes = mortonCodeCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
        if numNodes == 0 { 
            computeEncoder.endEncoding()
            return 
        }
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((Int(numNodes) + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }

    func octreeLayerOffset(layer: Int, leafCount: Int) -> Int {
        if layer == 0 {
            return 0
        }
        
        // For a full octree, each layer has exactly leafCount / 8^layer nodes
        // We need to sum up all previous layers
        var offset = leafCount // Layer 0 starts at 0, Layer 1 starts at leafCount
        
        for l in 1..<layer {
            // Each subsequent layer has leafCount / 8^l nodes
            let layerSize = max(1, leafCount / Int(pow(8.0, Double(l))))
            offset += layerSize
        }
        
        return offset
    }

    func aggregateInternalNodes(commandBuffer: MTLCommandBuffer, inputOffset: Int, outputOffset: Int, nodeCount: Int, layer: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Aggregate Internal Nodes Layer \(layer)"
        if nodeCount <= 0 { 
            computeEncoder.endEncoding()
            return 
        }

        computeEncoder.setComputePipelineState(aggregateNodesPipelineState)
        computeEncoder.setBuffer(octreeNodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(uniqueIndicesBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(mortonCodeCountBuffer, offset: 0, index: 4)
        var inputOffsetVal = UInt32(inputOffset)
        var outputOffsetVal = UInt32(outputOffset)
        computeEncoder.setBytes(&inputOffsetVal, length: MemoryLayout<UInt32>.size, index: 5)
        computeEncoder.setBytes(&outputOffsetVal, length: MemoryLayout<UInt32>.size, index: 6)
        var layerVal = UInt32(layer)
        computeEncoder.setBytes(&layerVal, length: MemoryLayout<UInt32>.size, index: 7)
        computeEncoder.setBuffer(unsortedMortonCodesBuffer, offset: 0, index: 8)
        computeEncoder.setBuffer(unsortedIndicesBuffer, offset: 0, index: 9)
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((nodeCount + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }

    func buildInternalNodes(commandBuffer: MTLCommandBuffer) {
        let leafCount = sphereCount
        let maxLayers = 8 // Fixed number of layers to avoid infinite loops
        
        for layer in 0..<maxLayers {
            let estimatedNodeCount = max(1, leafCount / Int(pow(8.0, Double(layer))))
            
            let inputOffset = octreeLayerOffset(layer: layer, leafCount: leafCount)
            let outputOffset = octreeLayerOffset(layer: layer + 1, leafCount: leafCount)

            // print("Layer \(layer), inputOffset: \(inputOffset), outputOffset: \(outputOffset), estimatedNodeCount: \(estimatedNodeCount), layerSize: \(outputOffset - inputOffset)")
            
            clearBuffer(commandBuffer: commandBuffer, buffer: sortedMortonCodesBuffer, count: sphereCount, dataType: UInt64.self)
            clearBuffer(commandBuffer: commandBuffer, buffer: sortedIndicesBuffer, count: sphereCount, dataType: UInt32.self)
            
            if layer >= sortThreshold {
                // Use the stable radix sorter
                radixSorter.sort(commandBuffer: commandBuffer,
                                 input: unsortedMortonCodesBuffer,
                                 inputIndices: unsortedIndicesBuffer,
                                 output: sortedMortonCodesBuffer,
                                 outputIndices: sortedIndicesBuffer,
                                 estimatedLength: UInt32(estimatedNodeCount),
                                 actualCountBuffer: mortonCodeCountBuffer)
                
                countUniqueMortonCodes(commandBuffer: commandBuffer, aggregate: true)
            } else {
                unsortedInternalMortonCodes(commandBuffer: commandBuffer, layer: layer)
            }

            clearBuffer(commandBuffer: commandBuffer, buffer: unsortedMortonCodesBuffer, count: sphereCount, dataType: UInt64.self)
            clearBuffer(commandBuffer: commandBuffer, buffer: unsortedIndicesBuffer, count: sphereCount, dataType: UInt32.self)
            
            aggregateInternalNodes(commandBuffer: commandBuffer, inputOffset: inputOffset, outputOffset: outputOffset, nodeCount: estimatedNodeCount, layer: layer + 1)
        }
    }
    
    // ... (keep your renderScene, renderSceneDirect, postProcess, and resetUIntBuffer methods)
    func renderScene(commandBuffer: MTLCommandBuffer) {
        let sceneRenderPassDescriptor = MTLRenderPassDescriptor()
        sceneRenderPassDescriptor.colorAttachments[0].texture = sceneTexture
        sceneRenderPassDescriptor.colorAttachments[0].loadAction = .clear
        sceneRenderPassDescriptor.colorAttachments[0].storeAction = .store
        sceneRenderPassDescriptor.colorAttachments[0].clearColor = backgroundColor
        
        sceneRenderPassDescriptor.depthAttachment.texture = depthTexture
        sceneRenderPassDescriptor.depthAttachment.loadAction = .clear
        sceneRenderPassDescriptor.depthAttachment.storeAction = .dontCare
        sceneRenderPassDescriptor.depthAttachment.clearDepth = 1.0
        
        renderSceneToPass(commandBuffer: commandBuffer, renderPassDescriptor: sceneRenderPassDescriptor, label: "Scene Render Pass")
    }

    func renderSceneDirect(commandBuffer: MTLCommandBuffer, screenPassDescriptor: MTLRenderPassDescriptor) {
        screenPassDescriptor.colorAttachments[0].clearColor = backgroundColor
        screenPassDescriptor.depthAttachment.texture = depthTexture // Use the memoryless depth texture
        screenPassDescriptor.depthAttachment.loadAction = .clear
        screenPassDescriptor.depthAttachment.storeAction = .dontCare
        screenPassDescriptor.depthAttachment.clearDepth = 1.0
        renderSceneToPass(commandBuffer: commandBuffer, renderPassDescriptor: screenPassDescriptor, label: "Direct Scene Render Pass")
    }
    
    private func renderSceneToPass(commandBuffer: MTLCommandBuffer, renderPassDescriptor: MTLRenderPassDescriptor, label: String) {
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
        renderEncoder.label = label
        if renderPassDescriptor.depthAttachment.texture != nil {
            renderEncoder.setDepthStencilState(depthStencilState)
        }
        renderEncoder.setCullMode(.back)
        
        renderEncoder.setVertexBuffer(globalUniformsBuffer, offset: 0, index: 1)
        renderEncoder.setVertexBuffer(positionMassBuffer, offset: 0, index: 2)
        renderEncoder.setVertexBuffer(velocityRadiusBuffer, offset: 0, index: 3)
        renderEncoder.setVertexBuffer(colorTypeBuffer, offset: 0, index: 4)
        
        var lastPipeline: MTLRenderPipelineState? = nil
        if numStars > 0 {
            if lastPipeline !== spherePipelineState {
                renderEncoder.setRenderPipelineState(spherePipelineState)
                lastPipeline = spherePipelineState
            }
            for (index, vertexBuffer) in starSphere.vertexBuffers.enumerated() {
                renderEncoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: index)
            }
            let submesh = starSphere.submeshes[0]
            renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType, indexCount: submesh.indexCount, indexType: submesh.indexType, indexBuffer: submesh.indexBuffer.buffer, indexBufferOffset: submesh.indexBuffer.offset, instanceCount: Int(numStars))
        }
        
        if numPlanets > 0 {
            if lastPipeline !== spherePipelineState {
                renderEncoder.setRenderPipelineState(spherePipelineState)
                lastPipeline = spherePipelineState
            }
            for (index, vertexBuffer) in planetSphere.vertexBuffers.enumerated() {
                renderEncoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: index)
            }
            let submesh = planetSphere.submeshes[0]
            renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType, indexCount: submesh.indexCount, indexType: submesh.indexType, indexBuffer: submesh.indexBuffer.buffer, indexBufferOffset: submesh.indexBuffer.offset, instanceCount: Int(numPlanets), baseVertex: 0, baseInstance: Int(numStars))
        }
        
        if numDust > 0 {
            if lastPipeline !== dustPipelineState {
                renderEncoder.setRenderPipelineState(dustPipelineState)
                lastPipeline = dustPipelineState
            }
            let dustInstanceStart = Int(numStars + numPlanets)
            renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: 1, instanceCount: Int(numDust), baseInstance: dustInstanceStart)
        }
        
        renderEncoder.endEncoding()
    }

    private func postProcess(commandBuffer: MTLCommandBuffer, source: MTLTexture, screenPassDescriptor: MTLRenderPassDescriptor) {
        screenPassDescriptor.depthAttachment.texture = nil
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: screenPassDescriptor) else { return }
        renderEncoder.label = "PostProcess Pass"
        renderEncoder.setRenderPipelineState(postProcessPipelineState)
        renderEncoder.setFragmentTexture(source, index: 0)
        renderEncoder.setFragmentSamplerState(textureSampler, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.endEncoding()
    }
    
    func resetUIntBuffer(computeEncoder: MTLComputeCommandEncoder, buffer: MTLBuffer) {
        // Note: This function doesn't create its own encoder, it uses the one passed in
        // The label should be set by the calling function
        computeEncoder.setComputePipelineState(resetUIntBufferPipelineState)
        computeEncoder.setBuffer(buffer, offset: 0, index: 0)
        computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }
}
