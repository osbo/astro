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
    var numDust: Int32 = 1000000
    
    var numSpheres: Int32 {
        return numStars + numPlanets + numDust
    }
    
    var starRadius: Float = 50000.0 // 500 * (1_048_575 / 10_000)
    var planetRadius: Float = 10000.0 // 100 * (1_048_575 / 10_000)
    var dustRadius: Float = 1
    var starMass: Float = 1000000 // 1_000_000 * (1_048_575 / 10_000)
    var planetMass: Float = 10000 // 10_000 * (1_048_575 / 10_000)
    var dustMass: Float = 100 // 100 * (1_048_575 / 10_000)
    var initialVelocityMaximum: Float = 5000 // 50 * (1_048_575 / 10_000)
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
    var mortonCodesBuffer: MTLBuffer!
    var indicesBuffer: MTLBuffer!
    var sortedMortonCodesBuffer: MTLBuffer!
    var sortedIndicesBuffer: MTLBuffer!
    var octreeNodesBuffer: MTLBuffer!
    var uniqueMortonCodeCountBuffer: MTLBuffer!
    var uniqueMortonCodeStartIndicesBuffer: MTLBuffer!
    // Preallocated buffers for internal node building
    var indexBufferA: MTLBuffer!
    var indexBufferB: MTLBuffer!
    var mortonBufferA: MTLBuffer!
    var mortonBufferB: MTLBuffer!
    // Persistent 0..n-1 index buffer
    var sphereIndices: MTLBuffer!

    var pipelinesCreated = false
    var generateMortonCodesPipelineState: MTLComputePipelineState!
    var aggregateLeafNodesPipelineState: MTLComputePipelineState!
    var countUniqueMortonCodesPipelineState: MTLComputePipelineState!
    var aggregateNodesPipelineState: MTLComputePipelineState!
    var fillIndicesPipelineState: MTLComputePipelineState!
    
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
        // Defer makeBuffers until after pipelines are created
        createTextureSampler(device: device)
        radixSorter = MetalKernelsRadixSort(device: device)
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
        guard currentDrawableSize.width > 0 && currentDrawableSize.height > 0 else { return }
        let pointer = globalUniformsBuffer.contents().bindMemory(to: GlobalUniforms.self, capacity: 1)
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
        // Only create sceneTexture if post-processing is enabled
        if usePostProcessing {
            let fullResDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: Int(size.width), height: Int(size.height), mipmapped: false)
            fullResDesc.storageMode = .private
            fullResDesc.usage = [.renderTarget, .shaderRead]
            sceneTexture = device.makeTexture(descriptor: fullResDesc)
            sceneTexture.label = "Scene Texture"
        } else {
            sceneTexture = nil
        }
        // Always create a custom .memoryless depth texture
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
            fatalError("Could not load default Metal library. Make sure .metal files are added to the target.")
        }

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
        

        let spherePipelineDescriptor = MTLRenderPipelineDescriptor()
        spherePipelineDescriptor.label = "Sphere Render Pipeline"
        spherePipelineDescriptor.vertexFunction = sphereVertexFunction
        spherePipelineDescriptor.fragmentFunction = fragmentFunction
        spherePipelineDescriptor.vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(starSphere.vertexDescriptor)
        spherePipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        spherePipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        
        let dustPipelineDescriptor = MTLRenderPipelineDescriptor()
        dustPipelineDescriptor.label = "Dust Render Pipeline"
        dustPipelineDescriptor.vertexFunction = dustVertexFunction
        dustPipelineDescriptor.fragmentFunction = fragmentFunction
        dustPipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        dustPipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        
        // Enable alpha blending for dust
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
        } catch {
            fatalError("Failed to create pipeline state: \(error)")
        }
    }
    
    func makeBuffers() {
        let sphereCount = Int(numSpheres)
        guard sphereCount > 0 else { return }
        self.globalUniformsBuffer = device.makeBuffer(length: MemoryLayout<GlobalUniforms>.stride, options: .storageModeShared)
        self.globalUniformsBuffer.label = "Global Uniforms Buffer"
        
        self.positionMassBuffer = device.makeBuffer(length: MemoryLayout<PositionMass>.stride * sphereCount, options: .storageModeShared)
        self.positionMassBuffer.label = "Position Mass Buffer"
        
        self.velocityRadiusBuffer = device.makeBuffer(length: MemoryLayout<VelocityRadius>.stride * sphereCount, options: .storageModeShared)
        self.velocityRadiusBuffer.label = "Velocity Radius Buffer"
        
        self.colorTypeBuffer = device.makeBuffer(length: MemoryLayout<ColorType>.stride * sphereCount, options: .storageModeShared)
        self.colorTypeBuffer.label = "Color Type Buffer"
        
        self.mortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * sphereCount, options: .storageModeShared)
        self.mortonCodesBuffer.label = "Morton Codes Buffer"
        
        self.indicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.indicesBuffer.label = "Indices Buffer"
        
        self.sortedMortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * sphereCount, options: .storageModeShared)
        self.sortedMortonCodesBuffer.label = "Sorted Morton Codes Buffer"
        
        self.sortedIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.sortedIndicesBuffer.label = "Sorted Indices Buffer"
        
        // Octree nodes buffer - worst case: full binary tree with numSpheres leaves
        let octreeNodeCount = 2 * sphereCount - 1
        self.octreeNodesBuffer = device.makeBuffer(length: MemoryLayout<OctreeLeafNode>.stride * octreeNodeCount, options: .storageModeShared)
        self.octreeNodesBuffer.label = "Octree Nodes Buffer"
        
        // Buffer to store the count of unique morton codes
        self.uniqueMortonCodeCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        self.uniqueMortonCodeCountBuffer.label = "Unique Morton Code Count Buffer"
        
        // Buffer to store the starting indices of unique morton codes
        self.uniqueMortonCodeStartIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.uniqueMortonCodeStartIndicesBuffer.label = "Unique Morton Code Start Indices Buffer"
        // Preallocate buffers for internal node building (max size needed)
        let maxOctreeNodeCount = 2 * sphereCount - 1
        self.indexBufferA = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxOctreeNodeCount, options: .storageModeShared)
        self.indexBufferA.label = "Index Buffer A"
        self.indexBufferB = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxOctreeNodeCount, options: .storageModeShared)
        self.indexBufferB.label = "Index Buffer B"
        self.mortonBufferA = device.makeBuffer(length: MemoryLayout<UInt64>.stride * maxOctreeNodeCount, options: .storageModeShared)
        self.mortonBufferA.label = "Morton Buffer A"
        self.mortonBufferB = device.makeBuffer(length: MemoryLayout<UInt64>.stride * maxOctreeNodeCount, options: .storageModeShared)
        self.mortonBufferB.label = "Morton Buffer B"
        // Persistent 0..n-1 index buffer
        self.sphereIndices = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.sphereIndices.label = "Sphere Indices Buffer"
        // Fill the sphereIndices buffer using the fillIndices kernel
        fillIndices(buffer: sphereIndices, startIdx: 0, endIdx: sphereCount)
    }
    
    // Generic function to fill a buffer with startIdx..(endIdx-1) using the fillIndices kernel
    func fillIndices(buffer: MTLBuffer, startIdx: Int, endIdx: Int) {
        guard let commandQueue = self.commandQueue else { return }
        let commandBuffer = commandQueue.makeCommandBuffer()!
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Fill Indices Buffer"
        computeEncoder.setComputePipelineState(fillIndicesPipelineState)
        computeEncoder.setBuffer(buffer, offset: 0, index: 0)
        var startVal = UInt32(startIdx)
        var endVal = UInt32(endIdx)
        computeEncoder.setBytes(&startVal, length: MemoryLayout<UInt32>.size, index: 1)
        computeEncoder.setBytes(&endVal, length: MemoryLayout<UInt32>.size, index: 2)
        let count = endIdx - startIdx
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((count + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func makeSpheres() {
        positions = []
        velocities = []
        colors = []
        doublePositions = []
        for i in 0..<numSpheres {
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
            // Store world position directly
            positions.append(PositionMass(position: simd_float3(Float(position.x), Float(position.y), Float(position.z)), mass: mass))
            velocities.append(VelocityRadius(velocity: velocity, radius: radius))
            colors.append(ColorType(color: color, type: type))
        }
        // Initial upload
        positionMassBuffer.contents().copyMemory(from: positions, byteCount: MemoryLayout<PositionMass>.stride * positions.count)
        velocityRadiusBuffer.contents().copyMemory(from: velocities, byteCount: MemoryLayout<VelocityRadius>.stride * velocities.count)
        colorTypeBuffer.contents().copyMemory(from: colors, byteCount: MemoryLayout<ColorType>.stride * colors.count)
        // Use persistent 0..n-1 buffer for indices
        indicesBuffer.contents().copyMemory(from: sphereIndices.contents(), byteCount: MemoryLayout<UInt32>.stride * Int(numSpheres))
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
        // No per-frame CPU offset for floating origin
        updateGlobalUniforms()
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Main Command Buffer"
        
        if numSpheres > 0 {
            // Generate Morton codes for all particles at the beginning of every frame
            generateMortonCodes(commandBuffer: commandBuffer)
            
            if currentLayer >= sortThreshold {
                // Sort the Morton codes
                radixSorter.sort(commandBuffer: commandBuffer, input: mortonCodesBuffer, inputIndices: indicesBuffer, output: sortedMortonCodesBuffer, outputIndices: sortedIndicesBuffer, length: UInt32(numSpheres))

                // Count unique morton codes
                countUniqueMortonCodes(commandBuffer: commandBuffer, aggregate: false)
            } else {
                unsortedMortonCodes()
            }
            
            // Aggregate leaf nodes
            aggregateLeafNodes(commandBuffer: commandBuffer)
            
            // Build internal nodes
            buildInternalNodes(commandBuffer: commandBuffer, uniqueLeafNodeCount: UInt32(numSpheres))
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
    
    func generateMortonCodes(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Morton Code Generation"
        
        computeEncoder.setComputePipelineState(generateMortonCodesPipelineState)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(mortonCodesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(indicesBuffer, offset: 0, index: 2)
        
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((Int(numSpheres) + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

        currentLayer = 0
    }
    
    func countUniqueMortonCodes(commandBuffer: MTLCommandBuffer, aggregate: Bool) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Count Unique Morton Codes"
        
        computeEncoder.setComputePipelineState(countUniqueMortonCodesPipelineState)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(uniqueMortonCodeCountBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(uniqueMortonCodeStartIndicesBuffer, offset: 0, index: 2)
        
        var numSpheresVal = UInt32(numSpheres)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 3)
        var aggregateVal: UInt32 = aggregate ? 1 : 0
        computeEncoder.setBytes(&aggregateVal, length: MemoryLayout<UInt32>.size, index: 4)
        
        // Reset the count to 0 before dispatching
        var initialCount: UInt32 = 0
        uniqueMortonCodeCountBuffer.contents().copyMemory(from: &initialCount, byteCount: MemoryLayout<UInt32>.size)
        
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((Int(numSpheres) + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }
    
    // Alternate: Use unsortedMortonCodes kernel to fill uniqueMortonCodeStartIndicesBuffer with 0..n-1 and set count to numSpheres
    func unsortedMortonCodes() {
        // Instead of running the kernel, just copy the pre-filled buffer
        let count = Int(numSpheres)
        // Set the uniqueMortonCodeCount to numSpheres, mimicking the kernel's atomic store
        uniqueMortonCodeCountBuffer.contents().storeBytes(of: UInt32(numSpheres), as: UInt32.self)
        memcpy(uniqueMortonCodeStartIndicesBuffer.contents(), sphereIndices.contents(), MemoryLayout<UInt32>.stride * count)
    }
    
    func aggregateLeafNodes(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Aggregate Leaf Nodes"
        
        computeEncoder.setComputePipelineState(aggregateLeafNodesPipelineState)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(colorTypeBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(octreeNodesBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(mortonCodesBuffer, offset: 0, index: 5)
        computeEncoder.setBuffer(uniqueMortonCodeCountBuffer, offset: 0, index: 6)
        computeEncoder.setBuffer(uniqueMortonCodeStartIndicesBuffer, offset: 0, index: 7)
        
        var numSpheresVal = UInt32(numSpheres)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 8)
        
        // Dispatch enough threads for the worst-case scenario (one leaf node per sphere).
        // The kernel will check the actual unique count and exit early if needed.
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((Int(numSpheres) + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

        currentLayer += 1
    }
    
    func buildInternalNodes(commandBuffer: MTLCommandBuffer, uniqueLeafNodeCount: UInt32) {
        let maxPasses = Int(ceil(log2(Double(numSpheres))))
        let octreeNodeCount = 2 * Int(numSpheres) - 1
        let nodeBuffer = octreeNodesBuffer!
        let maxNodes = octreeNodeCount

        // Initialize indexBufferA with [0, 1, ..., numLeaves-1]
        let numLeaves = Int(numSpheres)
        // Use persistent 0..n-1 buffer for initial indices
        indexBufferA.contents().copyMemory(from: sphereIndices.contents(), byteCount: MemoryLayout<UInt32>.stride * numLeaves)

        var currentStart = 0
        var currentCount = numLeaves
        var nextStart = numLeaves
        var pass = 0
        var layerShift: UInt32 = 3
        var idxIn = indexBufferA!
        var idxOut = indexBufferB!
        var mortonIn = mortonBufferA!
        var mortonOut = mortonBufferB!
        while currentCount > 1 && pass < maxPasses {
            if currentLayer >= sortThreshold {
                // 1. Sort morton codes and indices for current level
                radixSorter.sort(commandBuffer: commandBuffer, input: mortonIn, inputIndices: idxIn, output: mortonOut, outputIndices: idxOut, length: UInt32(currentCount))
                // 2. Count unique parent codes (aggregate: true)
                countUniqueMortonCodes(commandBuffer: commandBuffer, aggregate: true)
            } else {
                // 1. Unsorted morton codes
                unsortedMortonCodes()
                // 2. No need to count unique morton codes, as all are unique and indices are 0..n-1
            }
            // 3. Prepare output index buffer for next level
            let nextCount = max(1, (currentCount + 7) / 8)
            // Dynamically select the scratch buffer for nextIndices so it is never the same as idxOut
            let nextIndicesBuffer: MTLBuffer = (idxOut === indexBufferA) ? indexBufferB! : indexBufferA!
            fillIndices(buffer: nextIndicesBuffer, startIdx: nextStart, endIdx: nextStart + nextCount)
            idxOut.contents().copyMemory(from: nextIndicesBuffer.contents(), byteCount: MemoryLayout<UInt32>.stride * nextCount)
            // 4. Aggregate nodes: kernel writes to nodeBuffer[nextStart..nextStart+nextCount]
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.label = "Aggregate Internal Nodes"
                computeEncoder.setComputePipelineState(aggregateNodesPipelineState)
                computeEncoder.setBuffer(mortonOut, offset: 0, index: 0)
                computeEncoder.setBuffer(idxOut, offset: 0, index: 1)
                computeEncoder.setBuffer(nodeBuffer, offset: 0, index: 2) // all nodes in one buffer
                computeEncoder.setBuffer(mortonIn, offset: 0, index: 3)
                computeEncoder.setBuffer(uniqueMortonCodeCountBuffer, offset: 0, index: 4)
                computeEncoder.setBuffer(uniqueMortonCodeStartIndicesBuffer, offset: 0, index: 5)
                var numChildren = UInt32(currentCount)
                var inputNodeStart = UInt32(currentStart)
                var outputNodeStart = UInt32(nextStart)
                computeEncoder.setBytes(&numChildren, length: MemoryLayout<UInt32>.size, index: 6)
                computeEncoder.setBytes(&inputNodeStart, length: MemoryLayout<UInt32>.size, index: 7)
                computeEncoder.setBytes(&outputNodeStart, length: MemoryLayout<UInt32>.size, index: 8)
                computeEncoder.setBytes(&layerShift, length: MemoryLayout<UInt32>.size, index: 9)
                let threadGroupSize = MTLSizeMake(64, 1, 1)
                let threadGroups = MTLSizeMake((currentCount + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
                computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
                computeEncoder.endEncoding()
            }
            // Swap index/morton buffers for next pass
            swap(&idxIn, &idxOut)
            swap(&mortonIn, &mortonOut)
            currentStart = nextStart
            currentCount = nextCount
            nextStart += nextCount
            pass += 1
            currentLayer += 1
            layerShift += 3
        }
    }
    
    func renderScene(commandBuffer: MTLCommandBuffer) {
        let sceneRenderPassDescriptor = MTLRenderPassDescriptor()
        sceneRenderPassDescriptor.colorAttachments[0].texture = sceneTexture
        sceneRenderPassDescriptor.colorAttachments[0].loadAction = .clear
        sceneRenderPassDescriptor.colorAttachments[0].storeAction = .store
        sceneRenderPassDescriptor.colorAttachments[0].clearColor = backgroundColor

        // Use custom depth texture
        sceneRenderPassDescriptor.depthAttachment.texture = depthTexture
        sceneRenderPassDescriptor.depthAttachment.loadAction = .clear
        sceneRenderPassDescriptor.depthAttachment.storeAction = .dontCare
        sceneRenderPassDescriptor.depthAttachment.clearDepth = 1.0

        renderSceneToPass(commandBuffer: commandBuffer, renderPassDescriptor: sceneRenderPassDescriptor, label: "Scene Render Pass")
    }
    
    func renderSceneDirect(commandBuffer: MTLCommandBuffer, screenPassDescriptor: MTLRenderPassDescriptor) {
        // Set the clear color for direct rendering
        screenPassDescriptor.colorAttachments[0].clearColor = backgroundColor
        // Do not attach a depth texture to the screen pass
        screenPassDescriptor.depthAttachment.texture = nil
        renderSceneToPass(commandBuffer: commandBuffer, renderPassDescriptor: screenPassDescriptor, label: "Direct Scene Render Pass")
    }
    
    private func renderSceneToPass(commandBuffer: MTLCommandBuffer, renderPassDescriptor: MTLRenderPassDescriptor, label: String) {
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
        renderEncoder.label = label
        // Only set depthStencilState if a depth texture is attached
        if let depthAttachment = renderPassDescriptor.depthAttachment, depthAttachment.texture != nil {
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
        // No depth for post-process pass
        screenPassDescriptor.depthAttachment.texture = nil

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: screenPassDescriptor) else { return }
        renderEncoder.label = "PostProcess Pass"
        renderEncoder.setRenderPipelineState(postProcessPipelineState)
        renderEncoder.setFragmentTexture(source, index: 0)
        renderEncoder.setFragmentSamplerState(textureSampler, index: 0)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.endEncoding()
    }
}
