import MetalKit
import MetalPerformanceShaders
import simd

class Renderer: NSObject, MTKViewDelegate {
    var parent: ContentView
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var depthStencilState: MTLDepthStencilState!
    var sceneTexture: MTLTexture!
    var currentDrawableSize: CGSize = .zero
    var textureSampler: MTLSamplerState!
    
    var camera = Camera()
    var lastFrameTime = CFAbsoluteTimeGetCurrent()
    var currentTime: Float = 0.0
    
    // --- Simulation Parameters ---
    var usePostProcessing: Bool = false
    var numStars: Int32 = 3
    var numPlanets: Int32 = 10
    var numDust: Int32 = 0
    
    var numSpheres: Int32 {
        return numStars + numPlanets + numDust
    }
    
    var starRadius: Int32 = 500
    var planetRadius: Int32 = 100
    var dustRadius: Int32 = 1
    var starMass: Int32 = 100000
    var planetMass: Int32 = 1000
    var dustMass: Int32 = 10
    var initialVelocityMaximum: Int32 = 50
    var spawnBounds: Int32 = 10000
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
    var octreeLeafNodesBuffer: MTLBuffer!
    var uniqueMortonCodeCountBuffer: MTLBuffer!
    var uniqueMortonCodeStartIndicesBuffer: MTLBuffer!

    var pipelinesCreated = false
    var generateMortonCodesPipelineState: MTLComputePipelineState!
    var aggregateLeafNodesPipelineState: MTLComputePipelineState!
    var countUniqueMortonCodesPipelineState: MTLComputePipelineState!
    var radixSorter: MetalKernelsRadixSort!

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
        makeBuffers()
        makeSpheres()
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
        pointer[0].viewMatrix = camera.viewMatrix()
        pointer[0].cameraPosition = camera.position
        let aspect = Float(currentDrawableSize.width) / Float(currentDrawableSize.height)
        pointer[0].projectionMatrix = createProjectionMatrix(fov: .pi/2.5, aspect: aspect, near: 10.0, far: 500000.0)
    }
    
    private func createTextureSampler(device: MTLDevice) {
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        textureSampler = device.makeSamplerState(descriptor: samplerDescriptor)
    }
    
    private func setupOffscreenTextures(size: CGSize) {
        let fullResDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: Int(size.width), height: Int(size.height), mipmapped: false)
        fullResDesc.storageMode = .private
        fullResDesc.usage = [.renderTarget, .shaderRead]
        sceneTexture = device.makeTexture(descriptor: fullResDesc)
        sceneTexture.label = "Scene Texture"
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        currentDrawableSize = size
        setupOffscreenTextures(size: size)
        
        if !pipelinesCreated {
            makePipelines(view: view)
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
        
        // Octree leaf nodes buffer - worst case: one leaf node per sphere
        self.octreeLeafNodesBuffer = device.makeBuffer(length: MemoryLayout<OctreeLeafNode>.stride * sphereCount, options: .storageModeShared)
        self.octreeLeafNodesBuffer.label = "Octree Leaf Nodes Buffer"
        
        // Buffer to store the count of unique morton codes
        self.uniqueMortonCodeCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        self.uniqueMortonCodeCountBuffer.label = "Unique Morton Code Count Buffer"
        
        // Buffer to store the starting indices of unique morton codes
        self.uniqueMortonCodeStartIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.uniqueMortonCodeStartIndicesBuffer.label = "Unique Morton Code Start Indices Buffer"
    }
    
    func makeSpheres() {
        var positions: [PositionMass] = []
        var velocities: [VelocityRadius] = []
        var colors: [ColorType] = []
        
        for i in 0..<numSpheres {
            let position = simd_float3(
                Float(Int32.random(in: -spawnBounds...spawnBounds)),
                Float(Int32.random(in: -spawnBounds...spawnBounds)),
                Float(Int32.random(in: -spawnBounds...spawnBounds))
            )
            let velocity = simd_float3(
                Float(Int32.random(in: -initialVelocityMaximum...initialVelocityMaximum)),
                Float(Int32.random(in: -initialVelocityMaximum...initialVelocityMaximum)),
                Float(Int32.random(in: -initialVelocityMaximum...initialVelocityMaximum))
            )
            
            var type: UInt32 = 0
            var mass: Float = 0
            var radius: Float = 0
            var color: simd_float4 = simd_float4(0,0,0,1)

            if i < numStars {
                type = 0
                mass = Float(starMass)
                radius = Float(starRadius)
                
                // Distribute star colors across the OKLab spectrum
                let hue = Float(i) / Float(numStars)
                let oklabColor = simd_float3(0.8, 0.2 * cos(2 * .pi * hue), 0.2 * sin(2 * .pi * hue))
                let srgbColor = oklabToSrgb(oklab: oklabColor)
                color = simd_float4(srgbColor.x, srgbColor.y, srgbColor.z, 1.0)

            } else if i < numStars + numPlanets {
                type = 1
                mass = Float(planetMass)
                radius = Float(planetRadius)
                color = simd_float4(0, 0, 0, 1) // Black for planets
            } else {
                type = 2
                mass = Float(dustMass)
                radius = Float(dustRadius)
                color = simd_float4(0, 0, 0, 0) // Transparent for dust
            }
            
            positions.append(PositionMass(position: position, mass: mass))
            velocities.append(VelocityRadius(velocity: velocity, radius: radius))
            colors.append(ColorType(color: color, type: type))
        }
        
        positionMassBuffer.contents().copyMemory(from: positions, byteCount: MemoryLayout<PositionMass>.stride * positions.count)
        velocityRadiusBuffer.contents().copyMemory(from: velocities, byteCount: MemoryLayout<VelocityRadius>.stride * velocities.count)
        colorTypeBuffer.contents().copyMemory(from: colors, byteCount: MemoryLayout<ColorType>.stride * colors.count)
        
        // Initialize indices buffer with sequential indices
        let indices = Array(0..<numSpheres).map { UInt32($0) }
        indicesBuffer.contents().copyMemory(from: indices, byteCount: MemoryLayout<UInt32>.stride * indices.count)
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
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Main Command Buffer"
        
        if numSpheres > 0 {
            // Generate Morton codes for all particles at the beginning of every frame
            generateMortonCodes(commandBuffer: commandBuffer)
            
            // Sort the Morton codes
            radixSorter.sort(commandBuffer: commandBuffer, input: mortonCodesBuffer, inputIndices: indicesBuffer, output: sortedMortonCodesBuffer, outputIndices: sortedIndicesBuffer, length: UInt32(numSpheres))
            
            // Count unique morton codes
            countUniqueMortonCodes(commandBuffer: commandBuffer)
            
            // Aggregate leaf nodes
            aggregateLeafNodes(commandBuffer: commandBuffer)
        }
        
        if usePostProcessing {
            renderScene(commandBuffer: commandBuffer, view: view)
            postProcess(commandBuffer: commandBuffer, source: sceneTexture, screenPassDescriptor: screenRenderPassDescriptor)
        } else {
            renderSceneDirect(commandBuffer: commandBuffer, view: view, screenPassDescriptor: screenRenderPassDescriptor)
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
    }
    
    func countUniqueMortonCodes(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Count Unique Morton Codes"
        
        computeEncoder.setComputePipelineState(countUniqueMortonCodesPipelineState)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(uniqueMortonCodeCountBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(uniqueMortonCodeStartIndicesBuffer, offset: 0, index: 2)
        
        var numSpheresVal = UInt32(numSpheres)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 3)
        
        // Reset the count to 0 before dispatching
        var initialCount: UInt32 = 0
        uniqueMortonCodeCountBuffer.contents().copyMemory(from: &initialCount, byteCount: MemoryLayout<UInt32>.size)
        
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((Int(numSpheres) + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }
    
    func aggregateLeafNodes(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Aggregate Leaf Nodes"
        
        computeEncoder.setComputePipelineState(aggregateLeafNodesPipelineState)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(colorTypeBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(octreeLeafNodesBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(uniqueMortonCodeCountBuffer, offset: 0, index: 5)
        computeEncoder.setBuffer(uniqueMortonCodeStartIndicesBuffer, offset: 0, index: 6)
        
        var numSpheresVal = UInt32(numSpheres)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 7)
        
        // Dispatch enough threads for the worst-case scenario (one leaf node per sphere).
        // The kernel will check the actual unique count and exit early if needed.
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((Int(numSpheres) + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        
//         debugPrintMortonCodes()
    }
    
    func debugPrintMortonCodes() {
        guard let mortonBuffer = mortonCodesBuffer,
              let indicesBuffer = indicesBuffer else { return }
        
        let mortonPointer = mortonBuffer.contents().bindMemory(to: UInt64.self, capacity: Int(numSpheres))
        let indicesPointer = indicesBuffer.contents().bindMemory(to: UInt32.self, capacity: Int(numSpheres))
        let positionPointer = positionMassBuffer.contents().bindMemory(to: PositionMass.self, capacity: Int(numSpheres))
        
        let numToPrint = min(10, Int(numSpheres))
        print("=== First \(numToPrint) Morton Codes ===")
        
        for i in 0..<numToPrint {
            let mortonCode = mortonPointer[i]
            let particleIndex = indicesPointer[i]
            let position = positionPointer[i]
            
            print("Particle \(i):")
            print("  Morton Code: \(mortonCode)")
            print("  Particle Index: \(particleIndex)")
            print("  Position: (\(position.position.x), \(position.position.y), \(position.position.z))")
            print("  Mass: \(position.mass)")
            print("---")
        }
        print("================================")
    }
    
    func renderScene(commandBuffer: MTLCommandBuffer, view: MTKView) {
        let sceneRenderPassDescriptor = MTLRenderPassDescriptor()
        sceneRenderPassDescriptor.colorAttachments[0].texture = sceneTexture
        sceneRenderPassDescriptor.colorAttachments[0].loadAction = .clear
        sceneRenderPassDescriptor.colorAttachments[0].storeAction = .store
        sceneRenderPassDescriptor.colorAttachments[0].clearColor = backgroundColor
        
        if let depthTex = view.depthStencilTexture {
            sceneRenderPassDescriptor.depthAttachment.texture = depthTex
            sceneRenderPassDescriptor.depthAttachment.loadAction = .clear
            sceneRenderPassDescriptor.depthAttachment.storeAction = .dontCare
            sceneRenderPassDescriptor.depthAttachment.clearDepth = 1.0
        }
        
        renderSceneToPass(commandBuffer: commandBuffer, renderPassDescriptor: sceneRenderPassDescriptor, label: "Scene Render Pass")
    }
    
    func renderSceneDirect(commandBuffer: MTLCommandBuffer, view: MTKView, screenPassDescriptor: MTLRenderPassDescriptor) {
        // Set the clear color for direct rendering
        screenPassDescriptor.colorAttachments[0].clearColor = backgroundColor
        
        renderSceneToPass(commandBuffer: commandBuffer, renderPassDescriptor: screenPassDescriptor, label: "Direct Scene Render Pass")
    }
    
    private func renderSceneToPass(commandBuffer: MTLCommandBuffer, renderPassDescriptor: MTLRenderPassDescriptor, label: String) {
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
        renderEncoder.label = label
        renderEncoder.setDepthStencilState(depthStencilState)
        renderEncoder.setCullMode(.back)
        
        renderEncoder.setVertexBuffer(globalUniformsBuffer, offset: 0, index: 1)
        renderEncoder.setVertexBuffer(positionMassBuffer, offset: 0, index: 2)
        renderEncoder.setVertexBuffer(velocityRadiusBuffer, offset: 0, index: 3)
        renderEncoder.setVertexBuffer(colorTypeBuffer, offset: 0, index: 4)

        if numStars > 0 {
            renderEncoder.setRenderPipelineState(spherePipelineState)
            for (index, vertexBuffer) in starSphere.vertexBuffers.enumerated() {
                renderEncoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: index)
            }
            let submesh = starSphere.submeshes[0]
            renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType, indexCount: submesh.indexCount, indexType: submesh.indexType, indexBuffer: submesh.indexBuffer.buffer, indexBufferOffset: submesh.indexBuffer.offset, instanceCount: Int(numStars))
        }

        if numPlanets > 0 {
            renderEncoder.setRenderPipelineState(spherePipelineState)
            for (index, vertexBuffer) in planetSphere.vertexBuffers.enumerated() {
                renderEncoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: index)
            }
            let submesh = planetSphere.submeshes[0]
            renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType, indexCount: submesh.indexCount, indexType: submesh.indexType, indexBuffer: submesh.indexBuffer.buffer, indexBufferOffset: submesh.indexBuffer.offset, instanceCount: Int(numPlanets), baseVertex: 0, baseInstance: Int(numStars))
        }
        
        if numDust > 0 {
            renderEncoder.setRenderPipelineState(dustPipelineState)
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
}
