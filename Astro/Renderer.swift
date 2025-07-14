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
    var numPlanets: Int32 = 100
    var numDust: Int32 = 1000000
    
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
    var spherePositionsBuffer: MTLBuffer!
    var sphereVelocityBuffer: MTLBuffer!
    var mortonCodesBuffer: MTLBuffer!
    var indicesBuffer: MTLBuffer!
    var sortedMortonCodesBuffer: MTLBuffer!
    var sortedIndicesBuffer: MTLBuffer!

    var pipelinesCreated = false
    var computePipelineState: MTLComputePipelineState!
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
        
        let postProcessPipelineDescriptor = MTLRenderPipelineDescriptor()
        postProcessPipelineDescriptor.label = "Post-Process Pipeline"
        postProcessPipelineDescriptor.vertexFunction = postProcessVertexFunction
        postProcessPipelineDescriptor.fragmentFunction = postProcessFragmentFunction
        postProcessPipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        
        do {
            spherePipelineState = try device.makeRenderPipelineState(descriptor: spherePipelineDescriptor)
            dustPipelineState = try device.makeRenderPipelineState(descriptor: dustPipelineDescriptor)
            postProcessPipelineState = try device.makeRenderPipelineState(descriptor: postProcessPipelineDescriptor)
            computePipelineState = try device.makeComputePipelineState(function: generateMortonCodesFunction)
        } catch {
            fatalError("Failed to create pipeline state: \(error)")
        }
    }
    
    func makeBuffers() {
        let sphereCount = Int(numSpheres)
        guard sphereCount > 0 else { return }
        self.globalUniformsBuffer = device.makeBuffer(length: MemoryLayout<GlobalUniforms>.stride, options: .storageModeShared)
        self.globalUniformsBuffer.label = "Global Uniforms Buffer"
        
        self.spherePositionsBuffer = device.makeBuffer(length: MemoryLayout<simd_int4>.stride * sphereCount, options: .storageModeShared)
        self.spherePositionsBuffer.label = "Sphere Positions and Mass"
        
        self.sphereVelocityBuffer = device.makeBuffer(length: MemoryLayout<simd_int4>.stride * sphereCount, options: .storageModeShared)
        self.sphereVelocityBuffer.label = "Sphere Velocities and Radii"
        
        self.mortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * sphereCount, options: .storageModeShared)
        self.mortonCodesBuffer.label = "Morton Codes Buffer"
        
        self.indicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.indicesBuffer.label = "Indices Buffer"
        
        self.sortedMortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * sphereCount, options: .storageModeShared)
        self.sortedMortonCodesBuffer.label = "Sorted Morton Codes Buffer"
        
        self.sortedIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * sphereCount, options: .storageModeShared)
        self.sortedIndicesBuffer.label = "Sorted Indices Buffer"
    }
    
    func makeSpheres() {
        var spherePositions: [simd_int4] = []
        var sphereVelocities: [simd_int4] = []
        
        for _ in 0..<numSpheres {
            let position = simd_int4(Int32.random(in: -spawnBounds...spawnBounds), Int32.random(in: -spawnBounds...spawnBounds), Int32.random(in: -spawnBounds...spawnBounds), 0)
            let velocity = simd_int4(Int32.random(in: -initialVelocityMaximum...initialVelocityMaximum), Int32.random(in: -initialVelocityMaximum...initialVelocityMaximum), Int32.random(in: -initialVelocityMaximum...initialVelocityMaximum), 0)
            spherePositions.append(position)
            sphereVelocities.append(velocity)
        }
        
        for i in 0..<Int(numStars) {
            spherePositions[i].w = starMass
            sphereVelocities[i].w = starRadius
        }
        for i in Int(numStars)..<Int(numStars + numPlanets) {
            spherePositions[i].w = planetMass
            sphereVelocities[i].w = planetRadius
        }
        for i in Int(numStars + numPlanets)..<Int(numSpheres) {
            spherePositions[i].w = dustMass
            sphereVelocities[i].w = dustRadius
        }
        
        spherePositionsBuffer.contents().copyMemory(from: spherePositions, byteCount: MemoryLayout<simd_int4>.stride * spherePositions.count)
        sphereVelocityBuffer.contents().copyMemory(from: sphereVelocities, byteCount: MemoryLayout<simd_int4>.stride * sphereVelocities.count)
        
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
        
        // Generate Morton codes for all particles at the beginning of every frame
        generateMortonCodes(commandBuffer: commandBuffer)
        
        // Sort the Morton codes
        radixSorter.sort(commandBuffer: commandBuffer, input: mortonCodesBuffer, inputIndices: indicesBuffer, output: sortedMortonCodesBuffer, outputIndices: sortedIndicesBuffer, length: UInt32(numSpheres))
        
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
        
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(spherePositionsBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(mortonCodesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(indicesBuffer, offset: 0, index: 2)
        
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
        let positionPointer = spherePositionsBuffer.contents().bindMemory(to: simd_int4.self, capacity: Int(numSpheres))
        
        let numToPrint = min(10, Int(numSpheres))
        print("=== First \(numToPrint) Morton Codes ===")
        
        for i in 0..<numToPrint {
            let mortonCode = mortonPointer[i]
            let particleIndex = indicesPointer[i]
            let position = positionPointer[i]
            
            print("Particle \(i):")
            print("  Morton Code: \(mortonCode)")
            print("  Particle Index: \(particleIndex)")
            print("  Position: (\(position.x), \(position.y), \(position.z))")
            print("  Mass: \(position.w)")
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
        renderEncoder.setVertexBuffer(spherePositionsBuffer, offset: 0, index: 2)
        renderEncoder.setVertexBuffer(sphereVelocityBuffer, offset: 0, index: 3)

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
