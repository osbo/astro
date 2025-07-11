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

    var pipelinesCreated = false

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
        let fullResDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: Int(size.width), height: Int(size.height), mipmapped: false)
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

        let spherePipelineDescriptor = MTLRenderPipelineDescriptor()
        spherePipelineDescriptor.label = "Sphere Render Pipeline"
        spherePipelineDescriptor.vertexFunction = sphereVertexFunction
        spherePipelineDescriptor.fragmentFunction = fragmentFunction
        spherePipelineDescriptor.vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(starSphere.vertexDescriptor)
        spherePipelineDescriptor.colorAttachments[0].pixelFormat = sceneTexture.pixelFormat
        spherePipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        
        let dustPipelineDescriptor = MTLRenderPipelineDescriptor()
        dustPipelineDescriptor.label = "Dust Render Pipeline"
        dustPipelineDescriptor.vertexFunction = dustVertexFunction
        dustPipelineDescriptor.fragmentFunction = fragmentFunction
        dustPipelineDescriptor.colorAttachments[0].pixelFormat = sceneTexture.pixelFormat
        dustPipelineDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        
        let postProcessPipelineDescriptor = MTLRenderPipelineDescriptor()
        postProcessPipelineDescriptor.label = "Post-Process Pipeline"
        postProcessPipelineDescriptor.vertexFunction = postProcessVertexFunction
        postProcessPipelineDescriptor.fragmentFunction = postProcessFragmentFunction
        postProcessPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        
        do {
            spherePipelineState = try device.makeRenderPipelineState(descriptor: spherePipelineDescriptor)
            dustPipelineState = try device.makeRenderPipelineState(descriptor: dustPipelineDescriptor)
            postProcessPipelineState = try device.makeRenderPipelineState(descriptor: postProcessPipelineDescriptor)
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
        
        renderScene(commandBuffer: commandBuffer, view: view)
        postProcess(commandBuffer: commandBuffer, source: sceneTexture, screenPassDescriptor: screenRenderPassDescriptor)
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
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
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: sceneRenderPassDescriptor) else { return }
        renderEncoder.label = "Scene Render Pass"
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
