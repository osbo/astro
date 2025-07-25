import MetalKit
import MetalPerformanceShaders
import simd
#if canImport(bridge_h)
import bridge_h
#endif

let INVALID_MORTON_CODE: UInt64 = 0xFFFFFFFFFFFFFFFF

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
    var numStars: Int32 = 5
    var numPlanets: Int32 = 200
    var numDust: Int32 = 500000
    
    var numSpheres: Int32 {
        return numStars + numPlanets + numDust
    }
    var sphereCount: Int {
        return Int(numStars) + Int(numPlanets) + Int(numDust)
    }
    // Add this:
    var octreeCount: Int { return Int(numStars) + Int(numPlanets)}
    
    var starRadius: Float = 50000.0
    var planetRadius: Float = 10000.0
    var dustRadius: Float = 1
    var starMass: Float = 10000
    var planetMass: Float = 1000
    var dustMass: Float = 1
    var initialVelocityMaximum: Float = 20000
    var spawnBounds: Float = 1000000
    var backgroundColor: MTLClearColor = MTLClearColor(red: 0/255, green: 0/255, blue: 1/255, alpha: 1.0)
    
    // --- Metal Objects ---
    var starSphere: MTKMesh!
    var planetSphere: MTKMesh!
    
    var spherePipelineState: MTLRenderPipelineState!
    var dustPipelineState: MTLRenderPipelineState!
    var postProcessPipelineState: MTLRenderPipelineState!
    var planetPipelineState: MTLRenderPipelineState! // Add this property to the Renderer class
    
    var globalUniformsBuffer: MTLBuffer!
    var positionMassBuffer: MTLBuffer!
    var velocityRadiusBuffer: MTLBuffer!
    var colorTypeBuffer: MTLBuffer!
    var mortonCodesBuffer: MTLBuffer!
    var unsortedMortonCodesBuffer: MTLBuffer!
    var unsortedIndicesBuffer: MTLBuffer!
    var sortedMortonCodesBuffer: MTLBuffer!
    var sortedIndicesBuffer: MTLBuffer!
    var uniqueIndicesBuffer: MTLBuffer!
    var parentCountBuffer: MTLBuffer!
    var childCountBuffer: MTLBuffer!
    var layerCountBuffer: MTLBuffer!
    var octreeNodesBuffer: MTLBuffer!
    var forceBuffer: MTLBuffer!
    var debugBuffer: MTLBuffer!
    
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
    var barnesHutPipelineState: MTLComputePipelineState!
    var copyAndResetParentCountPipelineState: MTLComputePipelineState!
    var copyLastScanToParentCountPipelineState: MTLComputePipelineState!
    var updateSpheresPipelineState: MTLComputePipelineState!
    var lightingPassPipelineState: MTLComputePipelineState!

    var radixSorter: MetalKernelsRadixSort!
    
    // Floating origin: store world positions as double
    var doublePositions: [SIMD3<Double>] = []
    var positions: [PositionMass] = []
    var velocities: [VelocityRadius] = []
    var colors: [ColorType] = []
    
    // Add arrays to store per-layer offsets and sizes
    let maxLayers: Int = 7
    var layerOffsets: [Int] = []
    var layerSizes: [Int] = []
    var layer: Int = 0
    
    var flagsBuffer: MTLBuffer!
    var scanBuffer: MTLBuffer!
    var scanAuxBuffer: MTLBuffer!
    
    // 1. In class Renderer, add pipeline states:
    var markUniquesPipelineState: MTLComputePipelineState!
    var scatterUniquesPipelineState: MTLComputePipelineState!
    var prefixSumPipelineState: MTLComputePipelineState!
    
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
        radixSorter = MetalKernelsRadixSort(device: device, maxLength: UInt32(numSpheres))
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
        guard let planetFragmentFunction = library.makeFunction(name: "planet_fragment_shader") else {
            fatalError("Could not find planet_fragment_shader function.")
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
        guard let copyAndResetParentCountFunction = library.makeFunction(name: "copyAndResetParentCount") else {
            fatalError("Could not find copyAndResetParentCount function in Octree.metal.")
        }
        guard let copyLastScanToParentCountFunction = library.makeFunction(name: "copyLastScanToParentCount") else {
            fatalError("Could not find copyLastScanToParentCount function in Octree.metal.")
        }
        guard let barnesHutKernel = library.makeFunction(name: "barnesHut") else {
            fatalError("Could not find barnesHut kernel.")
        }
        guard let updateSpheresKernel = library.makeFunction(name: "updateSpheres") else {
            fatalError("Could not find updateSpheres kernel.")
        }
        guard let lightingPassKernel = library.makeFunction(name: "lightingPass") else {
            fatalError("Could not find lightingPass kernel.")
        }
        barnesHutPipelineState = try! device.makeComputePipelineState(function: barnesHutKernel)
        updateSpheresPipelineState = try! device.makeComputePipelineState(function: updateSpheresKernel)
        lightingPassPipelineState = try! device.makeComputePipelineState(function: lightingPassKernel)
        let spherePipelineDescriptor = MTLRenderPipelineDescriptor()
        spherePipelineDescriptor.label = "Sphere Render Pipeline"
        spherePipelineDescriptor.vertexFunction = sphereVertexFunction
        spherePipelineDescriptor.fragmentFunction = fragmentFunction
        spherePipelineDescriptor.vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(starSphere.vertexDescriptor)
        spherePipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        spherePipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        let planetPipelineDescriptor = MTLRenderPipelineDescriptor()
        planetPipelineDescriptor.label = "Planet Render Pipeline"
        planetPipelineDescriptor.vertexFunction = sphereVertexFunction
        planetPipelineDescriptor.fragmentFunction = planetFragmentFunction
        planetPipelineDescriptor.vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(planetSphere.vertexDescriptor)
        planetPipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
        planetPipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
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
            planetPipelineState = try device.makeRenderPipelineState(descriptor: planetPipelineDescriptor)
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
            copyAndResetParentCountPipelineState = try device.makeComputePipelineState(function: copyAndResetParentCountFunction)
            copyLastScanToParentCountPipelineState = try device.makeComputePipelineState(function: copyLastScanToParentCountFunction)
        } catch {
            fatalError("Failed to create pipeline state: \(error)")
        }
        
        // 2. In makePipelines(view:):
        if let markUniquesFunction = library.makeFunction(name: "markUniques") {
            markUniquesPipelineState = try! device.makeComputePipelineState(function: markUniquesFunction)
        }
        if let scatterUniquesFunction = library.makeFunction(name: "scatterUniques") {
            scatterUniquesPipelineState = try! device.makeComputePipelineState(function: scatterUniquesFunction)
        }
        if let prefixSumFunction = library.makeFunction(name: "prefixSum") {
            prefixSumPipelineState = try! device.makeComputePipelineState(function: prefixSumFunction)
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
        // The following buffers are only needed for octreeCount, not sphereCount
        self.mortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * octreeCount, options: .storageModeShared)
        self.mortonCodesBuffer.label = "Morton Codes Buffer"
        self.unsortedMortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * octreeCount, options: .storageModeShared)
        self.unsortedMortonCodesBuffer.label = "Unsorted Morton Codes Buffer"
        self.unsortedIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * octreeCount, options: .storageModeShared)
        self.unsortedIndicesBuffer.label = "Unsorted Indices Buffer"
        self.sortedMortonCodesBuffer = device.makeBuffer(length: MemoryLayout<UInt64>.stride * octreeCount, options: .storageModeShared)
        self.sortedMortonCodesBuffer.label = "Sorted Morton Codes Buffer"
        self.sortedIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * octreeCount, options: .storageModeShared)
        self.sortedIndicesBuffer.label = "Sorted Indices Buffer"
        self.uniqueIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * octreeCount, options: .storageModeShared)
        self.uniqueIndicesBuffer.label = "Unique Indices Buffer"
        self.parentCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        self.parentCountBuffer.label = "Parent Count Buffer"
        self.childCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        self.childCountBuffer.label = "Child Count Buffer"
        self.layerCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        self.layerCountBuffer.label = "Layer Count Buffer"
        self.flagsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * octreeCount, options: .storageModeShared)
        self.flagsBuffer.label = "Flags Buffer"
        self.scanBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * octreeCount, options: .storageModeShared)
        self.scanBuffer.label = "Scan Buffer"
        let scanAuxSize = ((octreeCount + 2 * 512 - 1) / (2 * 512))
        self.scanAuxBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * scanAuxSize, options: .storageModeShared)
        self.scanAuxBuffer.label = "Scan Aux Buffer"
        self.debugBuffer = device.makeBuffer(length: MemoryLayout<simd_float4>.stride * sphereCount, options: .storageModeShared)
        self.debugBuffer.label = "Debug Buffer"
        // Calculate per-layer sizes and offsets for 9 layers, each removing 3 bits (24, 21, ..., 0)
        let maxLayers = 7
        let totalBits = 18
        let bitsPerLayer = 3
        layerSizes = []
        layerOffsets = []
        var offset = 0
        for layer in 0..<maxLayers {
            let bitsRemaining = totalBits - bitsPerLayer * layer
            let possibleValues = bitsRemaining >= 0 ? 1 << bitsRemaining : 1
            let size = min(octreeCount, possibleValues)
            layerSizes.append(size)
            layerOffsets.append(offset)
            offset += size
        }
        self.octreeNodesBuffer = device.makeBuffer(length: MemoryLayout<OctreeNode>.stride * offset, options: .storageModeShared)
        self.octreeNodesBuffer.label = "Octree Nodes Buffer"
        self.forceBuffer = device.makeBuffer(length: MemoryLayout<float3>.stride * sphereCount, options: .storageModeShared)
        self.forceBuffer.label = "Force Buffer"
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
            computeEncoder.setBuffer(buffer, offset: 0, index: 0)
        } else {
            computeEncoder.setComputePipelineState(clearBufferPipelineState)
            computeEncoder.setBuffer(buffer, offset: 0, index: 0)
        }
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
        var factor: Float = 1
        for i in 0..<sphereCount {
            var type: UInt32 = 0
            var mass: Float = 0
            var radius: Float = 0
            var color: simd_float4 = simd_float4(0,0,0,1)
            if i < numStars {
                factor = 1.3
                type = 0
                mass = starMass
                radius = starRadius
                let hue = Float(i) / Float(numStars)
                let oklabColor = simd_float3(0.8, 0.2 * cos(2 * .pi * hue), 0.2 * sin(2 * .pi * hue))
                let srgbColor = oklabToSrgb(oklab: oklabColor)
                color = simd_float4(srgbColor.x, srgbColor.y, srgbColor.z, 1.0)
            } else if i < numStars + numPlanets {
                factor = 2
                type = 1
                mass = planetMass
                radius = planetRadius
                color = simd_float4(0, 0, 0, 1)
            } else {
                factor = 2
                type = 2
                mass = dustMass
                radius = dustRadius
                color = simd_float4(0, 0, 0, 0)
            }
            let position = SIMD3<Double>(
                Double.random(in: -Double(spawnBounds * factor)...Double(spawnBounds * factor)),
                Double.random(in: -Double(spawnBounds * factor)...Double(spawnBounds * factor)),
                Double.random(in: -Double(spawnBounds * factor)...Double(spawnBounds * factor))
            )
            doublePositions.append(position)
            let velocity = simd_float3(
                Float.random(in: -initialVelocityMaximum...initialVelocityMaximum),
                Float.random(in: -initialVelocityMaximum...initialVelocityMaximum),
                Float.random(in: -initialVelocityMaximum...initialVelocityMaximum)
            )
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
        self.layer = 0
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "Main Command Buffer"
        
        if octreeCount > 0 {
            commandBuffer.pushDebugGroup("Layer 0: Leaf Nodes")
            clearBuffer(commandBuffer: commandBuffer, buffer: sortedMortonCodesBuffer, count: octreeCount, dataType: UInt64.self)
            clearBuffer(commandBuffer: commandBuffer, buffer: sortedIndicesBuffer, count: octreeCount, dataType: UInt32.self)
            clearOctreeNodeBuffer(commandBuffer: commandBuffer, buffer: octreeNodesBuffer, count: layerOffsets.last! + layerSizes.last!)
            generateMortonCodes(commandBuffer: commandBuffer, count: octreeCount)
            fillIndices(commandBuffer: commandBuffer, buffer: unsortedIndicesBuffer, startIdx: 0, endIdx: octreeCount)
            let parentCountPointer = parentCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
            parentCountPointer[0] = UInt32(octreeCount)
            clearBuffer(commandBuffer: commandBuffer, buffer: sortedMortonCodesBuffer, count: octreeCount, dataType: UInt64.self)
            clearBuffer(commandBuffer: commandBuffer, buffer: sortedIndicesBuffer, count: octreeCount, dataType: UInt32.self)
            radixSorter.sort(commandBuffer: commandBuffer,
                             input: unsortedMortonCodesBuffer,
                             inputIndices: unsortedIndicesBuffer,
                             output: sortedMortonCodesBuffer,
                             outputIndices: sortedIndicesBuffer,
                             estimatedLength: UInt32(octreeCount),
                             actualCountBuffer: parentCountBuffer)
            countUniqueMortonCodes(commandBuffer: commandBuffer, aggregate: false, count: octreeCount)
            aggregateLeafNodes(commandBuffer: commandBuffer, count: octreeCount)
            commandBuffer.popDebugGroup()
            self.layer = 1 // Start at first internal layer
            buildInternalNodes(commandBuffer: commandBuffer, count: octreeCount)
        }
        // Barnes-Hut and rendering still use sphereCount
        if sphereCount > 0 {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            barnesHut(computeEncoder: computeEncoder)
            lightingPass(computeEncoder: computeEncoder)
            updateSpheres(computeEncoder: computeEncoder, dt: deltaTime)
            computeEncoder.endEncoding()
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
    func generateMortonCodes(commandBuffer: MTLCommandBuffer, count: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Generate Morton Codes"
        computeEncoder.setComputePipelineState(generateMortonCodesPipelineState)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(mortonCodesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(unsortedMortonCodesBuffer, offset: 0, index: 2)
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((count + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        self.layer = 0
    }

    func countUniqueMortonCodes(commandBuffer: MTLCommandBuffer, aggregate: Bool, count: Int) {
        // Only use the scan-based unique counting sequence
        clearBuffer(commandBuffer: commandBuffer, buffer: flagsBuffer, count: count, dataType: UInt32.self)
        clearBuffer(commandBuffer: commandBuffer, buffer: scanBuffer, count: count, dataType: UInt32.self)
        clearBuffer(commandBuffer: commandBuffer, buffer: uniqueIndicesBuffer, count: count, dataType: UInt32.self)
        // 1. markUniques
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.label = "Mark Uniques"
            computeEncoder.setComputePipelineState(markUniquesPipelineState)
            computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(flagsBuffer, offset: 0, index: 1)
            var numSpheresVal = UInt32(count)
            computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 2)
            var aggregateVal: UInt32 = aggregate ? 1 : 0
            computeEncoder.setBytes(&aggregateVal, length: MemoryLayout<UInt32>.size, index: 3)
            let threadGroupSize = MTLSizeMake(64, 1, 1)
            let threadGroups = MTLSizeMake((count + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            computeEncoder.endEncoding()
        }
        // 2. multi-pass prefixSum
        if let compute = commandBuffer.makeComputeCommandEncoder() {
            compute.label = "Prefix Sum"
            radixSorter.scanKernel.encodeScanTo(compute, input: flagsBuffer, output: scanBuffer, length: UInt32(count))
            compute.endEncoding()
        }
        // 3. scatterUniques
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.label = "Scatter Uniques"
            computeEncoder.setComputePipelineState(scatterUniquesPipelineState)
            computeEncoder.setBuffer(flagsBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(scanBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(uniqueIndicesBuffer, offset: 0, index: 2)
            var numSpheresVal = UInt32(count)
            computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 3)
            let threadGroupSize = MTLSizeMake(64, 1, 1)
            let threadGroups = MTLSizeMake((count + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            computeEncoder.endEncoding()
        }
        // 4. copyLastScanToParentCount (GPU-side unique count)
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            computeEncoder.label = "Copy Last Scan To Parent Count"
            computeEncoder.setComputePipelineState(copyLastScanToParentCountPipelineState)
            computeEncoder.setBuffer(scanBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(flagsBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(parentCountBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(childCountBuffer, offset: 0, index: 3)
            var countVal = UInt32(count)
            computeEncoder.setBytes(&countVal, length: MemoryLayout<UInt32>.size, index: 4)
            let threadGroupSize = MTLSizeMake(1, 1, 1)
            let threadGroups = MTLSizeMake(1, 1, 1)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            computeEncoder.endEncoding()
        }
    }

    func unsortedLeafMortonCodes(commandBuffer: MTLCommandBuffer) {
        fillIndices(commandBuffer: commandBuffer, buffer: uniqueIndicesBuffer, startIdx: 0, endIdx: octreeCount)
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Unsorted Leaf Morton Codes"
        computeEncoder.setComputePipelineState(unsortedLeafPipelineState)
        var numSpheresVal = UInt32(octreeCount)
        computeEncoder.setBuffer(parentCountBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(layerCountBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 2)
        computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        computeEncoder.endEncoding()
        self.layer += 1
    }

    func unsortedInternalMortonCodes(commandBuffer: MTLCommandBuffer, layer: Int) {
        let count = max(1, Int(ceil(Double(octreeCount) / pow(8.0, Double(layer)))))
        fillIndices(commandBuffer: commandBuffer, buffer: uniqueIndicesBuffer, startIdx: 0, endIdx: count)
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Unsorted Internal Morton Codes Layer \(layer)"
        computeEncoder.setComputePipelineState(unsortedInternalPipelineState)
        computeEncoder.setBuffer(parentCountBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(layerCountBuffer, offset: 0, index: 1)
        computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        computeEncoder.endEncoding()
        self.layer += 1
    }
    
    func aggregateLeafNodes(commandBuffer: MTLCommandBuffer, count: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Aggregate Leaf Nodes"
        computeEncoder.setComputePipelineState(aggregateLeafNodesPipelineState)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(uniqueIndicesBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(colorTypeBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(parentCountBuffer, offset: 0, index: 5)
        computeEncoder.setBuffer(octreeNodesBuffer, offset: 0, index: 6)
        computeEncoder.setBuffer(unsortedMortonCodesBuffer, offset: 0, index: 7)
        computeEncoder.setBuffer(unsortedIndicesBuffer, offset: 0, index: 8)
        // let numNodes = parentCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
        // if numNodes == 0 {
        //     computeEncoder.endEncoding()
        //     return
        // }
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((Int(count) + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }

    func aggregateInternalNodes(commandBuffer: MTLCommandBuffer, inputOffset: Int, outputOffset: Int, nodeCount: Int) {
        // Prevent out-of-bounds access for the last layer
        guard self.layer < layerSizes.count else { return }
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Aggregate Internal Nodes Layer \(self.layer)"
        if nodeCount <= 0 {
            computeEncoder.endEncoding()
            return
        }

        computeEncoder.setComputePipelineState(aggregateNodesPipelineState)
        computeEncoder.setBuffer(octreeNodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedMortonCodesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(sortedIndicesBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(uniqueIndicesBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(parentCountBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(childCountBuffer, offset: 0, index: 12)
        var inputOffsetVal = UInt32(inputOffset)
        var outputOffsetVal = UInt32(outputOffset)
        var inputLayerSizeVal = UInt32(layerSizes[self.layer])
        var outputLayerSizeVal: UInt32 = 0
        if self.layer + 1 < layerSizes.count {
            outputLayerSizeVal = UInt32(layerSizes[self.layer + 1])
        }
        var layerVal = UInt32(self.layer)
        computeEncoder.setBytes(&inputOffsetVal, length: MemoryLayout<UInt32>.size, index: 5)
        computeEncoder.setBytes(&outputOffsetVal, length: MemoryLayout<UInt32>.size, index: 6)
        computeEncoder.setBytes(&layerVal, length: MemoryLayout<UInt32>.size, index: 7)
        computeEncoder.setBytes(&inputLayerSizeVal, length: MemoryLayout<UInt32>.size, index: 8)
        computeEncoder.setBytes(&outputLayerSizeVal, length: MemoryLayout<UInt32>.size, index: 9)
        computeEncoder.setBuffer(unsortedMortonCodesBuffer, offset: 0, index: 10)
        computeEncoder.setBuffer(unsortedIndicesBuffer, offset: 0, index: 11)
        let threadGroupSize = MTLSizeMake(64, 1, 1)
        let threadGroups = MTLSizeMake((nodeCount + threadGroupSize.width - 1) / threadGroupSize.width, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

//        print("inputOffset: \(inputOffsetVal), outputOffset: \(outputOffsetVal), layer: \(layerVal), inputLayerSize: \(inputLayerSizeVal), outputLayerSize: \(outputLayerSizeVal)")
    }

    func buildInternalNodes(commandBuffer: MTLCommandBuffer, count: Int) {
        while self.layer < maxLayers{
            commandBuffer.pushDebugGroup("Layer \(self.layer): Internal Nodes")
            let inputNodeCount = layerSizes[self.layer - 1]
            let nodeCount = layerSizes[self.layer]
            let inputOffset = layerOffsets[self.layer - 1]
            let outputOffset = layerOffsets[self.layer]
//            print("Building internal nodes for layer \(self.layer) with \(nodeCount) nodes, input offset: \(inputOffset), output offset: \(outputOffset)")

            // 1. Clear buffers for this layer's sort
           clearBuffer(commandBuffer: commandBuffer, buffer: sortedMortonCodesBuffer, count: count, dataType: UInt64.self)
           clearBuffer(commandBuffer: commandBuffer, buffer: sortedIndicesBuffer, count: count, dataType: UInt32.self)

        //     // 2. Sort the Morton codes for this layer
            radixSorter.sort(commandBuffer: commandBuffer,
                             input: unsortedMortonCodesBuffer,
                             inputIndices: unsortedIndicesBuffer,
                             output: sortedMortonCodesBuffer,
                             outputIndices: sortedIndicesBuffer,
                             estimatedLength: UInt32(inputNodeCount),
                             actualCountBuffer: parentCountBuffer)

//            print("inputNodeCount: \(inputNodeCount)")

            // 3. Count unique Morton codes for this layer
            countUniqueMortonCodes(commandBuffer: commandBuffer, aggregate: true, count: count)

            // 4. Clear buffers for next unsorted output
            clearBuffer(commandBuffer: commandBuffer, buffer: unsortedMortonCodesBuffer, count: count, dataType: UInt64.self)
            clearBuffer(commandBuffer: commandBuffer, buffer: unsortedIndicesBuffer, count: count, dataType: UInt32.self)

            // 5. Aggregate nodes for this layer
            aggregateInternalNodes(commandBuffer: commandBuffer,
                                   inputOffset: inputOffset,
                                   outputOffset: outputOffset,
                                   nodeCount: nodeCount)

            // Only increment layer after successful aggregation
            self.layer += 1
            commandBuffer.popDebugGroup()
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
        renderEncoder.setVertexBuffer(forceBuffer, offset: 0, index: 5)
        
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
            if lastPipeline !== planetPipelineState {
                renderEncoder.setRenderPipelineState(planetPipelineState)
                lastPipeline = planetPipelineState
            }
            for (index, vertexBuffer) in planetSphere.vertexBuffers.enumerated() {
                renderEncoder.setVertexBuffer(vertexBuffer.buffer, offset: vertexBuffer.offset, index: index)
            }
            renderEncoder.setFragmentBuffer(positionMassBuffer, offset: 0, index: 2)
            renderEncoder.setFragmentBuffer(colorTypeBuffer, offset: 0, index: 3)
            var numStarsVal = UInt32(numStars)
            renderEncoder.setFragmentBytes(&numStarsVal, length: MemoryLayout<UInt32>.size, index: 4)
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

    func barnesHut(computeEncoder: MTLComputeCommandEncoder) {
        computeEncoder.label = "Barnes-Hut"
        computeEncoder.setComputePipelineState(barnesHutPipelineState)
        computeEncoder.setBuffer(octreeNodesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(velocityRadiusBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(colorTypeBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(forceBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(mortonCodesBuffer, offset: 0, index: 5)
        var rootNodeIndexVal = UInt32(layerOffsets.last!)
        computeEncoder.setBytes(&rootNodeIndexVal, length: MemoryLayout<UInt32>.size, index: 6)
        var numSpheresVal = UInt32(sphereCount)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 7)
        var thetaVal: Float = 0.7
        computeEncoder.setBytes(&thetaVal, length: MemoryLayout<Float>.size, index: 8)
        // computeEncoder.setBuffer(debugBuffer, offset: 0, index: 9)
        let threadGroupSize = MTLSize(width: 64, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (sphereCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        // print("Debug buffer:")
        // let ptr = debugBuffer.contents().bindMemory(to: simd_float4.self, capacity: sphereCount)
        // let debugArray = Array(UnsafeBufferPointer(start: ptr, count: 100))
        // for value in debugArray {
        //     print(value)
        // }
    }

    func lightingPass(computeEncoder: MTLComputeCommandEncoder) {
        computeEncoder.label = "Lighting Pass"
        computeEncoder.setComputePipelineState(lightingPassPipelineState)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(colorTypeBuffer, offset: 0, index: 1)
        var numSpheresVal = UInt32(sphereCount)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 2)
        var numStarsVal = UInt32(numStars)
        computeEncoder.setBytes(&numStarsVal, length: MemoryLayout<UInt32>.size, index: 3)
        let threadGroupSize = MTLSize(width: 64, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (sphereCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }

    func updateSpheres(computeEncoder: MTLComputeCommandEncoder, dt: Float) {
        computeEncoder.label = "Update Spheres"
        computeEncoder.setComputePipelineState(updateSpheresPipelineState)
        computeEncoder.setBuffer(positionMassBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(velocityRadiusBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(forceBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(colorTypeBuffer, offset: 0, index: 3)
        var dtVal = dt
        computeEncoder.setBytes(&dtVal, length: MemoryLayout<Float>.size, index: 4)
        var numSpheresVal = UInt32(sphereCount)
        computeEncoder.setBytes(&numSpheresVal, length: MemoryLayout<UInt32>.size, index: 5)
        let threadGroupSize = MTLSize(width: 64, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (sphereCount + threadGroupSize.width - 1) / threadGroupSize.width, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }

    func copyAndResetParentCount(commandBuffer: MTLCommandBuffer) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Copy and Reset Parent Count"
        computeEncoder.setComputePipelineState(copyAndResetParentCountPipelineState)
        computeEncoder.setBuffer(parentCountBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(childCountBuffer, offset: 0, index: 1)
        let threadGroupSize = MTLSizeMake(1, 1, 1)
        let threadGroups = MTLSizeMake(1, 1, 1)
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
    }
}
