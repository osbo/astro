import SwiftUI
import MetalKit

struct ContentView: NSViewRepresentable {
    func makeCoordinator() -> Renderer {
        Renderer(self)
    }

    func makeNSView(context: NSViewRepresentableContext<ContentView>) -> MTKView {
        let mtkView = CustomMTKView()
        let renderer = context.coordinator
        mtkView.delegate = renderer
        mtkView.renderer = renderer // Assign renderer here
        mtkView.preferredFramesPerSecond = 120 // Lowered slightly, PP is expensive
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.depthStencilPixelFormat = .invalid // Prevent MTKView from allocating unused depth texture
        mtkView.colorPixelFormat = .rgba16Float // Ensure this matches textures/pipelines
        mtkView.framebufferOnly = false  // Needs to be false for drawable texture access? Check if true works. Often false needed if view provides drawable texture. Let's keep false.

        // Enable HDR output
        if let metalLayer = mtkView.layer as? CAMetalLayer {
            metalLayer.wantsExtendedDynamicRangeContent = true
            if let cs = CGColorSpace(name: "kCGColorSpaceExtendedLinearSRGB" as CFString) {
                metalLayer.colorspace = cs
            }
        }

        // Ensure depth texture is created (needed for scene pass)
        // mtkView.depthStencilPixelFormat = .depth32Float // This line is removed as per the edit hint.

        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device creation failed")
        }
        mtkView.device = metalDevice


        // Initial size setup *before* first draw
        // Get initial frame size correctly
        mtkView.needsLayout = true
        mtkView.layoutSubtreeIfNeeded() // Try to force layout early


        // Crucial for receiving key events immediately
        // Delay slightly maybe?
        DispatchQueue.main.async {
             let success = mtkView.window?.makeFirstResponder(mtkView)
             if success == false {
                 // If window isn't ready, try again later maybe?
                 // Or rely on mouseDown/viewDidMoveToWindow
             }
        }

        return mtkView
    }

    func updateNSView(_ nsView: MTKView, context: NSViewRepresentableContext<ContentView>) {

         // Re-ensure first responder status, especially if window focus changes
         DispatchQueue.main.async {
             if nsView.window?.isKeyWindow ?? false && nsView.window?.firstResponder != nsView {
                 nsView.window?.makeFirstResponder(nsView)
             }
         }
    }

    // --- CustomMTKView ---
    // Keep your CustomMTKView class as it was, handling input.
    // No changes needed inside CustomMTKView for post-processing.
    class CustomMTKView: MTKView {
        var renderer: Renderer! // Renderer must be assigned
        private var lastDragLocation: CGPoint?

        override var acceptsFirstResponder: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
        }

        override func viewDidMoveToSuperview() {
            super.viewDidMoveToSuperview()
        }


        override func acceptsFirstMouse(for event: NSEvent?) -> Bool {
            // Useful for making sure clicks activate the view immediately
            return true
        }

        override func mouseDown(with event: NSEvent) {
             lastDragLocation = convert(event.locationInWindow, from: nil)
        }


        override func mouseDragged(with event: NSEvent) {
            guard let last = lastDragLocation else { return }
            let current = convert(event.locationInWindow, from: nil)
            let delta = CGPoint(x: current.x - last.x, y: current.y - last.y)
            lastDragLocation = current

            let sensitivity: Double = 0.006
            renderer.camera.yaw   -= Double(delta.x) * sensitivity
            renderer.camera.pitch += Double(delta.y) * sensitivity
            renderer.camera.pitch = max(-Double.pi/2 + 0.01, min(Double.pi/2 - 0.01, renderer.camera.pitch))
        }

        override func scrollWheel(with event: NSEvent) {
            let scrollAmount = event.deltaY
            let zoomSensitivity: Float = 10000 // scale up
            let moveVector = renderer.camera.forward * Double(scrollAmount) * Double(zoomSensitivity)
            renderer.camera.targetPosition += moveVector
        }

        override func keyDown(with event: NSEvent) {
            let speed: Float = 100000 // scale up
            switch event.charactersIgnoringModifiers?.lowercased() {
            case "w":
                renderer.camera.targetPosition += SIMD3<Double>(0, 1, 0) * Double(speed)
            case "s":
                renderer.camera.targetPosition -= SIMD3<Double>(0, 1, 0) * Double(speed)
            case "a":
                renderer.camera.targetPosition -= renderer.camera.right * Double(speed)
            case "d":
                renderer.camera.targetPosition += renderer.camera.right * Double(speed)
            case "r":
                if !event.isARepeat {
                    renderer.resetSimulation()
                }
            default:
                 super.keyDown(with: event) // Call super for unhandled keys
            }
        }
    }
}
