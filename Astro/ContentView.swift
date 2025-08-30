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
        mtkView.renderer = renderer
        mtkView.preferredFramesPerSecond = 120
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.depthStencilPixelFormat = .invalid
        mtkView.colorPixelFormat = .rgba16Float
        mtkView.framebufferOnly = false


        if let metalLayer = mtkView.layer as? CAMetalLayer {
            metalLayer.wantsExtendedDynamicRangeContent = true
            if let cs = CGColorSpace(name: "kCGColorSpaceExtendedLinearSRGB" as CFString) {
                metalLayer.colorspace = cs
            }
        }



        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal device creation failed")
        }
        mtkView.device = metalDevice



        mtkView.needsLayout = true
        mtkView.layoutSubtreeIfNeeded()



        DispatchQueue.main.async {
             let success = mtkView.window?.makeFirstResponder(mtkView)
             if success == false {
             }
        }

        return mtkView
    }

    func updateNSView(_ nsView: MTKView, context: NSViewRepresentableContext<ContentView>) {


         DispatchQueue.main.async {
             if nsView.window?.isKeyWindow ?? false && nsView.window?.firstResponder != nsView {
                 nsView.window?.makeFirstResponder(nsView)
             }
         }
    }


    class CustomMTKView: MTKView {
        var renderer: Renderer!
        private var lastDragLocation: CGPoint?

        override var acceptsFirstResponder: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
        }

        override func viewDidMoveToSuperview() {
            super.viewDidMoveToSuperview()
        }


        override func acceptsFirstMouse(for event: NSEvent?) -> Bool {
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
            let zoomSensitivity: Float = 10000
            let moveVector = renderer.camera.forward * Double(scrollAmount) * Double(zoomSensitivity)
            renderer.camera.targetPosition += moveVector
        }

        override func keyDown(with event: NSEvent) {
            let speed: Float = 100000
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
                 super.keyDown(with: event)
            }
        }
    }
}
