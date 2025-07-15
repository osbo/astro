import simd
import Foundation

class Camera {
    var position = SIMD3<Float>(0, 0, 500)
    var targetPosition = SIMD3<Float>(0, 0, 500)
    var yaw: Float = .pi
    var pitch: Float = 0

    var forward: SIMD3<Float> {
        let x = cos(pitch) * sin(yaw)
        let y = sin(pitch)
        let z = cos(pitch) * cos(yaw)
        return simd_normalize(SIMD3<Float>(x, y, z))
    }

    var right: SIMD3<Float> {
        return simd_normalize(simd_cross(forward, [0, 1, 0]))
    }

    var up: SIMD3<Float> {
        return simd_cross(right, forward)
    }

    func update(deltaTime: Float) {
        let factor: Float = 1 - pow(0.95, deltaTime * 60)
        position += (targetPosition - position) * factor
    }

    func viewMatrix() -> simd_float4x4 {
        return lookAt(from: position, to: position + forward, up: up)
    }
}
