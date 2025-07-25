import simd
import Foundation

class Camera {
    var position = SIMD3<Double>(0, 0, 1800000)
    var targetPosition = SIMD3<Double>(0, 0, 1800000)
    var yaw: Double = .pi
    var pitch: Double = 0

    var forward: SIMD3<Double> {
        let x = cos(pitch) * sin(yaw)
        let y = sin(pitch)
        let z = cos(pitch) * cos(yaw)
        return simd_normalize(SIMD3<Double>(x, y, z))
    }

    var right: SIMD3<Double> {
        return simd_normalize(simd_cross(forward, [0, 1, 0]))
    }

    var up: SIMD3<Double> {
        return simd_cross(right, forward)
    }

    func update(deltaTime: Float) {
        let factor: Double = 1 - pow(0.95, Double(deltaTime) * 60)
        position += (targetPosition - position) * factor
    }

    // Floating origin: always return view matrix from (0,0,0) in camera space
    func floatViewMatrix() -> simd_float4x4 {
        // Camera at origin, looking in forward direction
        let eye = SIMD3<Double>(0, 0, 0)
        let center = forward
        let upVec = up
        let viewD = lookAtD(from: eye, to: center, up: upVec)
        // Convert each column from double to float
        return simd_float4x4(
            simd_float4(Float(viewD.columns.0.x), Float(viewD.columns.0.y), Float(viewD.columns.0.z), Float(viewD.columns.0.w)),
            simd_float4(Float(viewD.columns.1.x), Float(viewD.columns.1.y), Float(viewD.columns.1.z), Float(viewD.columns.1.w)),
            simd_float4(Float(viewD.columns.2.x), Float(viewD.columns.2.y), Float(viewD.columns.2.z), Float(viewD.columns.2.w)),
            simd_float4(Float(viewD.columns.3.x), Float(viewD.columns.3.y), Float(viewD.columns.3.z), Float(viewD.columns.3.w))
        )
    }

    // Helper to get camera position as float3 for GPU
    var floatPosition: simd_float3 {
        return simd_float3(Float(position.x), Float(position.y), Float(position.z))
    }
}

// Double-precision lookAt
func lookAtD(from eye: SIMD3<Double>, to center: SIMD3<Double>, up: SIMD3<Double>) -> simd_double4x4 {
    let z = simd_normalize(eye - center)
    let x = simd_normalize(simd_cross(up, z))
    let y = simd_cross(z, x)

    let rot = simd_double4x4(columns: (
        SIMD4<Double>(x.x, y.x, z.x, 0),
        SIMD4<Double>(x.y, y.y, z.y, 0),
        SIMD4<Double>(x.z, y.z, z.z, 0),
        SIMD4<Double>(0,   0,   0,   1)
    ))
    let trans = translationMatrixD(-eye)
    return rot * trans
}

func translationMatrixD(_ t: SIMD3<Double>) -> simd_double4x4 {
    var m = matrix_identity_double4x4
    m.columns.3 = SIMD4<Double>(t, 1)
    return m
}
