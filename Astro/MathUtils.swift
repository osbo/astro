import simd

func createProjectionMatrix(fov: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let y = 1 / tan(fov * 0.5)
    let x = y / aspect
    let zRange = far / (near - far)
    return simd_float4x4(columns: (
        simd_float4(x, 0, 0, 0),
        simd_float4(0, y, 0, 0),
        simd_float4(0, 0, zRange, -1),
        simd_float4(0, 0, zRange * near, 0)
    ))
}

func translationMatrix(_ t: simd_float3) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m.columns.3 = simd_float4(t, 1)
    return m
}

func scalingMatrix(_ s: Float) -> simd_float4x4 {
    return simd_float4x4(columns: (
        simd_float4(s, 0, 0, 0),
        simd_float4(0, s, 0, 0),
        simd_float4(0, 0, s, 0),
        simd_float4(0, 0, 0, 1)
    ))
}

func lookAt(from eye: SIMD3<Float>, to center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    let z = simd_normalize(eye - center)
    let x = simd_normalize(simd_cross(up, z))
    let y = simd_cross(z, x)

    let rot = simd_float4x4(columns: (
        SIMD4<Float>(x.x, y.x, z.x, 0),
        SIMD4<Float>(x.y, y.y, z.y, 0),
        SIMD4<Float>(x.z, y.z, z.z, 0),
        SIMD4<Float>(0,   0,   0,   1)
    ))

    let trans = translationMatrix(-eye)
    return rot * trans
}
