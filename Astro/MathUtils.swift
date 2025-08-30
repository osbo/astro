import simd

func createProjectionMatrix(fov: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let y = 1.0 / tan(fov * 0.5)
    let x = y / aspect
    let z = far / (near - far)
    return simd_float4x4(
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, -1],
        [0, 0, z * near, 0]
    )
}


func oklabToSrgb(oklab: simd_float3) -> simd_float3 {
    let l = oklab.x
    let a = oklab.y
    let b = oklab.z

    let l_ = l + 0.3963377774 * a + 0.2158037573 * b
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b

    let l_cubed = l_ * l_ * l_
    let m_cubed = m_ * m_ * m_
    let s_cubed = s_ * s_ * s_

    return simd_float3(
        +4.0767416621 * l_cubed - 3.3077115913 * m_cubed + 0.2309699292 * s_cubed,
        -1.2684380046 * l_cubed + 2.6097574011 * m_cubed - 0.3413193945 * s_cubed,
        -0.0041960863 * l_cubed - 0.7034186147 * m_cubed + 1.7076147010 * s_cubed
    )
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
