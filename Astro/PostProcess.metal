#include <metal_stdlib>

using namespace metal;

// A struct to pass data from the post-process vertex shader to the fragment shader.
struct PostProcessVertexOut {
    float4 position [[position]];
    float2 texCoords;
};

/**
 * @brief Vertex shader for a full-screen quad.
 *
 * This shader generates the four vertices of a quad that covers the entire screen
 * and also provides the corresponding texture coordinates for each vertex.
 *
 * @param vid The vertex ID, automatically provided by Metal (0, 1, 2, or 3).
 * @return The vertex data, including clip-space position and texture coordinates.
 */
vertex PostProcessVertexOut postProcessVertex(uint vid [[vertex_id]]) {
    PostProcessVertexOut out;
    
    // The quad vertices in homogeneous clip space.
    float4 positions[4] = {
        float4(-1.0, -1.0, 0.0, 1.0), // Bottom-left
        float4( 1.0, -1.0, 0.0, 1.0), // Bottom-right
        float4(-1.0,  1.0, 0.0, 1.0), // Top-left
        float4( 1.0,  1.0, 0.0, 1.0)  // Top-right
    };
    
    // The corresponding texture coordinates for each vertex.
    // (0,0) is top-left, (1,1) is bottom-right.
    float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    
    out.position = positions[vid];
    out.texCoords = texCoords[vid]; // Pass the texture coordinates to the fragment shader.
    
    return out;
}

/**
 * @brief Fragment shader that samples a texture and displays it.
 *
 * This shader takes the texture coordinates from the vertex shader and samples
 * the source texture (your rendered scene) at that location.
 *
 * @param in The interpolated data from the vertex shader.
 * @param sourceTexture The texture containing the rendered 3D scene.
 * @param sourceSampler A sampler to define how the texture is read.
 * @return The final color for the pixel on the screen.
 */
fragment float4 postProcessFragment(PostProcessVertexOut in [[stage_in]],
                                    texture2d<float> sourceTexture [[texture(0)]],
                                    sampler sourceSampler [[sampler(0)]])
{
    // Sample the texture at the coordinates provided by the vertex shader.
    return sourceTexture.sample(sourceSampler, in.texCoords);
}
