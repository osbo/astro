#include <metal_stdlib>

using namespace metal;


struct PostProcessVertexOut {
    float4 position [[position]];
    float2 texCoords;
};


vertex PostProcessVertexOut postProcessVertex(uint vid [[vertex_id]]) {
    PostProcessVertexOut out;
    

    float4 positions[4] = {
        float4(-1.0, -1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0),
        float4(-1.0,  1.0, 0.0, 1.0),
        float4( 1.0,  1.0, 0.0, 1.0)
    };
    

    float2 texCoords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    
    out.position = positions[vid];
    out.texCoords = texCoords[vid];
    
    return out;
}


fragment float4 postProcessFragment(PostProcessVertexOut in [[stage_in]],
                                    texture2d<float> sourceTexture [[texture(0)]],
                                    sampler sourceSampler [[sampler(0)]])
{

    return sourceTexture.sample(sourceSampler, in.texCoords);
}
