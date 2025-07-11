#include <metal_stdlib>
#include <simd/simd.h>
#include "bridge.h"

using namespace metal;
struct VertexOut {
    float4 position [[position]];
};

struct DustVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
};

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

vertex VertexOut sphere_vertex(VertexIn in [[stage_in]],
                               constant GlobalUniforms &globalUniforms [[buffer(1)]],
                               device const int4 *spherePositions [[buffer(2)]],
                               device const int4 *sphereVelocities [[buffer(3)]],
                               uint instanceID [[instance_id]])
{
    VertexOut out;
    int4 positionData = spherePositions[instanceID];
    int4 velocityData = sphereVelocities[instanceID];
    float radius = float(velocityData.w);
    float3 instancePosition = float3(positionData.x, positionData.y, positionData.z);
    float3 worldPosition = (in.position * radius) + instancePosition;

    out.position = globalUniforms.projectionMatrix * globalUniforms.viewMatrix * float4(worldPosition, 1.0);
    
    return out;
}

vertex DustVertexOut dust_point_vertex(uint vertexID [[vertex_id]],
                                  uint instanceID [[instance_id]],
                                  constant GlobalUniforms &globalUniforms [[buffer(1)]],
                                  device const int4 *spherePositions [[buffer(2)]],
                                  device const int4 *sphereVelocities [[buffer(3)]])
{
    DustVertexOut out;
    int4 positionData = spherePositions[instanceID];
    int4 velocityData = sphereVelocities[instanceID];
    float radius = float(velocityData.w);
    float3 particleCenterWorld = float3(positionData.x, positionData.y, positionData.z);
    
    out.position = globalUniforms.projectionMatrix * globalUniforms.viewMatrix * float4(particleCenterWorld, 1.0);
    
    // Set point size based on radius
    out.pointSize = radius;

    return out;
}

fragment float4 fragment_shader(VertexOut in [[stage_in]])
{
    return float4(1.0, 1.0, 1.0, 1.0);
}
