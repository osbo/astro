#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float4 color;
    uint instanceID;
};

struct DustVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
};

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

vertex VertexOut sphere_vertex(VertexIn in [[stage_in]],
                               constant GlobalUniforms &globalUniforms [[buffer(1)]],
                               device const PositionMass *positions [[buffer(2)]],
                               device const VelocityRadius *velocities [[buffer(3)]],
                               device const ColorType *colors [[buffer(4)]],
                               uint instanceID [[instance_id]])
{
    VertexOut out;
    PositionMass positionData = positions[instanceID];
    VelocityRadius velocityData = velocities[instanceID];
    ColorType colorData = colors[instanceID];

    float3 worldPosition = (in.position * velocityData.radius) + positionData.position;
    float3 relPosition = worldPosition - globalUniforms.cameraPosition;
    out.position = globalUniforms.projectionMatrix * globalUniforms.viewMatrix * float4(relPosition, 1.0);
    out.color = colorData.color;
    out.instanceID = instanceID;
    return out;
}

vertex DustVertexOut dust_point_vertex(uint vertexID [[vertex_id]],
                                  uint instanceID [[instance_id]],
                                  constant GlobalUniforms &globalUniforms [[buffer(1)]],
                                  device const PositionMass *positions [[buffer(2)]],
                                  device const VelocityRadius *velocities [[buffer(3)]],
                                  device const ColorType *colors [[buffer(4)]])
{
    DustVertexOut out;
    PositionMass positionData = positions[instanceID];
    VelocityRadius velocityData = velocities[instanceID];
    ColorType colorData = colors[instanceID];

    float3 particleCenterWorld = positionData.position;
    float3 relPosition = particleCenterWorld - globalUniforms.cameraPosition;
    out.position = globalUniforms.projectionMatrix * globalUniforms.viewMatrix * float4(relPosition, 1.0);
    out.pointSize = velocityData.radius;
    

    out.color = colorData.color;

    return out;
}

fragment float4 fragment_shader(VertexOut in [[stage_in]])
{
   return in.color;
}
