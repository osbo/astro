#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float3 normal;
    float3 worldPosition;
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
                               device const float3 *forces [[buffer(5)]],
                               uint instanceID [[instance_id]])
{
    VertexOut out;
    PositionMass positionData = positions[instanceID];
    VelocityRadius velocityData = velocities[instanceID];
    ColorType colorData = colors[instanceID];

    float3 worldPosition = (in.position * velocityData.radius) + positionData.position;
    float3 relPosition = worldPosition - globalUniforms.cameraPosition;
    out.position = globalUniforms.projectionMatrix * globalUniforms.viewMatrix * float4(relPosition, 1.0);
    out.normal = in.normal;
    out.worldPosition = worldPosition;
    out.color = colorData.color;
    out.instanceID = instanceID;
    return out;
}

vertex DustVertexOut dust_point_vertex(uint vertexID [[vertex_id]],
                                  uint instanceID [[instance_id]],
                                  constant GlobalUniforms &globalUniforms [[buffer(1)]],
                                  device const PositionMass *positions [[buffer(2)]],
                                  device const VelocityRadius *velocities [[buffer(3)]],
                                  device const ColorType *colors [[buffer(4)]],
                                  device const float3 *forces [[buffer(5)]])
{
    DustVertexOut out;
    PositionMass positionData = positions[instanceID];
    VelocityRadius velocityData = velocities[instanceID];
    ColorType colorData = colors[instanceID];

    float3 particleCenterWorld = positionData.position;
    float3 relPosition = particleCenterWorld - globalUniforms.cameraPosition;
    out.position = globalUniforms.projectionMatrix * globalUniforms.viewMatrix * float4(relPosition, 1.0);
    out.pointSize = velocityData.radius;
    
    // Set alpha based on brightness
    out.color = colorData.color;

    return out;
}

fragment float4 fragment_shader(VertexOut in [[stage_in]])
{
   return in.color;
    // return (1,1,1,1);
}

fragment float4 planet_fragment_shader(VertexOut in [[stage_in]],
                                      constant GlobalUniforms &globalUniforms [[buffer(1)]],
                                      device const PositionMass *positions [[buffer(2)]],
                                      device const ColorType *colors [[buffer(3)]],
                                      constant uint &numStars [[buffer(4)]])
{
    float3 normal = length(in.normal) > 1e-9f ? normalize(in.normal) : float3(0,0,1);
    float3 color = float3(0.0);
    for (uint i = 0; i < numStars; ++i) {
        float3 lightPos = positions[i].position;
        float3 lightColor = colors[i].color.rgb;
        float3 lightDirVec = lightPos - in.worldPosition;
        float3 lightDir = length(lightDirVec) > 1e-9f ? normalize(lightDirVec) : float3(0,0,1);
        float dist = length(lightPos - in.worldPosition);
        float att = 1.0f / pow(max(dist / LIGHT_ATTENUATION_DISTANCE, 1e-6f), 2.0f);
        float diff = max(dot(normal, lightDir), 0.0);
        color += lightColor * diff * att;
    }
    color = clamp(color, 0.0, 1.0);
    return float4(color, 1.0);
}
