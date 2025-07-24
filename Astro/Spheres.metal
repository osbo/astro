#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float3 normal;
    float3 worldPosition;
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
                               constant uint &numStars [[buffer(5)]],
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
    if (colorData.type == 1) { // planet
        // Planets: aggregate lighting influences with Lambertian diffuse
        float3 litColor = float3(0.0f);
        // Compute world-space normal for a sphere
        float3 normal = normalize(worldPosition - positionData.position);
        for (uint i = 0; i < numStars; ++i) {
            float3 lightColor = colors[i].color.rgb;
            float3 lightPos = positions[i].position;
            float3 lightDir = normalize(lightPos - worldPosition);
            float lightDist = length(lightPos - worldPosition);
            float normDist = max(lightDist / LIGHT_ATTENUATION_DISTANCE, 1e-9f);
            float att = 1.0f / (normDist * normDist);
            float NdotL = max(dot(normal, lightDir), 0.0f);
            litColor += lightColor * att * NdotL;
        }
        out.color = float4(litColor, 1.0f);
    } else {
        // Stars and dust: use color as is
        out.color = colorData.color;
    }
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
    
    // Set alpha based on brightness
    float brightness = max(colorData.color.r, max(colorData.color.g, colorData.color.b));
    out.color = float4(colorData.color.rgb, brightness);

    return out;
}

fragment float4 fragment_shader(VertexOut in [[stage_in]])
{
   return in.color;
    // return (1,1,1,1);
}
