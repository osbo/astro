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
                               device const LightingInfluences *lightingInfluences [[buffer(5)]],
                               constant uint &numStars [[buffer(6)]],
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
        // Planets: aggregate lighting influences
        LightingInfluences influences = lightingInfluences[instanceID - numStars];
        float3 baseColor = colorData.color.rgb;
        float3 litColor = float3(0.0f);
        for (uint i = 0; i < 8; ++i) {
            float3 lightColor = influences.colors[i].rgb;
            float3 lightDir = influences.positions[i].xyz;
            float normDist = length(lightDir) / LIGHT_ATTENUATION_DISTANCE;
            float att = 1.0f / (normDist * normDist + 1.0f);
            float3 lightDirNorm = normalize(lightDir);
            float NdotL = max(dot(normalize(in.normal), lightDirNorm), 0.0f);
            litColor += lightColor * NdotL * att;
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
