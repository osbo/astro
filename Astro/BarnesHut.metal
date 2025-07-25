#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

#define MAX_STACK_SIZE 60000
#define INVALID_CHILD_INDEX 0xFFFFFFFFu

uint64_t interleaveBitsBarnesHut(uint32_t x, uint32_t y, uint32_t z) {
    x &= 0x1FFFFF;  // 21 bits
    y &= 0x1FFFFF;  // 21 bits
    z &= 0x1FFFFF;  // 21 bits
    uint64_t result = 0;
    for (int i = 0; i < 21; i++) {
        result |= ((uint64_t)(x & (1 << i)) << (2 * i));
        result |= ((uint64_t)(y & (1 << i)) << (2 * i + 1));
        result |= ((uint64_t)(z & (1 << i)) << (2 * i + 2));
    }
    return result;
}

kernel void barnesHut(
    device OctreeNode* octreeNodesBuffer [[buffer(0)]],
    device PositionMass* positionMassBuffer [[buffer(1)]],
    device VelocityRadius* velocityRadiusBuffer [[buffer(2)]],
    device ColorType* colorTypeBuffer [[buffer(3)]],
    device float3* forceBuffer [[buffer(4)]],
    device uint64_t* mortonCodesBuffer [[buffer(5)]],
    constant uint& rootNodeIndex [[buffer(6)]],
    constant uint& numSpheres [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numSpheres) return;
    uint stack[MAX_STACK_SIZE];
    uint stackPtr = 0;
    stack[stackPtr++] = rootNodeIndex;

    const float g = 1e11;
    const float softening = 0.01;
    float3 totalForce = float3(0.0f);

    float3 myPos = positionMassBuffer[gid].position;
    float myType = colorTypeBuffer[gid].type;
    uint64_t myMortonCode = 0xFFFFFFFFFFFFFFFFu;
    if (myType != 2) {
        myMortonCode = mortonCodesBuffer[gid];
    }

    while (stackPtr > 0) {
        uint nodeIdx = stack[--stackPtr];
        OctreeNode node = octreeNodesBuffer[nodeIdx];
        if (node.mortonCode == 0xFFFFFFFFFFFFFFFFu) {
            continue;
        }
        float3 d = (node.centerOfMass - myPos);
        float distSqr = dot(d, d) + softening;
        distSqr = max(distSqr, 1e-6f);
        float dist = sqrt(distSqr);
        dist = max(dist, 1e-6f);
        float sideLength = pow(2.0f, 21.0f + (float)(node.layer));
        bool isLeaf = node.layer == 0;
        uint shift = 3 * (node.layer);
        bool isSelf = false;
        if (myType != 2) {
            isSelf = ((myMortonCode >> shift) == node.mortonCode) && isLeaf;
        }
        if (isLeaf) {
            if (!isSelf && distSqr > 50) {
                float3 force = g * node.totalMass * d / (distSqr * dist);
                totalForce += force;
            }
        } else {
            if (sideLength / dist < theta) {
                float3 force = g * node.totalMass * d / (distSqr * dist);
                totalForce += force;
            } else {
                for (uint i = 0; i < 8; ++i) {
                    uint childIdx = node.children[i];
                    if (childIdx != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) {
                        stack[stackPtr++] = childIdx;
                    }
                }
            }
        }
    }
    forceBuffer[gid] = clamp(totalForce, -1e5, 1e5);
}

kernel void lightingPass(
    device PositionMass* positionMassBuffer [[buffer(0)]],
    device ColorType* colorTypeBuffer [[buffer(1)]],
    constant uint& numSpheres [[buffer(2)]],
    constant uint& numStars [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numSpheres) return;
    uint type = colorTypeBuffer[gid].type;
    if (type != 2) return;
    float3 position = positionMassBuffer[gid].position;
    float3 totalLight = float3(0.0f);
    for (uint i = 0; i < numStars; ++i) {
        float3 lightSourcePos = positionMassBuffer[i].position;
        float3 lightSourceColor = colorTypeBuffer[i].color.rgb;
        float3 lightSourceDir = lightSourcePos - position;
        float lightSourceDist = length(lightSourceDir);
        float lightSourceDistNorm = max(lightSourceDist / LIGHT_ATTENUATION_DISTANCE, 1e-9f);
        float att = 1.0f / (lightSourceDistNorm * lightSourceDistNorm);
        totalLight += lightSourceColor * att;
    }
    colorTypeBuffer[gid].color.rgb = totalLight;
    colorTypeBuffer[gid].color.w = min(max(totalLight.r, max(totalLight.g, totalLight.b)), 0.5);
}

kernel void updateSpheres(
    device PositionMass* positionMassBuffer [[buffer(0)]],
    device VelocityRadius* velocityRadiusBuffer [[buffer(1)]],
    device float3* forceBuffer [[buffer(2)]],
    device ColorType* colorTypeBuffer [[buffer(3)]],
    constant float& dt [[buffer(4)]],
    constant uint& numSpheres [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numSpheres) return;
    float3 force = forceBuffer[gid];
    float3 acceleration = force;

    if (colorTypeBuffer[gid].type == 0) {
        acceleration *= 5e0;
    } else if (colorTypeBuffer[gid].type == 1) {
        acceleration *= 5e0;
    } else if (colorTypeBuffer[gid].type == 2) {
        acceleration *= 1e0;
    }

    float3 velocity = velocityRadiusBuffer[gid].velocity * 0.9995;
    velocity += acceleration * dt;
    velocityRadiusBuffer[gid].velocity = velocity;
    float3 position = positionMassBuffer[gid].position;
    position += velocity * dt;
    positionMassBuffer[gid].position = position;
}
