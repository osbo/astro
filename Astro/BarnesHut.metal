#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

#define MAX_STACK_SIZE 1024
#define INVALID_CHILD_INDEX 0xFFFFFFFFu

uint64_t interleaveBitsBarnesHut(uint32_t x, uint32_t y, uint32_t z) {
    x &= 0x1FFFFF;
    y &= 0x1FFFFF;
    z &= 0x1FFFFF;
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
        stackPtr--;
        uint nodeIdx = stack[stackPtr];
        OctreeNode node = octreeNodesBuffer[nodeIdx];
        if (node.mortonCode == 0xFFFFFFFFFFFFFFFFu) {
            continue;
        }
        float3 d = (node.centerOfMass - myPos);
        float distSqr = dot(d, d) + softening;
        distSqr = max(distSqr, 1e-6f);
        float dist = sqrt(distSqr);
        dist = max(dist, 1e-6f);
        float sideLength = float(1 << (21 + node.layer));
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

                uint child0 = node.children[0];
                uint child1 = node.children[1];
                uint child2 = node.children[2];
                uint child3 = node.children[3];
                uint child4 = node.children[4];
                uint child5 = node.children[5];
                uint child6 = node.children[6];
                uint child7 = node.children[7];
                
                if (child0 != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) stack[stackPtr++] = child0;
                if (child1 != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) stack[stackPtr++] = child1;
                if (child2 != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) stack[stackPtr++] = child2;
                if (child3 != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) stack[stackPtr++] = child3;
                if (child4 != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) stack[stackPtr++] = child4;
                if (child5 != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) stack[stackPtr++] = child5;
                if (child6 != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) stack[stackPtr++] = child6;
                if (child7 != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) stack[stackPtr++] = child7;
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
    if (type != 1 && type != 2) return;
    float3 position = positionMassBuffer[gid].position;
    float3 totalLight = float3(0.0f);

    if (numStars >= 1) {
        float3 lightSourcePos = positionMassBuffer[0].position;
        float3 lightSourceColor = colorTypeBuffer[0].color.rgb;
        float3 lightSourceDir = lightSourcePos - position;
        float lightSourceDist = length(lightSourceDir);
        float lightSourceDistNorm = max(lightSourceDist / LIGHT_ATTENUATION_DISTANCE, 1e-9f);
        float att = 1.0f / (lightSourceDistNorm * lightSourceDistNorm);
        totalLight += lightSourceColor * att;
    }
    if (numStars >= 2) {
        float3 lightSourcePos = positionMassBuffer[1].position;
        float3 lightSourceColor = colorTypeBuffer[1].color.rgb;
        float3 lightSourceDir = lightSourcePos - position;
        float lightSourceDist = length(lightSourceDir);
        float lightSourceDistNorm = max(lightSourceDist / LIGHT_ATTENUATION_DISTANCE, 1e-9f);
        float att = 1.0f / (lightSourceDistNorm * lightSourceDistNorm);
        totalLight += lightSourceColor * att;
    }
    if (numStars >= 3) {
        float3 lightSourcePos = positionMassBuffer[2].position;
        float3 lightSourceColor = colorTypeBuffer[2].color.rgb;
        float3 lightSourceDir = lightSourcePos - position;
        float lightSourceDist = length(lightSourceDir);
        float lightSourceDistNorm = max(lightSourceDist / LIGHT_ATTENUATION_DISTANCE, 1e-9f);
        float att = 1.0f / (lightSourceDistNorm * lightSourceDistNorm);
        totalLight += lightSourceColor * att;
    }
    if (numStars >= 4) {
        float3 lightSourcePos = positionMassBuffer[3].position;
        float3 lightSourceColor = colorTypeBuffer[3].color.rgb;
        float3 lightSourceDir = lightSourcePos - position;
        float lightSourceDist = length(lightSourceDir);
        float lightSourceDistNorm = max(lightSourceDist / LIGHT_ATTENUATION_DISTANCE, 1e-9f);
        float att = 1.0f / (lightSourceDistNorm * lightSourceDistNorm);
        totalLight += lightSourceColor * att;
    }
    if (numStars >= 5) {
        float3 lightSourcePos = positionMassBuffer[4].position;
        float3 lightSourceColor = colorTypeBuffer[4].color.rgb;
        float3 lightSourceDir = lightSourcePos - position;
        float lightSourceDist = length(lightSourceDir);
        float lightSourceDistNorm = max(lightSourceDist / LIGHT_ATTENUATION_DISTANCE, 1e-9f);
        float att = 1.0f / (lightSourceDistNorm * lightSourceDistNorm);
        totalLight += lightSourceColor * att;
    }
    colorTypeBuffer[gid].color.rgb = totalLight;
    if (type == 2) {
        colorTypeBuffer[gid].color.w = min(max(totalLight.r, max(totalLight.g, totalLight.b)), 0.25);
    } else {
        colorTypeBuffer[gid].color.w = 1.0;
    }
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
