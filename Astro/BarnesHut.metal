#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

#define MAX_STACK_SIZE 16384
#define INVALID_CHILD_INDEX 0xFFFFFFFFu

kernel void barnesHut(
    device OctreeNode* octreeNodesBuffer [[buffer(0)]],
    device PositionMass* positionMassBuffer [[buffer(1)]],
    device VelocityRadius* velocityRadiusBuffer [[buffer(2)]],
    device ColorType* colorTypeBuffer [[buffer(3)]],
    device float3* forceBuffer [[buffer(4)]],
    constant uint& rootNodeIndex [[buffer(5)]],
    constant uint& numSpheres [[buffer(6)]],
    constant float& theta [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numSpheres) return;
    // Example: assume root node is at index 0
    uint stack[MAX_STACK_SIZE];
    uint stackPtr = 0;
    stack[stackPtr++] = rootNodeIndex; // root node index
    
    // Define gravitational constant and softening
    const float g = 50000; // Or use a simulation-appropriate value
    const float softening = 0.01;
    float3 totalForce = float3(0.0f);

    float3 myPos = positionMassBuffer[gid].position;
    float myMass = positionMassBuffer[gid].mass;
    float myType = colorTypeBuffer[gid].type;
    float3 myColor = colorTypeBuffer[gid].color.rgb;

    while (stackPtr > 0) {
        uint nodeIdx = stack[--stackPtr];
        OctreeNode node = octreeNodesBuffer[nodeIdx];
        // Compute distance from this sphere to node center of mass
        float3 d = node.centerOfMass - myPos;
        float distSqr = dot(d, d) + softening;
        float dist = sqrt(distSqr);
        // Compute node side length based on layer
        const uint maxLayer = 8;
        float sideLength = pow(2.0f, 21.0f - (float)(maxLayer - node.layer));
        bool isLeaf = true;
        for (uint i = 0; i < 8; ++i) {
            if (node.children[i] != INVALID_CHILD_INDEX) { isLeaf = false; break; }
        }
        bool useApprox = isLeaf || (sideLength / dist < theta);
        bool isSelf = (dist < 1e-6f);
        if (useApprox && !isSelf) {
            float3 force = g * node.totalMass * d / (distSqr * dist);

            // if (myType == 0) {
            //     force *= 1;
            // } else if (myType == 1) {
            //     force *= 10;
            // } else if (myType == 2) {
            //     force *= 100;
            // }

            totalForce += force;
        } else if (!isLeaf) {
            for (uint i = 0; i < 8; ++i) {
                uint childIdx = node.children[i];
                if (childIdx != INVALID_CHILD_INDEX && stackPtr < MAX_STACK_SIZE) {
                    stack[stackPtr++] = childIdx;
                }
            }
        }
    }
    // Write totalForce to forceBuffer
    forceBuffer[gid] = totalForce;

    // Stars: do not aggregate lighting
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
    colorTypeBuffer[gid].color.w = max(totalLight.r, max(totalLight.g, totalLight.b));
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
    float mass = positionMassBuffer[gid].mass;
    float3 acceleration = force / mass;
    float3 velocity = velocityRadiusBuffer[gid].velocity * 0.9999;
    velocity += acceleration * dt;
    velocityRadiusBuffer[gid].velocity = velocity;
    float3 position = positionMassBuffer[gid].position;
    position += velocity * dt;
    positionMassBuffer[gid].position = position;
}
