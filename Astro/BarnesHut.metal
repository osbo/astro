#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

#define MAX_STACK_SIZE 64

kernel void barnesHut(
    device OctreeNode* octreeNodesBuffer [[buffer(0)]],
    device PositionMass* positionMassBuffer [[buffer(1)]],
    device VelocityRadius* velocityRadiusBuffer [[buffer(2)]],
    device ColorType* colorTypeBuffer [[buffer(3)]],
    device LightingInfluences* lightingInfluencesBuffer [[buffer(4)]],
    constant float& dt [[buffer(5)]],
    constant uint& rootNodeIndex [[buffer(6)]],
    constant uint& numStars [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    // Example: assume root node is at index 0
    uint stack[MAX_STACK_SIZE];
    uint stackPtr = 0;
    stack[stackPtr++] = rootNodeIndex; // root node index
    
    // Define gravitational constant and softening
    const float g = 100000; // Or use a simulation-appropriate value
    const float softening = 1e-2f;
    float3 totalForce = float3(0.0f);

    while (stackPtr > 0) {
        uint nodeIdx = stack[--stackPtr];
        OctreeNode node = octreeNodesBuffer[nodeIdx];
        // Compute distance from this sphere to node center of mass
        float3 myPos = positionMassBuffer[gid].position;
        float myMass = positionMassBuffer[gid].mass;
        float3 d = node.centerOfMass - myPos;
        float distSqr = dot(d, d) + softening;
        float dist = sqrt(distSqr);
        // Compute node side length based on layer
        const uint maxLayer = 8;
        float sideLength = pow(2.0f, 21.0f - (float)(maxLayer - node.layer));
        // Check if node is a leaf
        bool isLeaf = true;
        for (uint i = 0; i < 8; ++i) {
            if (node.children[i] != 0) { isLeaf = false; break; }
        }
        // Theta criterion
        bool useApprox = isLeaf || (sideLength / dist < theta);
        bool isSelf = (dist < 1e-6f);
        if (useApprox && !isSelf) {
            // Accumulate gravitational force: F = G * m1 * m2 * r / |r|^3
            float3 force = g * myMass * node.totalMass * d / (distSqr * dist);
            totalForce += force;
        } else if (!isLeaf) {
            // Descend into children
            for (uint i = 0; i < 8; ++i) {
                uint childIdx = node.children[i];
                if (childIdx != 0 && stackPtr < MAX_STACK_SIZE) {
                    stack[stackPtr++] = childIdx;
                }
            }
        }
    }
    // After accumulating totalForce for this sphere:
    // Update velocity and position using F = m*a, a = F/m, v += a*dt, x += v*dt
    float3 velocity = velocityRadiusBuffer[gid].velocity;
    float mass = positionMassBuffer[gid].mass;
    float3 acceleration = totalForce / mass;
    velocity += acceleration * dt;
    float3 position = positionMassBuffer[gid].position;
    position += velocity * dt;
    // Write back updated values
    velocityRadiusBuffer[gid].velocity = velocity;
    positionMassBuffer[gid].position = position;
    // TODO: Write results to lightingInfluencesBuffer[gid]
}