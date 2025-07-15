#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

kernel void countUniqueMortonCodes(
    device const uint64_t* sortedMortonCodes [[buffer(0)]],
    device atomic_uint* uniqueMortonCodeCount [[buffer(1)]],
    device uint* uniqueMortonCodeStartIndices [[buffer(2)]],
    constant uint& numSpheres [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        // Initialize count to 0 and set the first start index
        atomic_store_explicit(uniqueMortonCodeCount, 0u, memory_order_relaxed);
        if (numSpheres > 0) {
            atomic_store_explicit(uniqueMortonCodeCount, 1u, memory_order_relaxed);
            uniqueMortonCodeStartIndices[0] = 0;
        }
    } else if (gid < numSpheres) {
        // Compare with previous element
        if (sortedMortonCodes[gid] != sortedMortonCodes[gid - 1]) {
            // Atomically increment the count if the current code is unique
            uint index = atomic_fetch_add_explicit(uniqueMortonCodeCount, 1u, memory_order_relaxed);
            uniqueMortonCodeStartIndices[index] = gid;
        }
    }
}

kernel void aggregateLeafNodes(
    device const uint64_t* sortedMortonCodes [[buffer(0)]],
    device const uint32_t* sortedIndices [[buffer(1)]],
    device const PositionMass* positionMassBuffer [[buffer(2)]],
    device const ColorType* colorTypeBuffer [[buffer(3)]],
    device OctreeLeafNode* octreeLeafNodes [[buffer(4)]],
    device const uint* uniqueMortonCodeCount [[buffer(5)]],
    device const uint* uniqueMortonCodeStartIndices [[buffer(6)]],
    constant uint& numSpheres [[buffer(7)]],
    uint nodeIndex [[thread_position_in_grid]])
{
    uint uniqueCount = uniqueMortonCodeCount[0];
    if (nodeIndex >= uniqueCount || uniqueCount <= 1) { // Also check if there's more than one unique code
        return;
    }

    uint startIndex = uniqueMortonCodeStartIndices[nodeIndex];
    uint endIndex;

    if (nodeIndex + 1 < uniqueMortonCodeCount[0]) {
        endIndex = uniqueMortonCodeStartIndices[nodeIndex + 1];
    } else {
        endIndex = numSpheres; // Last unique code goes to the end of the sorted list
    }

    uint64_t currentMortonCode = sortedMortonCodes[startIndex];

    // Initialize aggregation variables
    float3 centerOfMassSum = float3(0);
    float totalMass = 0;
    float4 emittedColorSum = float4(0.0);
    float3 emittedColorCenter = float3(0);
    uint emittedColorsCount = 0;

    // Aggregate data for all elements sharing this morton code
    for (uint i = startIndex; i < endIndex; ++i) {
        uint originalIndex = sortedIndices[i];
        PositionMass pm = positionMassBuffer[originalIndex];
        ColorType ct = colorTypeBuffer[originalIndex];

        centerOfMassSum += pm.position * pm.mass;
        totalMass += pm.mass;

        // Only stars emit color (type 0)
        if (ct.type == 0) {
            emittedColorSum += ct.color;
            emittedColorCenter += pm.position;
            emittedColorsCount += 1;
        }
    }

    // Calculate averages and store in the leaf node
    device OctreeLeafNode& leafNode = octreeLeafNodes[nodeIndex];
    leafNode.mortonCode = currentMortonCode;

    if (totalMass > 0.0f) {
        leafNode.centerOfMass = centerOfMassSum / totalMass;
    }
    leafNode.totalMass = totalMass;

    if (emittedColorsCount > 0) {
        leafNode.emittedColor = emittedColorSum;
        leafNode.emittedColorCenter = emittedColorCenter / emittedColorsCount;
    } else {
        leafNode.emittedColor = float4(0.0);
        leafNode.emittedColorCenter = float3(0.0);
    }
}
