#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

kernel void countUniqueMortonCodes(
    device const uint64_t* sortedMortonCodes [[buffer(0)]],
    device atomic_uint* uniqueMortonCodeCount [[buffer(1)]],
    device uint* uniqueMortonCodeStartIndices [[buffer(2)]],
    constant uint& numSpheres [[buffer(3)]],
    constant uint& aggregate [[buffer(4)]],
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
        bool isUnique;
        if (aggregate == 1) {
            isUnique = ((sortedMortonCodes[gid] >> 3) != (sortedMortonCodes[gid - 1] >> 3));
        } else {
            isUnique = (sortedMortonCodes[gid] != sortedMortonCodes[gid - 1]);
        }
        if (isUnique) {
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
    device ulong* nodeMortonCodes [[buffer(5)]],
    device const uint* uniqueMortonCodeCount [[buffer(6)]],
    device const uint* uniqueMortonCodeStartIndices [[buffer(7)]],
    constant uint& numSpheres [[buffer(8)]],
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
    nodeMortonCodes[nodeIndex] = currentMortonCode;

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

// Updated aggregateNodes kernel: single node buffer, explicit input/output offsets
kernel void aggregateNodes(
    device const uint64_t* sortedMortonCodes [[buffer(0)]],
    device const uint32_t* sortedIndices [[buffer(1)]],
    device OctreeLeafNode* nodeBuffer [[buffer(2)]], // all nodes in one buffer
    device const ulong* parentMortonCodesIn [[buffer(3)]],
    device const uint* uniqueMortonCodeCount [[buffer(4)]],
    device const uint* uniqueMortonCodeStartIndices [[buffer(5)]],
    constant uint& numChildren [[buffer(6)]],
    constant uint& inputNodeStart [[buffer(7)]],
    constant uint& outputNodeStart [[buffer(8)]],
    constant uint& layerShift [[buffer(9)]],
    uint nodeIndex [[thread_position_in_grid]])
{
    uint uniqueCount = uniqueMortonCodeCount[0];
    if (nodeIndex >= uniqueCount || uniqueCount <= 1) {
        return;
    }

    uint startIndex = uniqueMortonCodeStartIndices[nodeIndex];
    uint endIndex = (nodeIndex + 1 < uniqueCount) ? uniqueMortonCodeStartIndices[nodeIndex + 1] : numChildren;
    uint64_t parentMortonCode = sortedMortonCodes[startIndex] >> layerShift;

    // Aggregate data from child nodes
    float3 centerOfMassSum = float3(0);
    float totalMass = 0;
    float4 emittedColorSum = float4(0.0);
    float3 emittedColorCenter = float3(0);
    uint emittedColorsCount = 0;

    for (uint i = startIndex; i < endIndex; ++i) {
        uint childIdx = sortedIndices[i];
        OctreeLeafNode child = nodeBuffer[inputNodeStart + childIdx];

        centerOfMassSum += child.centerOfMass * child.totalMass;
        totalMass += child.totalMass;
        emittedColorSum += child.emittedColor;
        emittedColorCenter += child.emittedColorCenter;
        if (any(child.emittedColor != float4(0.0))) {
            emittedColorsCount += 1;
        }
    }

    device OctreeLeafNode& parentNode = nodeBuffer[outputNodeStart + nodeIndex];
    parentNode.mortonCode = parentMortonCode;
    // parentMortonCodes[nodeIndex] = parentMortonCode; // If you want to store parent morton codes separately

    if (totalMass > 0.0f) {
        parentNode.centerOfMass = centerOfMassSum / totalMass;
    }
    parentNode.totalMass = totalMass;

    if (emittedColorsCount > 0) {
        parentNode.emittedColor = emittedColorSum;
        parentNode.emittedColorCenter = emittedColorCenter / emittedColorsCount;
    } else {
        parentNode.emittedColor = float4(0.0);
        parentNode.emittedColorCenter = float3(0.0);
    }
}

kernel void extractMortonCodes(
    device const OctreeLeafNode* octreeNodes [[buffer(0)]],
    device ulong* mortonCodes [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    mortonCodes[gid] = octreeNodes[gid].mortonCode;
}
