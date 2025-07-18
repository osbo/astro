#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

kernel void countUniqueMortonCodes(
    device const uint64_t* sortedMortonCodes [[buffer(0)]],
    device atomic_uint* uniqueMortonCodeCount [[buffer(1)]],
    device uint* uniqueMortonCodeStartIndices [[buffer(2)]],
    constant uint& numSpheres [[buffer(3)]],
    constant uint& aggregate [[buffer(4)]],
    device uint* layerCountBuffer [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        // Initialize count to 0 and set the first start index
        atomic_store_explicit(uniqueMortonCodeCount, 0u, memory_order_relaxed);
        if (numSpheres > 0) {
            atomic_store_explicit(uniqueMortonCodeCount, 1u, memory_order_relaxed);
            uniqueMortonCodeStartIndices[0] = 0;
        }
        if (layerCountBuffer != nullptr) {
            layerCountBuffer[0] += 1;
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

// Kernel: fills buffer with startIdx..(endIdx-1)
kernel void fillIndices(
    device uint* buffer [[buffer(0)]],
    constant uint& startIdx [[buffer(1)]],
    constant uint& endIdx [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = startIdx + gid;
    if (idx < endIdx) {
        buffer[gid] = idx;
    } else {
        buffer[gid] = 0;
    }
}

kernel void aggregateLeafNodes(
    device uint64_t* sortedMortonCodesBuffer [[buffer(0)]],
    device uint* sortedIndicesBuffer [[buffer(1)]],
    device const uint* uniqueIndicesBuffer [[buffer(2)]],
    device const PositionMass* positionMassBuffer [[buffer(3)]],
    device const ColorType* colorTypeBuffer [[buffer(4)]],
    device const uint* mortonCodeCountBuffer [[buffer(5)]],
    device OctreeLeafNode* octreeNodesBuffer [[buffer(6)]],
    device uint64_t* unsortedMortonCodesBuffer [[buffer(7)]],
    device uint* unsortedIndicesBuffer [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint numNodes = mortonCodeCountBuffer[0];
    if (gid >= numNodes) return;

    uint start = uniqueIndicesBuffer[gid];
    uint end = (gid + 1 < numNodes) ? uniqueIndicesBuffer[gid + 1] : numNodes == 0 ? 0 : (uint) mortonCodeCountBuffer[0];
    // Defensive: if numNodes==0, end=0, else end=numSpheres
    
    float totalMass = 0.0;
    float3 centerOfMassSum = float3(0.0);
    float4 emittedColorSum = float4(0.0);
    float3 emittedColorCenterSum = float3(0.0);
    uint starCount = 0;

    uint64_t mortonCode = sortedMortonCodesBuffer[start];

    for (uint i = start; i < end; ++i) {
        uint sphereIdx = sortedIndicesBuffer[i];
        PositionMass pm = positionMassBuffer[sphereIdx];
        ColorType ct = colorTypeBuffer[sphereIdx];
        totalMass += pm.mass;
        centerOfMassSum += pm.position * pm.mass;
        if (ct.type == 0) {
            emittedColorSum += ct.color;
            emittedColorCenterSum += pm.position;
            starCount += 1;
        }
        sortedMortonCodesBuffer[i] = 0;
        sortedIndicesBuffer[i] = 0;
    }

    OctreeLeafNode node;
    node.mortonCode = mortonCode;
    node.totalMass = totalMass;
    node.centerOfMass = (totalMass > 0.0) ? (centerOfMassSum / totalMass) : float3(0.0);
    node.emittedColor = (starCount > 0) ? emittedColorSum : float4(0.0);
    node.emittedColorCenter = (starCount > 0) ? (emittedColorCenterSum / float(starCount)) : float3(0.0);

    octreeNodesBuffer[gid] = node;
    unsortedMortonCodesBuffer[gid] = mortonCode;
    unsortedIndicesBuffer[gid] = gid;
}

// Updated aggregateNodes kernel: single node buffer, explicit input/output offsets
kernel void aggregateNodes(
    device OctreeLeafNode* octreeNodesBuffer [[buffer(0)]],
    device const uint64_t* sortedMortonCodesBuffer [[buffer(1)]],
    device const uint* sortedIndicesBuffer [[buffer(2)]],
    device const uint* uniqueIndicesBuffer [[buffer(3)]],
    device const uint* mortonCodeCountBuffer [[buffer(4)]],
    constant uint& inputOffset [[buffer(5)]],
    constant uint& outputOffset [[buffer(6)]],
    device uint64_t* unsortedMortonCodesBuffer [[buffer(7)]],
    device uint* unsortedIndicesBuffer [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint numNodes = mortonCodeCountBuffer[0];
    if (gid >= numNodes) return;

    // Each internal node aggregates up to 8 children
    uint childStart = inputOffset + gid * 8;
    uint childEnd = min(childStart + 8, inputOffset + numNodes);

    float totalMass = 0.0;
    float3 centerOfMassSum = float3(0.0);
    float4 emittedColorSum = float4(0.0);
    float3 emittedColorCenterSum = float3(0.0);
    uint starCount = 0;
    uint64_t mortonCode = 0xFFFFFFFFFFFFFFFF;

    for (uint i = childStart; i < childEnd; ++i) {
        OctreeLeafNode child = octreeNodesBuffer[i];
        totalMass += child.totalMass;
        centerOfMassSum += child.centerOfMass * child.totalMass;
        if (any(child.emittedColor != float4(0.0))) {
            emittedColorSum += child.emittedColor;
            emittedColorCenterSum += child.emittedColorCenter;
            starCount += 1;
        }
        if (child.mortonCode < mortonCode) {
            mortonCode = child.mortonCode;
        }
    }

    OctreeLeafNode node;
    node.mortonCode = mortonCode;
    node.totalMass = totalMass;
    node.centerOfMass = (totalMass > 0.0) ? (centerOfMassSum / totalMass) : float3(0.0);
    node.emittedColor = (starCount > 0) ? emittedColorSum : float4(0.0);
    node.emittedColorCenter = (starCount > 0) ? (emittedColorCenterSum / float(starCount)) : float3(0.0);

    octreeNodesBuffer[outputOffset + gid] = node;
    unsortedMortonCodesBuffer[gid] = node.mortonCode;
    unsortedIndicesBuffer[gid] = gid;
}

kernel void extractMortonCodes(
    device const OctreeLeafNode* octreeNodes [[buffer(0)]],
    device ulong* mortonCodes [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    mortonCodes[gid] = octreeNodes[gid].mortonCode;
}

kernel void unsortedLeaf(
    device uint* mortonCodeCountBuffer [[buffer(0)]],
    device uint* layerCountBuffer [[buffer(1)]],
    constant uint& numSpheres [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        mortonCodeCountBuffer[0] = numSpheres;
        layerCountBuffer[0] = 0;
    }
}

kernel void unsortedInternal(
    device uint* mortonCodeCountBuffer [[buffer(0)]],
    device uint* layerCountBuffer [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        mortonCodeCountBuffer[0] = max(1u, mortonCodeCountBuffer[0] / 8u);
        layerCountBuffer[0] += 1u;
    }
}

kernel void resetUIntBuffer(device uint* buffer [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0) buffer[0] = 0;
}
