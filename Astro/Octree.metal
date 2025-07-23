#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

#define NUM_LAYERS 8
#define BITS_PER_LAYER 3
#define USED_BITS (NUM_LAYERS * BITS_PER_LAYER)
#define UNUSED_BITS (63 - USED_BITS)
#define INVALID_CHILD 0xFFFFFFFFu // Sentinel for invalid child index

kernel void countUniqueMortonCodes(
    device const uint64_t* sortedMortonCodes [[buffer(0)]],
    device atomic_uint* uniqueMortonCodeCount [[buffer(1)]],
    device uint* uniqueMortonCodeStartIndices [[buffer(2)]],
    constant uint& numSpheres [[buffer(3)]],
    constant uint& aggregate [[buffer(4)]],
    device uint* layerCountBuffer [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numSpheres) {
        return;
    }
    bool isUnique;
    if (gid == 0) {
        isUnique = true;
    } else if (aggregate == 1) {
        isUnique = (sortedMortonCodes[gid] >> 3) != (sortedMortonCodes[gid - 1] >> 3);
    } else {
        isUnique = (sortedMortonCodes[gid] != sortedMortonCodes[gid - 1]);
    }
    if (isUnique) {
        uint index = atomic_fetch_add_explicit(uniqueMortonCodeCount, 1u, memory_order_relaxed);
        uniqueMortonCodeStartIndices[index] = gid;
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
    device OctreeNode* octreeNodesBuffer [[buffer(6)]],
    device uint64_t* unsortedMortonCodesBuffer [[buffer(7)]],
    device uint* unsortedIndicesBuffer [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint numUnique = mortonCodeCountBuffer[0];
    if (gid >= numUnique) {
        return;
    }

    uint childStartIdx = uniqueIndicesBuffer[gid];
    uint childEndIdx = (gid + 1 < numUnique) ? uniqueIndicesBuffer[gid + 1] : numUnique;

    float3 totalCenterOfMass = float3(0.0, 0.0, 0.0);
    float totalMass = 0.0;
    float4 totalEmittedColor = float4(0.0, 0.0, 0.0, 0.0);
    float3 totalEmittedColorCenter = float3(0.0, 0.0, 0.0);
    uint32_t children[8] = {INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD};
    uint numSuns = 0;
    
    for (uint i = childStartIdx; i < childEndIdx; i++) {
        uint sphereIndex = sortedIndicesBuffer[i];
        PositionMass sphereData = positionMassBuffer[sphereIndex];
        ColorType colorData = colorTypeBuffer[sphereIndex];
        
        totalMass += sphereData.mass;
        totalCenterOfMass += sphereData.position * sphereData.mass;
        
        if (colorData.type == 0) {
            totalEmittedColor += colorData.color;
            totalEmittedColorCenter += sphereData.position;
            numSuns += 1;
        }
    }

    float3 finalCenterOfMass = (totalMass > 0.0) ? (totalCenterOfMass / totalMass) : float3(0.0, 0.0, 0.0);
    float4 finalEmittedColor = totalEmittedColor;
    float3 finalEmittedColorCenter = totalEmittedColorCenter;
    
    if (numSuns > 0) {
        finalEmittedColorCenter /= numSuns;
    }

    uint64_t firstMortonCode = sortedMortonCodesBuffer[childStartIdx];
    OctreeNode leafNode;
    leafNode.mortonCode = firstMortonCode;
    leafNode.centerOfMass = finalCenterOfMass;
    leafNode.totalMass = totalMass;
    leafNode.emittedColor = finalEmittedColor;
    leafNode.emittedColorCenter = finalEmittedColorCenter;
    leafNode.layer = 0;
    for (int i = 0; i < 8; i++) {
        leafNode.children[i] = INVALID_CHILD;
    }

    octreeNodesBuffer[gid] = leafNode;
    unsortedMortonCodesBuffer[gid] = firstMortonCode;
    unsortedIndicesBuffer[gid] = gid;

    return;
}

// Updated aggregateNodes kernel: single node buffer, explicit input/output offsets
kernel void aggregateNodes(
    device OctreeNode* octreeNodesBuffer [[buffer(0)]],
    device const uint64_t* sortedMortonCodesBuffer [[buffer(1)]],
    device const uint* sortedIndicesBuffer [[buffer(2)]],
    device const uint* uniqueIndicesBuffer [[buffer(3)]],
    device const uint* parentCountBuffer [[buffer(4)]],
    constant uint& inputOffset [[buffer(5)]],
    constant uint& outputOffset [[buffer(6)]],
    constant uint& layer [[buffer(7)]],
    constant uint& inputLayerSize [[buffer(8)]],
    constant uint& outputLayerSize [[buffer(9)]],
    device uint64_t* unsortedMortonCodesBuffer [[buffer(10)]],
    device uint* unsortedIndicesBuffer [[buffer(11)]],
    device const uint* childCountBuffer [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
     uint numUnique = parentCountBuffer[0];
     if (gid >= numUnique) {
         // This parent node is not valid and should not be processed.
         return;
     }

     uint childStartIdx = uniqueIndicesBuffer[gid];
     uint childEndIdx = (gid + 1 < numUnique) ? uniqueIndicesBuffer[gid + 1] : childCountBuffer[0];

     float3 totalCenterOfMass = float3(0.0, 0.0, 0.0);
     float totalMass = 0.0;
     float4 totalEmittedColor = float4(0.0, 0.0, 0.0, 0.0);
     float3 totalEmittedColorCenter = float3(0.0, 0.0, 0.0);
     uint32_t children[8] = {INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD, INVALID_CHILD};
     int childCount = 0;
     uint numSuns = 0;
    
     for (uint i = childStartIdx; i < childEndIdx; i++) {
         uint nodeIndex = sortedIndicesBuffer[i] + inputOffset;
         OctreeNode childNode = octreeNodesBuffer[nodeIndex];

         totalMass += childNode.totalMass;
         totalCenterOfMass += childNode.centerOfMass * childNode.totalMass;
        
         if (childNode.emittedColor.w > 0.0) {
             totalEmittedColor += childNode.emittedColor;
             totalEmittedColorCenter += childNode.emittedColorCenter;
             numSuns += 1;
         }

         if (childCount < 8) {
             children[childCount] = nodeIndex;
             childCount += 1;
         }
     }

     float3 finalCenterOfMass = (totalMass > 0.0) ? (totalCenterOfMass / totalMass) : float3(0.0, 0.0, 0.0);
     float4 finalEmittedColor = totalEmittedColor;
     float3 finalEmittedColorCenter = totalEmittedColorCenter;
    
     if (numSuns > 0) {
         finalEmittedColorCenter /= numSuns;
     }

     uint64_t firstMortonCode = sortedMortonCodesBuffer[childStartIdx];
     uint64_t shiftedMortonCode = firstMortonCode >> BITS_PER_LAYER;
     OctreeNode node;
     node.mortonCode = shiftedMortonCode;
     node.centerOfMass = finalCenterOfMass;
     node.totalMass = totalMass;
     node.emittedColor = finalEmittedColor;
     node.emittedColorCenter = finalEmittedColorCenter;
     node.layer = layer;
     for (int i = 0; i < 8; i++) {
         node.children[i] = (i < childCount) ? children[i] : INVALID_CHILD;
     }

     // Robust bounds check for output index
     uint outputIndex = gid + outputOffset;
     // if (outputIndex < outputOffset || outputIndex >= outputOffset + outputLayerSize) {
     //     return;
     // }

     octreeNodesBuffer[outputIndex] = node;
     unsortedMortonCodesBuffer[gid] = shiftedMortonCode;
     unsortedIndicesBuffer[gid] = gid;

     return;

//    uint numUnique = mortonCodeCountBuffer[0];
//    if (gid >= numUnique) {
//        return;
//    }
//    uint outputIndex = gid + outputOffset;
//
//    OctreeNode node;
//    node.mortonCode = 1;
//    node.centerOfMass = float3(0.0, 0.0, 0.0);
//    node.totalMass = 0.0;
//    node.emittedColor = float4(0.0, 0.0, 0.0, 0.0);
//    node.emittedColorCenter = float3(0.0, 0.0, 0.0);
//    for (int i = 0; i < 8; i++) {
//        node.children[i] = 0;
//    }
//    node.layer = layer;
//
//    octreeNodesBuffer[outputIndex] = node;
//    unsortedMortonCodesBuffer[gid] = 1;
//    unsortedIndicesBuffer[gid] = gid;
//
//    return;
}

kernel void extractMortonCodes(
    device const OctreeNode* octreeNodes [[buffer(0)]],
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

kernel void clearBuffer(
    device uint* buffer [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    buffer[gid] = 0;
}

kernel void clearBuffer64(
    device uint64_t* buffer [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    buffer[gid] = 0;
}

kernel void clearOctreeNodeBuffer(
    device OctreeNode* buffer [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    OctreeNode emptyNode;
    emptyNode.mortonCode = 0;
    emptyNode.centerOfMass = float3(0.0, 0.0, 0.0);
    emptyNode.totalMass = 0.0;
    emptyNode.emittedColor = float4(0.0, 0.0, 0.0, 0.0);
    emptyNode.emittedColorCenter = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 8; i++) {
        emptyNode.children[i] = INVALID_CHILD;
    }
    emptyNode.layer = 0;
    
    buffer[gid] = emptyNode;
}

kernel void copyAndResetParentCount(device uint* parentCountBuffer [[buffer(0)]],
                                    device uint* childCountBuffer [[buffer(1)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        childCountBuffer[0] = parentCountBuffer[0];
        parentCountBuffer[0] = 0;
    }
}
