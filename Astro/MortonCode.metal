#include <metal_stdlib>
#include "bridge.h"

using namespace metal;

// Morton code generation for 32-bit coordinates to 64-bit Morton codes
uint64_t interleaveBitsOptimized(uint32_t x, uint32_t y, uint32_t z) {
    // Use first 21 bits of each coordinate (21 * 3 = 63 bits, 1 bit left)
    x &= 0x1FFFFF;  // 21 bits
    y &= 0x1FFFFF;  // 21 bits  
    z &= 0x1FFFFF;  // 21 bits
    
    uint64_t result = 0;
    
    // Interleave bits: x, y, z -> result
    for (int i = 0; i < 21; i++) {
        result |= ((uint64_t)(x & (1 << i)) << (2 * i));
        result |= ((uint64_t)(y & (1 << i)) << (2 * i + 1));
        result |= ((uint64_t)(z & (1 << i)) << (2 * i + 2));
    }
    
    return result;
}

// Compute kernel to generate Morton codes from particle positions
kernel void generateMortonCodes(device const PositionMass *positions [[buffer(0)]],
                               device uint64_t *mortonCodes [[buffer(1)]],
                               device uint32_t *indices [[buffer(2)]],
                               uint particleIndex [[thread_position_in_grid]])
{
    // Get particle position
    PositionMass data = positions[particleIndex];
    
    // Convert to unsigned coordinates with offset to handle negative values
    uint32_t x = (uint32_t)(data.position.x + 1000000);
    uint32_t y = (uint32_t)(data.position.y + 1000000);
    uint32_t z = (uint32_t)(data.position.z + 1000000);
    
    // Generate Morton code using optimized bit interleaving
    uint64_t mortonCode = interleaveBitsOptimized(x, y, z);
    
    // Store morton code and index
    mortonCodes[particleIndex] = mortonCode;
    indices[particleIndex] = particleIndex;
} 