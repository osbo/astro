#include <metal_stdlib>
#include "bridge.h"

using namespace metal;


uint64_t interleaveBitsOptimized(uint32_t x, uint32_t y, uint32_t z) {

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


kernel void generateMortonCodes(device const PositionMass *positions [[buffer(0)]],
                               device uint64_t *mortonCodes [[buffer(1)]],
                               device uint64_t *unsortedMortonCodes [[buffer(2)]],
                               uint particleIndex [[thread_position_in_grid]])
{

    PositionMass data = positions[particleIndex];
    

    uint32_t x = (uint32_t)(data.position.x + 1000000);
    uint32_t y = (uint32_t)(data.position.y + 1000000);
    uint32_t z = (uint32_t)(data.position.z + 1000000);
    

    uint64_t mortonCode = interleaveBitsOptimized(x, y, z);


    #define NUM_LAYERS 6
    #define BITS_PER_LAYER 3
    #define USED_BITS (NUM_LAYERS * BITS_PER_LAYER)
    #define UNUSED_BITS (63 - USED_BITS)
    mortonCode = mortonCode >> UNUSED_BITS;
    

    mortonCodes[particleIndex] = mortonCode;
    unsortedMortonCodes[particleIndex] = mortonCode;
} 