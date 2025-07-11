#ifndef bridge_h
#define bridge_h

#import <simd/simd.h>

typedef struct {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float3 cameraPosition;
} GlobalUniforms;

// Structure to hold Morton code and particle index
typedef struct {
    uint64_t mortonCode;  // 64-bit Morton code
    uint32_t particleIndex;  // Index of the particle
} MortonCodeEntry;

#endif /* bridge_h */
