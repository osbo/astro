#ifndef bridge_h
#define bridge_h

#import <simd/simd.h>

typedef struct {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float3 cameraPosition;
} GlobalUniforms;

#endif /* bridge_h */
