#ifndef bridge_h
#define bridge_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type
#define NS_OPTIONS(_type, _name) _type _name; enum : _type
#include <metal_stdlib>
using namespace metal;
#else
#import <Foundation/Foundation.h>
#import <simd/simd.h>
#include <stdint.h>
#endif

// Structs shared between Metal and Swift/C

typedef struct {
#ifdef __METAL_VERSION__
    float3 position;
#else
    simd_float3 position;
#endif
    float mass;
} PositionMass;

typedef struct {
#ifdef __METAL_VERSION__
    float3 velocity;
#else
    simd_float3 velocity;
#endif
    float radius;
} VelocityRadius;

typedef struct {
#ifdef __METAL_VERSION__
    float4 color;
    uint type; // 0: star, 1: planet, 2: dust
#else
    simd_float4 color;
    uint32_t type; // 0: star, 1: planet, 2: dust
#endif
} ColorType;



typedef struct {
#ifdef __METAL_VERSION__
    uint64_t mortonCode;
    float3 centerOfMass;
    float totalMass;
    float4 emittedColor;
    float3 emittedColorCenter;
    uint32_t children[8];
    uint32_t layer;
#else
    uint64_t mortonCode;
    simd_float3 centerOfMass;
    float totalMass;
    simd_float4 emittedColor;
    simd_float3 emittedColorCenter;
    uint32_t children[8];
    uint32_t layer;
#endif
} OctreeNode;

typedef struct {
#ifdef __METAL_VERSION__
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
    float3 cameraPosition;
#else
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewMatrix;
    simd_float3 cameraPosition;
#endif
} GlobalUniforms;

typedef struct {
#ifdef __METAL_VERSION__
    float4 colors[8];
    float4 positions[8];
#else
    simd_float4 colors[8];
    simd_float4 positions[8];
#endif
} LightingInfluences;

// Sentinel value for invalid Morton code (shared between Metal and Swift)
#ifdef __METAL_VERSION__
#define INVALID_MORTON_CODE 0xFFFFFFFFFFFFFFFFull
#else
#define INVALID_MORTON_CODE 0xFFFFFFFFFFFFFFFFull
#endif


#endif /* bridge_h */
