//
//  RadixSort.metal
//  astro
//
//  Adapted from MetalKernels by Audulus LLC
//

#include <metal_stdlib>
using namespace metal;

// Constants
#define SCAN_BLOCKSIZE 512

// Buffer indices for scan operations
enum ScanBufferIndex {
    ScanBufferIndexInput,
    ScanBufferIndexOutput,
    ScanBufferIndexAux,
    ScanBufferIndexLength,
    ScanBufferIndexZeroff,
    ScanBufferIndexIndirectArguments,
    ScanBufferIndexLengths,
};

// Buffer indices for split operations
enum SplitBufferIndex {
    SplitBufferIndexInput,
    SplitBufferIndexInputIndices,
    SplitBufferIndexOutput,
    SplitBufferIndexOutputIndices,
    SplitBufferIndexCount,
    SplitBufferIndexBit,
    SplitBufferIndexE,
    SplitBufferIndexF
};

// --- Scan Kernels ---

kernel void scan_threadgroups(constant uint& len [[ buffer(ScanBufferIndexLength) ]],
                              device MTLDispatchThreadgroupsIndirectArguments* args [[ buffer(ScanBufferIndexIndirectArguments) ]],
                              device uint* lengths [[ buffer(ScanBufferIndexLengths) ]])
{
    args[0] = {uint(len/SCAN_BLOCKSIZE + 1), 1, 1};
    args[1] = {uint(len/(SCAN_BLOCKSIZE*SCAN_BLOCKSIZE) + 1), 1, 1};
    args[2] = {1, 1, 1};

    lengths[0] = len;
    lengths[1] = len/SCAN_BLOCKSIZE;
    lengths[2] = SCAN_BLOCKSIZE;
}

kernel void prefixFixup(device uint *input [[ buffer(ScanBufferIndexInput) ]],
                         device uint *aux [[ buffer(ScanBufferIndexAux) ]],
                         device const uint& len [[ buffer(ScanBufferIndexLength) ]],
                         uint threadIdx [[ thread_position_in_threadgroup ]],
                         uint blockIdx [[ threadgroup_position_in_grid ]])
{
    unsigned int t = threadIdx;
    unsigned int start = t + 2 * blockIdx * SCAN_BLOCKSIZE;
    if (start < len)                    input[start] += aux[blockIdx];
    if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[blockIdx];
}

kernel void prefixSum(device uint* input [[ buffer(ScanBufferIndexInput) ]],
                       device uint* output [[ buffer(ScanBufferIndexOutput) ]],
                       device uint* aux [[ buffer(ScanBufferIndexAux) ]],
                       device const uint& len [[ buffer(ScanBufferIndexLength) ]],
                       constant uint& zeroff [[ buffer(ScanBufferIndexZeroff) ]],
                       uint threadIdx [[ thread_position_in_threadgroup ]],
                       uint blockIdx [[ threadgroup_position_in_grid ]])
{
    threadgroup uint scan_array[SCAN_BLOCKSIZE << 1];
    unsigned int t1 = threadIdx + 2 * blockIdx * SCAN_BLOCKSIZE;
    unsigned int t2 = t1 + SCAN_BLOCKSIZE;
    
    // Pre-load into shared memory
    scan_array[threadIdx] = (t1<len) ? input[t1] : 0;
    scan_array[threadIdx + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction
    int stride;
    for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
        int index = (threadIdx + 1) * stride * 2 - 1;
        if (index < 2 * SCAN_BLOCKSIZE)
            scan_array[index] += scan_array[index - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Post reduction
    for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
        int index = (threadIdx + 1) * stride * 2 - 1;
        if (index + stride < 2 * SCAN_BLOCKSIZE)
            scan_array[index + stride] += scan_array[index];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Output values & aux
    if (t1+zeroff < len)    output[t1+zeroff] = scan_array[threadIdx];
    if (t2+zeroff < len)    output[t2+zeroff] = (threadIdx==SCAN_BLOCKSIZE-1 && zeroff) ? 0 : scan_array[threadIdx + SCAN_BLOCKSIZE];
    if ( threadIdx == 0 ) {
        if ( zeroff ) output[0] = 0;
        if (aux) aux[blockIdx] = scan_array[2 * SCAN_BLOCKSIZE - 1];
    }
}

// --- Split Kernels ---

kernel void split_prep(device const ulong* input [[ buffer(SplitBufferIndexInput) ]],
                       constant uint& bit [[ buffer(SplitBufferIndexBit) ]],
                       device uint* e [[ buffer(SplitBufferIndexE) ]],
                       constant uint& count [[ buffer(SplitBufferIndexCount) ]],
                       uint tid [[ thread_position_in_grid ]])
{
    if(tid >= count) {
        return;
    }

    e[tid] = (input[tid] & (1UL<<bit)) == 0;
}

kernel void split_scatter(device const ulong* input [[ buffer(SplitBufferIndexInput) ]],
                          device const uint* inputIndices [[ buffer(SplitBufferIndexInputIndices) ]],
                          device ulong* output [[ buffer(SplitBufferIndexOutput) ]],
                          device uint* outputIndices [[ buffer(SplitBufferIndexOutputIndices) ]],
                          constant uint& bit [[ buffer(SplitBufferIndexBit) ]],
                          device const uint* e [[ buffer(SplitBufferIndexE) ]],
                          device const uint* f [[ buffer(SplitBufferIndexF) ]],
                          constant uint& count [[ buffer(SplitBufferIndexCount) ]],
                          uint tid [[ thread_position_in_grid ]])
{
    if(tid >= count) {
        return;
    }

    // Get the total number of falses (elements with bit == 0)
    uint totalFalses = f[count-1] + e[count-1];
    bool b = (input[tid] & (1UL<<bit)) != 0;
    uint d;
    
    if (b) {
        // This element has bit == 1, so it goes after all the falses
        d = totalFalses + (tid - f[tid]);
    } else {
        // This element has bit == 0, so it goes at position f[tid]
        d = f[tid];
    }

    output[d] = input[tid];
    outputIndices[d] = inputIndices[tid];
}
