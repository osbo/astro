#include <metal_stdlib>
using namespace metal;


#define SCAN_BLOCKSIZE 512


enum ScanBufferIndex {
    ScanBufferIndexInput,
    ScanBufferIndexOutput,
    ScanBufferIndexAux,
    ScanBufferIndexLength,
    ScanBufferIndexZeroff,
};


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



kernel void prefixFixup(device uint *input [[ buffer(ScanBufferIndexInput) ]],
                         device uint *aux [[ buffer(ScanBufferIndexAux) ]],
                         constant uint& len [[ buffer(ScanBufferIndexLength) ]],
                         uint threadIdx [[ thread_position_in_threadgroup ]],
                         uint blockIdx [[ threadgroup_position_in_grid ]])
{
    uint t = threadIdx;
    uint start = t + 2 * blockIdx * SCAN_BLOCKSIZE;
    uint block_sum = aux[blockIdx];
    
    if (start < len)                    input[start] += block_sum;
    if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += block_sum;
}

kernel void prefixSum(device const uint* input [[ buffer(ScanBufferIndexInput) ]],
                       device uint* output [[ buffer(ScanBufferIndexOutput) ]],
                       device uint* aux [[ buffer(ScanBufferIndexAux) ]],
                       constant uint& len [[ buffer(ScanBufferIndexLength) ]],
                       constant uint& zeroff [[ buffer(ScanBufferIndexZeroff) ]],
                       uint threadIdx [[ thread_position_in_threadgroup ]],
                       uint blockIdx [[ threadgroup_position_in_grid ]])
{
    threadgroup uint scan_array[SCAN_BLOCKSIZE << 1];
    uint t1 = threadIdx + 2 * blockIdx * SCAN_BLOCKSIZE;
    uint t2 = t1 + SCAN_BLOCKSIZE;
    

    scan_array[threadIdx] = (t1<len) ? input[t1] : 0;
    scan_array[threadIdx + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    

    for (uint stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
        uint index = (threadIdx + 1) * stride * 2 - 1;
        if (index < 2 * SCAN_BLOCKSIZE)
            scan_array[index] += scan_array[index - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    

    for (uint stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
        uint index = (threadIdx + 1) * stride * 2 - 1;
        if (index + stride < 2 * SCAN_BLOCKSIZE)
            scan_array[index + stride] += scan_array[index];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    

    if (t1+zeroff < len)    output[t1+zeroff] = scan_array[threadIdx];
    if (t2+zeroff < len)    output[t2+zeroff] = (threadIdx==SCAN_BLOCKSIZE-1 && zeroff) ? 0 : scan_array[threadIdx + SCAN_BLOCKSIZE];
    if ( threadIdx == 0 ) {
        if ( zeroff ) output[0] = 0;
        if (aux) aux[blockIdx] = scan_array[2 * SCAN_BLOCKSIZE - 1];
    }
}



kernel void split_prep(device const ulong* input [[ buffer(SplitBufferIndexInput) ]],
                       constant uint& bit [[ buffer(SplitBufferIndexBit) ]],
                       device uint* e [[ buffer(SplitBufferIndexE) ]],
                       constant uint& count [[ buffer(SplitBufferIndexCount) ]],
                       uint tid [[ thread_position_in_grid ]])
{
    if(tid >= count) {
        return;
    }

    e[tid] = (input[tid] & (1UL << bit)) == 0;
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


    uint totalFalses = f[count-1] + e[count-1];
    

    bool isBitSet = (input[tid] & (1UL << bit)) != 0;
    
    uint destinationIndex;
    if (isBitSet) {
        destinationIndex = totalFalses + (tid - f[tid]);
    } else {
        destinationIndex = f[tid];
    }

    output[destinationIndex] = input[tid];
    outputIndices[destinationIndex] = inputIndices[tid];
}



kernel void copyBuffer(device const ulong* input [[ buffer(0) ]],
                       device ulong* output [[ buffer(1) ]],
                       device const uint* countBuffer [[ buffer(2) ]],
                       uint tid [[ thread_position_in_grid ]])
{
    uint count = countBuffer[0];
    if (tid >= count) return;
    output[tid] = input[tid];
}

kernel void copyIndices(device const uint* input [[ buffer(0) ]],
                        device uint* output [[ buffer(1) ]],
                        device const uint* countBuffer [[ buffer(2) ]],
                        uint tid [[ thread_position_in_grid ]])
{
    uint count = countBuffer[0];
    if (tid >= count) return;
    output[tid] = input[tid];
}