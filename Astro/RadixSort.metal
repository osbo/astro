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

kernel void prefixFixup(device uint *input [[ buffer(ScanBufferIndexInput) ]],
                         device uint *aux [[ buffer(ScanBufferIndexAux) ]],
                         constant uint& len [[ buffer(ScanBufferIndexLength) ]],
                         uint threadIdx [[ thread_position_in_threadgroup ]],
                         uint blockIdx [[ threadgroup_position_in_grid ]])
{
    threadgroup_barrier(mem_flags::mem_device); // Ensure aux is fully computed
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
    
    // Pre-load into shared memory
    scan_array[threadIdx] = (t1<len) ? input[t1] : 0;
    scan_array[threadIdx + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction
    for (uint stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
        uint index = (threadIdx + 1) * stride * 2 - 1;
        if (index < 2 * SCAN_BLOCKSIZE)
            scan_array[index] += scan_array[index - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Post reduction (down-sweep)
    for (uint stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
        uint index = (threadIdx + 1) * stride * 2 - 1;
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
    // e[i] is 1 if the bit is 0, and 0 if the bit is 1.
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

    // Get the total number of elements with bit 0 (falses)
    // This requires reading the last element of the scanned buffer (f)
    uint totalFalses = f[count-1] + e[count-1];
    
    // Check if the current element's bit is set or not
    bool isBitSet = (input[tid] & (1UL << bit)) != 0;
    
    uint destinationIndex;
    if (isBitSet) {
        // This element has bit == 1, so it goes into the 'true' partition.
        // Its position is after all the 'false' elements.
        // tid - f[tid] gives the index within the 'true' partition.
        destinationIndex = totalFalses + (tid - f[tid]);
    } else {
        // This element has bit == 0, so it goes into the 'false' partition.
        // Its position is simply its scanned index from f.
        destinationIndex = f[tid];
    }

    output[destinationIndex] = input[tid];
    outputIndices[destinationIndex] = inputIndices[tid];
}

// --- Copy Kernel ---

kernel void copyBuffer(device const ulong* input [[ buffer(0) ]],
                       device ulong* output [[ buffer(1) ]],
                       device const uint* countBuffer [[ buffer(2) ]],
                       uint tid [[ thread_position_in_grid ]])
{
    uint count = countBuffer[0];
    if(tid >= count) {
        return;
    }
    output[tid] = input[tid];
}

kernel void copyIndices(device const uint* input [[ buffer(0) ]],
                        device uint* output [[ buffer(1) ]],
                        device const uint* countBuffer [[ buffer(2) ]],
                        uint tid [[ thread_position_in_grid ]])
{
    uint count = countBuffer[0];
    if(tid >= count) {
        return;
    }
    output[tid] = input[tid];
}