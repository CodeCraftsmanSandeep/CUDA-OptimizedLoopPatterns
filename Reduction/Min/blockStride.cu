#include "project_defs.cuh"

// Useful constants
constexpr unsigned int min_work_per_thread  = 16;
constexpr unsigned int block_size_256_power = 8;
constexpr unsigned int block_size_256       = (1 << block_size_256_power);
constexpr unsigned int block_size_512_power = 9;
constexpr unsigned int block_size_512       = (1 << block_size_512_power);
constexpr unsigned int num_blocks_256_power = 8;
constexpr unsigned int num_blocks_256       = (1 << num_blocks_256_power);
constexpr unsigned int warp_size_power      = 5;
constexpr unsigned int warp_size            = (1 << warp_size_power);

// Reduction kernel
template <const unsigned int block_size_power>
__global__ void treeReductionBlockStrideKernel (const length_t N, const input_t* __restrict__ a, output_t* __restrict__ total_min)
{
    // Thread algebra
    const unsigned int global_thread_id = threadIdx.x + (blockIdx.x << block_size_power);
    const length_t num_chunks           = (N >> block_size_power);
    const length_t chunks_per_block     = num_chunks / gridDim.x;
    const length_t work_per_block       = chunks_per_block << block_size_power;
    length_t iter                       = (work_per_block * blockIdx.x) + threadIdx.x;
    const length_t end                  = iter + work_per_block;
    constexpr unsigned int block_stride = (1 << block_size_power);

    // Block-stride loop
    output_t partial_min = INT_MAX;          // Taking INT_MAX as infinity
    while(__builtin_expect(iter < end, 1)){  // Branch prediction: always taken
        partial_min = min(partial_min, static_cast<output_t> (a[iter]));
        iter += block_stride;
    }
    // Remaining work items
    iter = (work_per_block * gridDim.x) + global_thread_id;
    if(iter < N) partial_min = min(partial_min, static_cast<output_t> (a[iter]));

    // Tree reduction
    partial_min = min(partial_min, __shfl_down_sync(0xFFFFFFFF, partial_min, 16));
    partial_min = min(partial_min, __shfl_down_sync(0x0000FFFF, partial_min,  8));
    partial_min = min(partial_min, __shfl_down_sync(0x000000FF, partial_min,  4));
    partial_min = min(partial_min, __shfl_down_sync(0x0000000F, partial_min,  2));
    partial_min = min(partial_min, __shfl_down_sync(0x00000003, partial_min,  1));

    // Writing to shared memory
    extern __shared__ output_t warp_partial_min_values[];
    const uint8_t lane_id = (threadIdx.x & 31);
    const uint8_t warp_id = (threadIdx.x >> 5);
    if (lane_id == 0) warp_partial_min_values[warp_id] = partial_min;
    __syncthreads();

    // Reducing values in shared memory
    if (warp_id == 0){
        partial_min = warp_partial_min_values[lane_id];
        if constexpr (block_size_power >= 10)
            partial_min = min(partial_min, __shfl_down_sync(0xFFFFFFFF, partial_min, 16));
        if constexpr (block_size_power >=  9)
            partial_min = min(partial_min, __shfl_down_sync(0x0000FFFF, partial_min, 8));
        if constexpr (block_size_power >=  8)
            partial_min = min(partial_min, __shfl_down_sync(0x000000FF, partial_min, 4));
        if constexpr (block_size_power >=  7)
            partial_min = min(partial_min, __shfl_down_sync(0x0000000F, partial_min, 2));
        if constexpr (block_size_power >=  6)
            partial_min = min(partial_min, __shfl_down_sync(0x00000003, partial_min, 1));
        if (lane_id == 0)
            atomicMin(total_min, partial_min);
    }

}

// Kernel invocation
void computeReduction(const length_t N, const input_t* d_a, output_t* d_xor_val, const output_t init)
{
    // cudaMemset(d_xor_val, init, sizeof(output_t));
    cudaMemcpy(d_xor_val, &init, sizeof(output_t), cudaMemcpyHostToDevice);

    // Kernel invocation
    if (N < block_size_256 * min_work_per_thread)
    {
        treeReductionBlockStrideKernel       <block_size_256_power> 
                <<< 1, block_size_256, warp_size * sizeof(output_t) >>> (N, d_a, d_xor_val);
    }
    else if (N < num_blocks_256 * block_size_256 * min_work_per_thread) 
    {
        treeReductionBlockStrideKernel       <block_size_256_power>
                <<< (N >> block_size_256_power) / min_work_per_thread, block_size_256, warp_size * sizeof(output_t) >>> (N, d_a, d_xor_val);
    }
    else if (N < num_blocks_256 * block_size_512 * min_work_per_thread) 
    {
        treeReductionBlockStrideKernel       <block_size_256_power> 
               <<< num_blocks_256, block_size_256, warp_size * sizeof(output_t) >>> (N, d_a, d_xor_val);
    }
    else
    {
       treeReductionBlockStrideKernel       <block_size_512_power> 
              <<< num_blocks_256, block_size_512, warp_size * sizeof(output_t) >>> (N, d_a, d_xor_val);
    }

    cudaDeviceSynchronize();
    return;
}
