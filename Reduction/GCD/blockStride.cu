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

template <const unsigned int block_size_power>
__global__ void treeReductionKernelForSmallerSizes(const length_t N, const input_t* __restrict__ a, output_t* __restrict__ total_gcd)
{
    length_t iter = threadIdx.x + (blockIdx.x << block_size_power);
    const length_t grid_stride = gridDim.x << block_size_power;

    output_t partial_gcd = a[0]; // assuming at least one element is present in array
    // Grid-stride loop
    while(__builtin_expect(iter < N, 1)){    // Branch prediction: Always take
        partial_gcd = GCD(partial_gcd, static_cast<output_t>(a[iter]));
        iter += grid_stride;
    }

    // Tree reduction
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0xFFFFFFFF, partial_gcd, 16));
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x0000FFFF, partial_gcd,  8));
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x000000FF, partial_gcd,  4));
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x0000000F, partial_gcd,  2));
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x00000003, partial_gcd,  1));

    // Writing to shared memory
    extern __shared__ output_t warp_partial_gcd_values[];
    const uint8_t lane_id = (threadIdx.x & 31);
    const uint8_t warp_id = (threadIdx.x >> 5);
    if (lane_id == 0) warp_partial_gcd_values[warp_id] = partial_gcd;
    __syncthreads();

    // Reducing values in shared memory
    if (warp_id == 0){
        partial_gcd = warp_partial_gcd_values[lane_id];
        if constexpr (block_size_power >= 10)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0xFFFFFFFF, partial_gcd, 16));
        if constexpr (block_size_power >=  9)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x0000FFFF, partial_gcd, 8));
        if constexpr (block_size_power >=  8)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x000000FF, partial_gcd, 4));
        if constexpr (block_size_power >=  7)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x0000000F, partial_gcd, 2));
        if constexpr (block_size_power >=  6)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x00000003, partial_gcd, 1));
        if (lane_id == 0)
            atomicGCD(total_gcd, partial_gcd);
    }
}

// Reduction kernel
template <const unsigned int block_size_power>
__global__ void treeReductionBlockStrideKernel (const length_t N, const input_t* __restrict__ a, output_t* __restrict__ total_gcd)
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
    output_t partial_gcd = a[iter];             // Assuming number of threads >= N
    iter += block_stride;
    while(__builtin_expect(iter < end, 1)){     // Branch prediction: always taken
        partial_gcd = GCD(partial_gcd, static_cast<output_t> (a[iter]));
        iter += block_stride;
    }
    // Remaining work items
    iter = (work_per_block * gridDim.x) + global_thread_id;
    if(iter < N) partial_gcd = GCD(partial_gcd, static_cast<output_t> (a[iter]));

    // Tree reduction
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0xFFFFFFFF, partial_gcd, 16));
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x0000FFFF, partial_gcd,  8));
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x000000FF, partial_gcd,  4));
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x0000000F, partial_gcd,  2));
    partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x00000003, partial_gcd,  1));

    // Writing to shared memory
    extern __shared__ output_t warp_partial_gcd_values[];
    const uint8_t lane_id = (threadIdx.x & 31);
    const uint8_t warp_id = (threadIdx.x >> 5);
    if (lane_id == 0) warp_partial_gcd_values[warp_id] = partial_gcd;
    __syncthreads();

    // Reducing values in shared memory
    if (warp_id == 0){
        partial_gcd = warp_partial_gcd_values[lane_id];
        if constexpr (block_size_power >= 10)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0xFFFFFFFF, partial_gcd, 16));
        if constexpr (block_size_power >=  9)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x0000FFFF, partial_gcd, 8));
        if constexpr (block_size_power >=  8)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x000000FF, partial_gcd, 4));
        if constexpr (block_size_power >=  7)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x0000000F, partial_gcd, 2));
        if constexpr (block_size_power >=  6)
            partial_gcd = GCD(partial_gcd, __shfl_down_sync(0x00000003, partial_gcd, 1));
        if (lane_id == 0)
            atomicGCD(total_gcd, partial_gcd);
    }
}

// Kernel invocation
void computeReduction(const length_t N, const input_t* d_a, output_t* d_gcd, const output_t init)
{
    // cudaMemset(d_xor_val, init, sizeof(output_t));
    cudaMemcpy(d_gcd, &init, sizeof(output_t), cudaMemcpyHostToDevice);

    // Kernel invocation
    if (N < block_size_256 * min_work_per_thread)
    {
        treeReductionKernelForSmallerSizes   <block_size_256_power>
                <<< 1, block_size_256, warp_size * sizeof(output_t) >>> (N, d_a, d_gcd);
    }
    else if (N < num_blocks_256 * block_size_256 * min_work_per_thread) 
    {
        treeReductionBlockStrideKernel       <block_size_256_power>
                <<< (N >> block_size_256_power) / min_work_per_thread, block_size_256, warp_size * sizeof(output_t) >>> (N, d_a, d_gcd);
    }
    else if (N < num_blocks_256 * block_size_512 * min_work_per_thread) 
    {
        treeReductionBlockStrideKernel       <block_size_256_power> 
               <<< num_blocks_256, block_size_256, warp_size * sizeof(output_t) >>> (N, d_a, d_gcd);
    }
    else
    {
       treeReductionBlockStrideKernel       <block_size_512_power> 
              <<< num_blocks_256, block_size_512, warp_size * sizeof(output_t) >>> (N, d_a, d_gcd);
    }

    cudaDeviceSynchronize();
    return;
}
