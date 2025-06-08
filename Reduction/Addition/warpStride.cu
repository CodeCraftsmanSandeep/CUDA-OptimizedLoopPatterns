#include "project_defs.cuh"

// Useful constants
constexpr int min_work_per_thread  = 16;
constexpr int block_size_256_power = 8;
constexpr int block_size_256       = (1 << block_size_256_power);
constexpr int block_size_512_power = 9;
constexpr int block_size_512       = (1 << block_size_512_power);
constexpr int num_blocks_256_power = 8;
constexpr int num_blocks_256       = (1 << num_blocks_256_power);
constexpr int warp_size_power      = 5;
constexpr int warp_size            = (1 << warp_size_power);

// Kernel for reduction
template <const int block_size_power>
__global__ void treeReductionWarpStrideKernel(
    const length_t N,
    const input_t* __restrict__ a,
    output_t* __restrict__ total_sum)
{
    // Thread algebra
    const unsigned int global_thread_id    = ((blockIdx.x << block_size_power) + threadIdx.x);
    const unsigned int global_warp_id      = (global_thread_id >> warp_size_power);
    const length_t num_chunks              = (N >> warp_size_power);
    const unsigned int num_warps           = (gridDim.x << (block_size_power - warp_size_power));
    const length_t chunks_per_warp         = (num_chunks / num_warps);
    const length_t work_per_warp           = chunks_per_warp << warp_size_power;
    length_t iter                          = (work_per_warp * global_warp_id) + (threadIdx.x & 31);
    const length_t end                     = iter + work_per_warp;
    constexpr unsigned int warp_stride     = warp_size;

    // Warp-stride loop
    output_t partial_sum = 0;
    while (__builtin_expect(iter < end, 1)) {
        partial_sum += a[iter];
        iter += warp_stride;
    }
    // Remaining work-items
    iter = work_per_warp * num_warps + global_thread_id;
    if (iter < N) partial_sum += a[iter];

    // Tree reduction within warp
    partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, 16);
    partial_sum += __shfl_down_sync(0x0000FFFF, partial_sum, 8);
    partial_sum += __shfl_down_sync(0x000000FF, partial_sum, 4);
    partial_sum += __shfl_down_sync(0x0000000F, partial_sum, 2);
    partial_sum += __shfl_down_sync(0x00000003, partial_sum, 1);

    // Writing to shared memory
    extern __shared__ output_t warp_partial_sum[];
    const uint8_t lane_id = (threadIdx.x & 31);
    const uint8_t warp_id = (threadIdx.x >> 5);
    if (lane_id == 0) warp_partial_sum[warp_id] = partial_sum;
    __syncthreads();

    // Reducing values in shared memory
    if (warp_id == 0) {
        partial_sum = warp_partial_sum[lane_id];
        if constexpr (block_size_power >= 10) partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, 16);
        if constexpr (block_size_power >= 9)  partial_sum += __shfl_down_sync(0x0000FFFF, partial_sum, 8);
        if constexpr (block_size_power >= 8)  partial_sum += __shfl_down_sync(0x000000FF, partial_sum, 4);
        if constexpr (block_size_power >= 7)  partial_sum += __shfl_down_sync(0x0000000F, partial_sum, 2);
        if constexpr (block_size_power >= 6)  partial_sum += __shfl_down_sync(0x00000003, partial_sum, 1);
        if (lane_id == 0) atomicAdd(total_sum, partial_sum);
    }
}

void computeReduction(const length_t N, const input_t* d_a, output_t* d_sum, output_t init)
{
    cudaMemset(d_sum, init, sizeof(output_t));

    // Kernel invocation
    if (N < block_size_256 * min_work_per_thread)
    {
        treeReductionWarpStrideKernel<block_size_256_power>
            <<<1, block_size_256, warp_size * sizeof(output_t)>>>(N, d_a, d_sum);
    }
    else if (N < num_blocks_256 * block_size_256 * min_work_per_thread)
    {
        treeReductionWarpStrideKernel<block_size_256_power>
            <<< (N >> block_size_256_power) / min_work_per_thread, block_size_256, warp_size * sizeof(output_t) >>>(N, d_a, d_sum);
    }
    else if (N < num_blocks_256 * block_size_512 * min_work_per_thread)
    {
        treeReductionWarpStrideKernel<block_size_256_power>
            <<<num_blocks_256, block_size_256, warp_size * sizeof(output_t)>>>(N, d_a, d_sum);
    }
    else
    {
        treeReductionWarpStrideKernel<block_size_512_power>
            <<<num_blocks_256, block_size_512, warp_size * sizeof(output_t)>>>(N, d_a, d_sum);
    }

    cudaDeviceSynchronize();
    return;
}

