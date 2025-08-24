#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

// Functor for fused multiply + reduce
struct DotProductOp {
    const float* a;
    const float* b;

    DotProductOp(const float* a, const float* b) : a(a), b(b) {}

    __device__ float operator()(int idx) const {
        return a[idx] * b[idx];
    }
};

int main() {
    const int n = 1000;
    float *h_a = new float[n];
    float *h_b = new float[n];
    float h_result = 0;

    // Initialize inputs
    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;  // Expected dot product = 1*2*1000 = 2000
    }

    // Device pointers
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 1. Create transform iterator
    cub::CountingInputIterator<int> count_iter(0);
    DotProductOp dot_op(d_a, d_b);
    cub::TransformInputIterator<float, DotProductOp, decltype(count_iter)> 
        transform_iter(count_iter, dot_op);

    // 2. Determine temp storage size
    void* d_temp_storage = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_bytes, transform_iter, d_result, n);
    
    // 3. Allocate temp storage
    cudaMalloc(&d_temp_storage, temp_bytes);
    
    // 4. Single API call: Compute dot product via transform-reduce
    cub::DeviceReduce::Sum(d_temp_storage, temp_bytes, transform_iter, d_result, n);
    
    // Retrieve result
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Dot product: " << h_result << std::endl;

    // Cleanup
    cudaFree(d_a); 
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFree(d_temp_storage);
    delete[] h_a;
    delete[] h_b;

    return 0;
}
