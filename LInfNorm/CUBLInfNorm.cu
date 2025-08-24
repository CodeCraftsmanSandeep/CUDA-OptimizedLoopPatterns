#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define printError(fp) fprintf(fp, "Error in file %s at line: %d\n", __FILE__, __LINE__) 
#define ERROR_FILE stderr
#define OUTPUT_FILE stdout

void printGPUUsage(const char *msg) 
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("%s | Free: %.2f MB, Used: %.2f MB, Total: %.2f MB\n",
           msg,
           free_mem / (1024.0 * 1024.0),
           (total_mem - free_mem) / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));
}

double rtClock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

double findMean(const std::vector <double> times)
{
    if(times.size() == 0) return 0;

    return std::accumulate(times.begin(), times.end(), (double)0) / times.size();
}

double findMedian(std::vector <double> times)
{
    if(times.size() == 0) return 0;

    std::sort(times.begin(), times.end());
    if(times.size() & 1) return times[times.size()/2];
    return (times[times.size()/2 - 1] + times[times.size()/2])/2;
}

double findStandardDeviation(const std::vector<double> times) 
{
    if(times.size() <= 1) return 0;

    // Calculate mean
    double mean = findMean(times);

    // Calculate variance
    double variance = 0.0;
    for (double x : times) {
        variance += (x - mean) * (x - mean);
    }
    variance /= (times.size() - 1);

    // Return standard deviation
    return std::sqrt(variance);
}

const unsigned int NUM_RUNS = 11;
std::vector <double> execution_times;

// Functor for absolute difference calculation (returns int)
struct AbsDiffOpInt {
    const int* a;
    const int* b;

    AbsDiffOpInt(const int* a, const int* b) : a(a), b(b) {}

    __device__ int operator()(int idx) const {
        int diff = a[idx] - b[idx];
        return (diff < 0) ? -diff : diff;
    }
};

// Compute the L∞ norm (max absolute difference) between C and B,
// running NUM_RUNS times and collecting runtimes in execution_times.
int solve(const unsigned int n,
               const int* __restrict__ C,
               const int* __restrict__ B)    // passed in by caller
{
    // Device pointers
    int *d_C = nullptr, *d_B = nullptr, *d_result = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate device memory
    cudaMalloc(&d_C, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_C, C, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set up transform iterator for abs-diffs
    cub::CountingInputIterator<int> count_iter(0);
    AbsDiffOpInt abs_diff_op(d_C, d_B);
    cub::TransformInputIterator<int, AbsDiffOpInt, decltype(count_iter)>
        transform_iter(count_iter, abs_diff_op);

    // Determine temporary storage requirements
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           transform_iter, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int linf = 0;
    // Timing runs
    for (int run = 1; run <= NUM_RUNS; ++run) {
        double t_start = rtClock();

        // Perform max-reduction
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                               transform_iter, d_result, n);

           cudaMemcpy(&linf, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        double t_end = rtClock();
        execution_times.push_back((t_end - t_start) * 1e3);  // ms
    }

    // Optional: copy back final L∞ value if you need it

    // Cleanup
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_result);
    cudaFree(d_temp_storage);

    return linf;
}


__device__ int device_var;
__global__ void wakeUpKernel(){
    // A simple wakeUpKernel
    device_var = 2 * device_var * 100;
}

int main(const int argc, char* argv[]){
    // Waking up GPU
    wakeUpKernel <<<1, 1>>> ();
    cudaDeviceSynchronize();

    // Checking command line arguments
    if(argc != 3){
        printError(ERROR_FILE); 
        fprintf(ERROR_FILE, "Expected command line argument: input1_file_path input2_file_path\n");
        exit(EXIT_FAILURE);
    }

    const char* input1_file_path    = argv[1];
    const char* input2_file_path    = argv[2];

    // Taking C as input
    // nC ~> length of vector
    unsigned int nC;
    int* C;
    {
        FILE* fp = fopen(input1_file_path, "r");
        if (fp == NULL) {
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Opening %s failed!\n", input1_file_path);
            return 1;
        }

        if (fscanf(fp, "%u", &nC) != 1) {
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Failed to read nC from %s\n", input1_file_path);
            fclose(fp);
            return 1;
       }

        C = (int*) malloc(nC * sizeof(int));
        if (C == NULL) {
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Memory allocation failed for C\n");
            fclose(fp);
            return 1;
        }

        for (int i = 0; i < nC; i++) {
            if (fscanf(fp, "%d", &C[i]) != 1) {
                printError(ERROR_FILE);
                fprintf(ERROR_FILE, "Failed to read C[%d] from %s\n", i, input1_file_path);
                free(C);
                fclose(fp);
                return 1;
            }
        }

        fclose(fp);
    }

    // Taking matrix B as input
    // nB ~> length of vector
    unsigned int  nB;
    int* B;
    {
        FILE* fp = fopen(input2_file_path, "r");
        if (fp == NULL) {
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Opening %s failed!\n", input2_file_path);
            return 1;
        }

        if (fscanf(fp, "%u", &nB) != 1) {
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Failed to read nB from %s\n", input2_file_path);
            fclose(fp);
            return 1;
       }

        B = (int*) malloc(nB * sizeof(int));
        if (B == NULL) {
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Memory allocation failed for B\n");
            fclose(fp);
            return 1;
        }

        for (int i = 0; i < nB; i++) {
            if (fscanf(fp, "%d", &B[i]) != 1) {
                printError(ERROR_FILE);
                fprintf(ERROR_FILE, "Failed to read B[%d] from %s\n", i, input2_file_path);
                free(C);
                fclose(fp);
                return 1;
            }
        }

        fclose(fp);
    }

    if(nC != nB)
    {
        printError(ERROR_FILE);
        fprintf(ERROR_FILE, "Length of vectors should be same\n");
        return 1;
    }

    // Solving
    int LInf_norm = solve(nC, C, B);

    // Printing results
    {
        fprintf(OUTPUT_FILE, "n,LInf_norm,mean-time(ms),median-time(ms),std-deviation(ms)\n");
        fprintf(OUTPUT_FILE, "%d,%d,%.6f,%.6f,%.6f\n", nC, LInf_norm, findMean(execution_times), findMedian(execution_times), findStandardDeviation(execution_times));
    }

    free(B);
    free(C);

    return 0;
}
