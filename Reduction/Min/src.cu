#include "project_defs.cuh"

// computeReduction in another file, from which kernel is invoked to compute the results
output_t computeReduction(const length_t, const input_t*, output_t*, output_t);

void getResult(const length_t N, input_t* a, result* curr_result, output_t host_sum){
    // Allocating memory on device
    input_t* d_a;
    cudaError_t err = cudaMalloc(&d_a, N * sizeof(input_t));
    if(err != cudaSuccess){
        printError(ERROR_FILE);
        ERROR_FILE << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        free(curr_result);
        free(a);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_a, a, N * sizeof(input_t), cudaMemcpyHostToDevice);

    std::vector<double> times;
    output_t sum, curr_sum;

    output_t* d_sum;
    cudaMalloc(&d_sum, sizeof(output_t));

    double start_time, end_time, time_consumed;

    // Computing Reduction and saving result
    start_time = rtClock();
    computeReduction (N, d_a, d_sum, INT_MAX);
    cudaMemcpy(&sum, d_sum, sizeof(output_t), cudaMemcpyDeviceToHost);
    end_time = rtClock();
    
    time_consumed = end_time - start_time;
    times.push_back(time_consumed * 1e3);

    if(sum != host_sum){
        printError(ERROR_FILE);
        ERROR_FILE << "Host sum (" << host_sum << ") != Device sum (" << sum << ")" << std::endl;
        free(curr_result);
        cudaFree(d_a);
        free(a);
        exit(EXIT_FAILURE);
    }

    for(length_t RUN = 1; RUN < NUM_RUNS; RUN++){
        start_time = rtClock();
        computeReduction (N, d_a, d_sum, INT_MAX);
        cudaMemcpy(&curr_sum, d_sum, sizeof(output_t), cudaMemcpyDeviceToHost);
        end_time = rtClock();

        time_consumed = end_time - start_time;
        times.push_back(time_consumed * 1e3);
        
        if(curr_sum != sum){
            printError(ERROR_FILE);
            std::cerr << "curr_sum = " << curr_sum << ", prev_sum = " << sum << std::endl;
            ERROR_FILE << "The computation is incorrect!" << std::endl;
            cudaFree(d_a);
            free(curr_result);
            exit(EXIT_FAILURE);
        }
    }

    cudaFree(d_a);
    curr_result->setSum(sum);
    curr_result->setMeanTime(findMean(times));
    curr_result->setMedianTime(findMedian(times));
    curr_result->setStandardDeviation(findStandardDeviation(times));
}

// Host function
output_t host_reduce(const length_t N, input_t* a){
    output_t host_sum = INT_MAX;
    for(length_t i = 0; i < N; i++) host_sum = min(host_sum, (output_t)a[i]);
    return host_sum;
}

__device__ int wakeUpVar;
__global__ void wakeUpKernel(){
    // A simple wakeUpKernel
    wakeUpVar = 10 * threadIdx.x * blockIdx.x;
}

int main(const int argc, char* argv[]){
    // Waking up GPU
    wakeUpKernel <<<1, 1>>> ();
    cudaDeviceSynchronize();

    // Checking command line arguments
    if(argc != 2){
        printError(ERROR_FILE); 
        ERROR_FILE << "Expected command line argument: input_file_names" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Output header
    OUTPUT_FILE << "File-name,N,Result,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)" << std::endl; 

    // Getting input file name from command line arguments
    char* input_file_name = argv[1];

    // Opening input file for reading input
    FILE* fp = fopen(input_file_name, "r");
    if(fp == NULL){
        printError(ERROR_FILE);
        ERROR_FILE << "Opening " << input_file_name << " failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Reading array size
    length_t N;
    if(fscanf(fp, "%u\n", &N) != 1){
        printError(ERROR_FILE);
        ERROR_FILE << "Reading array size failed!" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Allocating memory dynamically on host
    input_t* a = (input_t*)malloc(N * sizeof(input_t));
    if(a == NULL){
        printError(ERROR_FILE);
        ERROR_FILE << "Memory allocation for " << N << " elements on host failed!" << std::endl;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Reading elements from the file
    for(length_t i = 0; i < N; i++){
        if(fscanf(fp, "%d", &a[i]) != 1){
            printError(ERROR_FILE);
            ERROR_FILE << "Reading element " << i << " from file " << input_file_name << " failed!" << std::endl;
            fclose(fp);
            free(a);
            exit(EXIT_FAILURE);
        }
    }

    // Closing the input file
    fclose(fp);

    // Storing the result 
    result* curr_result = (result*)malloc(sizeof(result));
    if(curr_result == NULL){
        printError(ERROR_FILE);
        ERROR_FILE << "Memory allocation failed for result!" << std::endl;
        free(a);
        exit(EXIT_FAILURE);
    }
    curr_result->setSize(N);
    curr_result->setFileName(input_file_name);
    curr_result->setNumRuns(NUM_RUNS);

    // Computing on host for checking sequential consistency
    output_t host_sum = host_reduce(N, a);

    // Calling getResult function
    getResult (N, a, curr_result, host_sum);

    // Calling printResult function
    printResult (curr_result);

    // Deallocating memory on host
    free(a);
    free(curr_result);

    return 0;
}

