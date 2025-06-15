#pragma once
#include <cuda.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <sys/time.h>

#define printError(fp) std::cerr << "Error in file " << __FILE__ << " at line: " << __LINE__ << std::endl
#define ERROR_FILE std::cerr
#define OUTPUT_FILE std::cout
#define NUM_RUNS 11

using length_t = unsigned int;
using input_t = int;
using output_t = long long int;

inline void printGPUUsage(const char *msg) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << msg << " | Free: " << free_mem / (1024.0 * 1024.0) << " MB, Used: "
              << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB, Total: "
              << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
}

inline double rtClock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) std::cerr << "Error return from gettimeofday: " << stat << std::endl;
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

class result{
private:
    char* file_name;
    length_t N;
    output_t sum;
    double mean_time, median_time, standard_deviation;
    length_t num_runs;

public:
    // Setters
    void setFileName(char* file_name){
        this->file_name = file_name;
    }
    void setSize(length_t N){
        this->N = N;
    }
    void setSum(output_t sum){
        this->sum = sum;
    }
    void setMeanTime(double mean_time){
        this->mean_time = mean_time;
    }
    void setMedianTime(double median_time){
        this->median_time = median_time;
    }
    void setStandardDeviation(double standard_deviation){
        this->standard_deviation = standard_deviation;
    }
    void setNumRuns(length_t num_runs){
        this->num_runs = num_runs;
    }

    // Getters
    char* getFileName(){
        return file_name;
    }
    length_t getSize(){
        return N;
    }
    output_t getSum(){
        return sum;
    }
    double getMeanTime(){
        return mean_time;
    }
    double getMedianTime(){
        return median_time;
    }
    double getStandardDeviation(){
        return standard_deviation;
    }
    length_t getNumRuns(){
        return num_runs;
    }
};

inline void printResult(result* curr_result){
    OUTPUT_FILE << curr_result->getFileName() << ","
                << curr_result->getSize() << ","
                << curr_result->getSum() << ","
                << curr_result->getNumRuns() << ","
                << curr_result->getMeanTime() << ","
                << curr_result->getMedianTime() << ","
                << curr_result->getStandardDeviation() << std::endl;
}

inline double findMean(const std::vector<double>& times){
    if(times.empty()) return 0.0;
    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

inline double findMedian(std::vector<double> times){
    if(times.empty()) return 0.0;
    std::sort(times.begin(), times.end());
    if(times.size() & 1) return times[times.size()/2];
    return (times[times.size()/2 - 1] + times[times.size()/2]) / 2;
}

inline double findStandardDeviation(const std::vector<double>& times) {
    if(times.size() <= 1) return 0.0;
    double mean = findMean(times);
    double variance = 0.0;
    for (double x : times) {
        variance += (x - mean) * (x - mean);
    }
    variance /= (times.size() - 1);
    return std::sqrt(variance);
}

// Custom atomicAdd for datatype: (long long int)
inline __device__ long long int atomicAdd(long long int* address, long long int val){
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + (unsigned long long)val);
    } while (assumed != old);

    return old;
}

// Custom atomicXor for datatype: long long int
inline __device__ long long int atomIicXor(long long int* address, long long int val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed ^ (unsigned long long)val);
    } while (assumed != old);

    return old;
}

inline __host__ __device__
int GCD(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// __host__ __device__ GCD function from previous example
inline __host__ __device__
long long int GCD(long long int a, long long int b) {
    // Handle cases where one or both numbers are zero
    // gcd(0, x) = |x|
    // gcd(x, 0) = |x|
    // gcd(0, 0) = 0 (conventionally)
    if (a == 0) return (b < 0) ? -b : b;
    if (b == 0) return (a < 0) ? -a : a;

    // Use Euclidean algorithm
    a = (a < 0) ? -a : a; // Take absolute values for GCD calculation
    b = (b < 0) ? -b : b;

    while (b != 0) {
        long long int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Custom atomicGCD for datatype: long long int
// Atomically updates *address with gcd(*address, val)
// Returns the old value that was in *address before the update.
inline __device__ long long int atomicGCD(long long int* address, long long int val) {
    // Cast to unsigned long long int for atomicCAS, as it operates on raw bits.
    // This is safe as long as the bit patterns are compatible (which they are for same-sized integers).
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old_ull_value;
    unsigned long long int assumed_ull_value;
    long long int new_gcd_value_ll; // To store the computed GCD before casting to ULL

    // Volatile is used to ensure the compiler doesn't optimize away reads within the loop,
    // though atomicCAS typically provides sufficient memory barriers.
    // However, it's good practice for values being actively spun on.
    volatile unsigned long long int current_ull_in_memory = *address_as_ull;

    do {
        assumed_ull_value = current_ull_in_memory; // Assume this is the current value

        // Convert assumed_ull_value back to long long int to perform GCD arithmetic
        long long int assumed_ll_value = (long long int)assumed_ull_value;

        // Calculate the new GCD value
        new_gcd_value_ll = GCD(assumed_ll_value, val);

        // Attempt to atomically compare and swap
        // If 'current_ull_in_memory' is still 'assumed_ull_value', then update it to 'new_gcd_value_ll'
        // Otherwise, 'current_ull_in_memory' is updated with the actual value found in *address_as_ull
        old_ull_value = atomicCAS(address_as_ull, assumed_ull_value, (unsigned long long int)new_gcd_value_ll);

        // If 'old_ull_value' is different from 'assumed_ull_value', it means another thread
        // modified the *address before our CAS. We need to retry with the new actual value.
        current_ull_in_memory = old_ull_value;

    } while (assumed_ull_value != old_ull_value); // Loop until CAS succeeds (old_ull_value matches assumed_ull_value)

    // Return the value that was in *address before this atomic operation began.
    // This is the 'old_ull_value' from the successful CAS.
    return (long long int)old_ull_value;
}
