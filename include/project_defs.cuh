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

