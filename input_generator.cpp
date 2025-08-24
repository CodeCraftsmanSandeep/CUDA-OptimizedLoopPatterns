#include <iostream>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <cstring>
#include <random>
#include <climits>
#include <sys/stat.h>  // for mkdir
#include <sys/types.h>
#include <filesystem>

int c = 0;

template <typename T>
void generate_random_array(unsigned int n, T min_val, T max_val, T* arr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis1(0.0, 1.0);

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist2(777, 8888);
        std::uniform_int_distribution<T> dist3(-8989, -55);

        for (unsigned int i = 0; i < n; ++i) {
            if (c % 3 == 2) {
                arr[i] = (dis1(gen) < 0.8) ? dist2(gen) : dist3(gen);
            } else {
                arr[i] = (dis1(gen) < 0.6) ? dist2(gen) : dist3(gen);
            }
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (int i = 0; i < n; ++i) {
            arr[i] = dist(gen);
        }
    } else {
        static_assert(std::is_arithmetic<T>::value, "Type must be numeric.");
    }
}

template <typename T>
void generateRandomArrays(const unsigned n, const int fd) {
    T* arr = new T[n];

    generate_random_array<T>(n, INT_MAX / 2, INT_MAX, arr);

    char buff[30];
    snprintf(buff, 30, "%u\n", n);
    write(fd, buff, strlen(buff));

    if constexpr (std::is_integral<T>::value) {
        for (int i = 0; i < n; i++) {
            snprintf(buff, 30, "%d\n", arr[i]);
            write(fd, buff, strlen(buff));
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        for (int i = 0; i < n; i++) {
            snprintf(buff, 30, "%.6f\n", arr[i]);
            write(fd, buff, strlen(buff));
        }
    }

    delete[] arr;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./input_generator.out <size> <output_directory>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::string outputDir = argv[2];

    // Ensure directory ends with slash
    if (outputDir.back() != '/')
        outputDir += '/';

    // Create directory if it doesn't exist
    std::filesystem::create_directories(outputDir);

    const char* dataType = "int";
    std::string filePath = outputDir + dataType + "_" + std::to_string(n) + ".txt";

    int fd = open(filePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);

    if (fd < 0) {
        std::cerr << "File open failed at " << filePath << "\n";
        return 1;
    }

    std::cout << "Generating file: " << filePath << std::endl;
    generateRandomArrays<int>(n, fd);

    close(fd);
    std::cout << "✅ Data written successfully.\n";

    return 0;
}


