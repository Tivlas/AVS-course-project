#include <omp.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

template <typename T>
void readFromBinary(std::vector<T>& v, const std::filesystem::path& fileName) {
    std::ifstream file(fileName, std::ios::binary);
    if (!file.good())
        throw std::invalid_argument("Invalid file! " + fileName.string());
    size_t size = 0;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    v.resize(size);
    file.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
    file.close();
}

template <typename T>
void matrixMul(const std::vector<T>& a, const std::vector<T>& b,
               std::vector<T>& result) {
    size_t size = sqrt(a.size());
    int j, k;
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (k = 0; k < size; k++) {
            for (j = 0; j < size; j++) {
                result[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }
}

template <typename T>
void matrixAdd(const std::vector<T>& a, const std::vector<T>& b,
               std::vector<T>& result) {
    size_t size = a.size();
    int i;
#pragma omp parallel for
    for (i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void printProgressIndicator(std::atomic<bool>& isCalculating) {
    const std::string indicators = "-\\|/";
    int index = 0;
    while (isCalculating) {
        std::cout << "\rCalculating... " << indicators[index++];
        index %= indicators.size();
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << '\n';
}

template <typename T>
auto measureTime(
    const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result,
    std::function<void(const std::vector<T>& a, const std::vector<T>& b,
                       std::vector<T>& result)>
        matrix_func) {
    auto start = std::chrono::high_resolution_clock::now();
    matrix_func(a, b, result);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return elapsed_time.count();
}

template <typename T>
void calculate(const std::vector<std::filesystem::path>& filePaths,
               const std::filesystem::path& outFileName) {
    std::fstream outfile(outFileName, std::ios::trunc | std::ios::out);
    if (!outfile.is_open()) {
        throw std::invalid_argument("Invalid outFileName! " +
                                    outFileName.string());
    }
    std::atomic<bool> isCalculating;
    isCalculating.store(true);
    std::thread progressThread(printProgressIndicator, std::ref(isCalculating));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (const auto& filePath : filePaths) {
        std::vector<T> a;
        readFromBinary(a, filePath);
        std::vector<T> b = a;
        std::vector<T> result(a.size(), 0);
        std::function<void(const std::vector<T>& a, const std::vector<T>& b,
                           std::vector<T>& result)>
            matrix_func;
        matrix_func = matrixMul<T>;
        auto time = measureTime<T>(a, b, result, matrix_func);
        outfile << std::to_string(time) + "    " +
                       filePath.filename().string() + "\n";
    }
    outfile.close();
    isCalculating.store(false);
    progressThread.join();
}

int main(int argc, char* argv[]) {
    omp_set_num_threads(omp_get_max_threads());
    std::filesystem::path outFileName = std::filesystem::absolute(argv[1]);
    std::string dataDir = argv[2];
    std::filesystem::directory_iterator iterator(dataDir);
    std::vector<std::filesystem::path> fileNames;
    for (const auto& entry : iterator) {
        if (entry.is_regular_file()) {
            std::filesystem::path filePath =
                std::filesystem::absolute(entry.path()).lexically_normal();
            fileNames.push_back(filePath);
        }
    }
    calculate<float>(fileNames, outFileName);
}
