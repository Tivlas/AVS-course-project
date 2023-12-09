#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <atomic>
#include <math.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 16
template <typename T>
__global__ void matrixMulKernel(T* c, const T* a, const T* b, const size_t N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int aBegin = N * BLOCK_SIZE * by;
    int aEnd = aBegin + N - 1;
    int aStep = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;

    int bStep = BLOCK_SIZE * N;
    T sum = 0;
    for (int ia = aBegin, ib = bBegin; ia < aEnd; ia += aStep, ib += bStep) {
        __shared__ T as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ T bs[BLOCK_SIZE][BLOCK_SIZE];
        as[ty][tx] = a[ia + N * ty + tx];
        bs[ty][tx] = b[ib + N * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[ty][k] * bs[k][tx];
        __syncthreads();
    }
    int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    c[ic] = sum;
}

template <typename T>
__global__ void matrixAddKernel(T* c, const T* a, const T* b, size_t N) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        auto idx = i * N + j;
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
void readFromBinary(thrust::host_vector<T>& v,
                    const std::filesystem::path& fileName) {
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
void saveToBinary(const thrust::host_vector<T>& v,  const std::filesystem::path& fileName) {
	std::ofstream file(fileName, std::ios::binary | std::ios::trunc | std::ios::out);
	file.seekp(0);
	size_t size = v.size();
	file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(v.data()), size * sizeof(T));
	file.close();
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
long long measureTime(const thrust::device_vector<T>& a,
                      thrust::device_vector<T>& b,
                      thrust::device_vector<T>& result, size_t N,
                      dim3 blockSize, dim3 numBlocks) {
    auto start = std::chrono::high_resolution_clock::now();
    matrixMulKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(result.data()),
        thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()),
        N);
    cudaDeviceSynchronize();
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
        thrust::host_vector<T> h_a;
        readFromBinary<T>(h_a, filePath);
        thrust::device_vector<T> d_a = h_a;
        thrust::device_vector<T> d_b = h_a;
        thrust::device_vector<T> d_c(h_a.size());
        const size_t N = sqrt(h_a.size());
        dim3 blockSize(16, 16);
        dim3 numBlocks((N + blockSize.x - 1) / blockSize.x,
                       (N + blockSize.y - 1) / blockSize.y);
        auto time = measureTime<T>(d_a, d_b, d_c, N, blockSize, numBlocks);
        outfile << std::to_string(time) + "    " +
                       filePath.filename().string() + "\n";
        h_a = d_c;
        auto resPath = outFileName;
        resPath.replace_extension(".dat");
        saveToBinary(h_a, resPath);
    }
    outfile.close();
    isCalculating.store(false);
    progressThread.join();
}

int main(int argc, char* argv[]) {
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
    return 0;
}