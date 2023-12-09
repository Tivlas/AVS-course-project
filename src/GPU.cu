#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T>
__global__ void matrixMulKernel(T* c, const T* a, const T* b, size_t N) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        auto idx = i * N + j;
        for (size_t k = 0; k < N; k++) {
            c[idx] += a[i * N + k] * b[k * N + j];
        }
    }
}

template <typename T>
__global__ void matrixMulKernelNotParallel(T* c, const T* a, const T* b,
                                           size_t N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

#define BLOCK_DIM 16
template <typename T>
__global__ void matrixMulKernelSharedMemory(T* c, const T* a, const T* b,
                                            const size_t N) {
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int i = blockY * blockDim.y + threadY;
    int j = blockX * blockDim.x + threadX;
    __shared__ T aShared[BLOCK_DIM][BLOCK_DIM];
    __shared__ T bShared[BLOCK_DIM][BLOCK_DIM];
    T sum = 0;
    for (int part = 0; part < ceil(static_cast<T>(N) / BLOCK_DIM); ++part) {
        int iRow = i;
        int iCol = part * BLOCK_DIM + threadX;
        int jCol = j;
        int jRow = part * BLOCK_DIM + threadY;

        aShared[threadY][threadX] =
            (iRow < N && iCol < N) ? a[iRow * N + iCol] : 0;
        bShared[threadY][threadX] =
            (jRow < N && jCol < N) ? b[jRow * N + jCol] : 0;

        __syncthreads();
        for (int idx = 0; idx < BLOCK_DIM; ++idx) {
            sum += aShared[threadY][idx] * bShared[idx][threadX];
        }
        __syncthreads();
    }
    if (i < N && j < N) {
        c[i * N + j] = sum;
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
                      const std::string& kernelType) {
    dim3 blockSize;
    dim3 numBlocks;
    if (kernelType == "ps" || kernelType == "p") {
        blockSize = dim3(BLOCK_DIM, BLOCK_DIM);
        numBlocks = dim3((N + blockSize.x - 1) / blockSize.x,
                         (N + blockSize.y - 1) / blockSize.y);
    } else {
        blockSize = dim3(1, 1);
        numBlocks = dim3(1, 1);
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto rawResult = thrust::raw_pointer_cast(result.data());
    auto rawA = thrust::raw_pointer_cast(a.data());
    auto rawB = thrust::raw_pointer_cast(b.data());
    if (kernelType == "p") {
        matrixMulKernel<<<numBlocks, blockSize>>>(rawResult, rawA, rawB, N);
    } else if (kernelType == "np") {
        matrixMulKernelNotParallel<<<numBlocks, blockSize>>>(rawResult, rawA,
                                                             rawB, N);
    } else if (kernelType == "ps") {
        matrixMulKernelSharedMemory<<<numBlocks, blockSize>>>(rawResult, rawA,
                                                              rawB, N);
    } else {
        throw std::invalid_argument("Invalid kernelType argument!");
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return elapsed_time.count();
}

template <typename T>
void calculate(const std::vector<std::filesystem::path>& filePaths,
               const std::filesystem::path& outFileName,
               const std::string& kernelType) {
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
        auto time = measureTime<T>(d_a, d_b, d_c, N, kernelType);
        outfile << std::to_string(time) + "    " +
                       filePath.filename().string() + "\n";
    }
    outfile.close();
    isCalculating.store(false);
    progressThread.join();
}

int main(int argc, char* argv[]) {
    std::filesystem::path outFileName = std::filesystem::absolute(argv[1]);
    std::string dataDir = argv[2];
    std::string kernelType = argv[3];
    if (kernelType != "p" && kernelType != "ps" && kernelType != "np") {
        std::cout << "3rd argument must be equal to p (parallel) or ps "
                     "(parallel with shared memory) or np (not parallel)!";
        return 0;
    }
    std::filesystem::directory_iterator iterator(dataDir);
    std::vector<std::filesystem::path> fileNames;
    for (const auto& entry : iterator) {
        if (entry.is_regular_file()) {
            std::filesystem::path filePath =
                std::filesystem::absolute(entry.path()).lexically_normal();
            fileNames.push_back(filePath);
        }
    }
    calculate<float>(fileNames, outFileName, kernelType);
    return 0;
}