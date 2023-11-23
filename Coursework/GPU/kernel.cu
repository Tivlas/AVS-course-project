#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>

template <typename T>
__global__ void matrixMulKernel(T* c, const T* a, const T* b, size_t N) {
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < N && j < N) {
		auto idx = i * N + j;
		c[idx] = 0;
		for(size_t k = 0; k < N; k++) {
			c[idx] += a[i * N + k] * b[k * N + j];
		}
	}
}

template <typename T>
__global__ void matrixAddKernel(T* c, const T* a, const T* b, size_t N) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if(i < N && j < N) {
		auto idx = i * N + j;
		c[idx] = a[idx] + b[idx];
	}
}

template <typename T>
void ReadFromBinary(thrust::host_vector<T>& v, const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	if(!file.good()) throw std::invalid_argument("Invalid file!");
	size_t size = 0;
	file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	v.resize(size);
	file.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
	file.close();
}

void printProgressIndicator(std::atomic<bool>& isCalculating) {
	const std::string indicators = "-\\|/";
	int index = 0;
	while(isCalculating) {
		std::cout << "\rCalculating... " << indicators[index++];
		index %= indicators.size();
		std::cout.flush();
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}
	std::cout << '\n';
}

template <typename T>
long long measureTime(const thrust::device_vector<T>& a, thrust::device_vector<T>& b, thrust::device_vector<T>& result, size_t N, dim3 blockSize, dim3 numBlocks, std::string op) {
	std::atomic<bool> isCalculating;
	isCalculating.store(true);
	std::thread progressThread(printProgressIndicator, std::ref(isCalculating));
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

	auto start = std::chrono::high_resolution_clock::now();
	if(op == "m") {
		matrixMulKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(result.data()), thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N);
	}
	else if(op == "a") {
		matrixAddKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(result.data()), thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N);
	}
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	isCalculating.store(false);
	progressThread.join();
	return elapsed_time.count();
}

template <typename T>
void calculate(std::string op, const std::string& fileName) {
	thrust::host_vector<T> h_a;
	try {
		ReadFromBinary<T>(h_a, fileName);
	}
	catch(const std::exception& e) {
		std::cout << e.what();
		return;
	}
	thrust::device_vector<T> d_a = h_a;
	thrust::device_vector<T> d_b = h_a;
	thrust::device_vector<T> d_c(h_a.size());
	const size_t N = sqrt(h_a.size());
	dim3 blockSize(16, 16);
	dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
	auto time = measureTime<T>(d_a, d_b, d_c, N, blockSize, numBlocks, op);
	std::cout << time << '\n';
}

int main(int argc, char* argv[]) {
	if(argc != 4) {
		std::cout << "Requires 3 args\n";
		return 0;
	}
	else {
		std::string op = argv[2];
		if(op != "a" && op != "m") {
			std::cout << "Usage: <file_path> <function (m or a)>\n";
			return 0;
		}
		std::string dataType = argv[3];
		if(dataType != "i" && dataType != "f") {
			std::cout << "Usage: <file_path> <function (m or a)> <dataType (i or f)\n";
			return 0;
		}
	}
	std::string fileName = argv[1];
	std::string op = argv[2];
	std::string dataType = argv[3];
	if(dataType == "i") {
		calculate<int>(op, fileName);
	}
	else if(dataType == "f") {
		calculate<float>(op, fileName);
	}
	return 0;
}