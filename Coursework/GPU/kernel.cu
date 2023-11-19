#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include "Headers/matrix_add.cuh"
#include "Headers/matrix_mul.cuh"
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>
#include <functional>

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
auto measureTime(const thrust::device_vector<T>& a, thrust::device_vector<T>& b, thrust::device_vector<T>& result, size_t N, dim3 blockSize, dim3 numBlocks, std::string op) {
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

int main(int argc, char* argv[]) {
	if(argc != 3) {
		std::cout << "Requires 2 args\n";
		return 0;
	}
	else {
		std::string op = argv[2];
		if(op != "a" && op != "m") {
			std::cout << "Usage: <file_path> <function (m or a)>\n";
			return 0;
		}
	}
	std::string fileName = argv[1];
	thrust::host_vector<int> h_a;
	try {
		ReadFromBinary(h_a, fileName);
	}
	catch(const std::exception& e) {
		std::cout << e.what();
		return 0;
	}
	thrust::host_vector<int> h_b = h_a;
	thrust::device_vector<int> d_a = h_a;
	thrust::device_vector<int> d_b = h_b;
	thrust::device_vector<int> d_c(h_a.size());
	const size_t N = sqrt(h_a.size());
	dim3 blockSize(16, 16);
	dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

	auto time = measureTime(d_a, d_b, d_c, N, blockSize, numBlocks, argv[2]);
	std::cout << time;
	return 0;
}