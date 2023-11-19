﻿#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>
#include <functional>

template <typename T>
__global__ void matrixMulKernel(T* c, const T* a, const T* b, size_t N) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < N && j < N) {
		auto idx = i * N + j;
		T sum = 0;
		for(size_t k = 0; k < N; k++) {
			sum += a[i * N + k] * b[k * N + j];
		}
		c[idx] = sum;
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
auto measureTime(const thrust::device_vector<T>& a, thrust::device_vector<T>& b, thrust::device_vector<T>& result, size_t N, std::string op) {
	std::atomic<bool> isCalculating;
	isCalculating.store(true);
	std::thread progressThread(printProgressIndicator, std::ref(isCalculating));
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

	auto start = std::chrono::high_resolution_clock::now();
	if(op == "m") {
		matrixMulKernel << <1, 1 >> > (thrust::raw_pointer_cast(result.data()), thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N);
	}
	else if(op == "a") {
		matrixAddKernel << <1, 1 >> > (thrust::raw_pointer_cast(result.data()), thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), N);
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
	auto time = measureTime(d_a, d_b, d_c, N, argv[2]);
	std::cout << time;
	return 0;
}