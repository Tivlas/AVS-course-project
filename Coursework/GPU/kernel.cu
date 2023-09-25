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

const std::string base_dir = "D:\\University\\Courseworks\\AVS\\AVS-course-project\\Coursework\\TestDataGenerator\\data\\";

template <typename T>
void ReadFromBinary(thrust::host_vector<T>& v, const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	size_t size = 0;
	file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	v.resize(size);
	file.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
	file.close();
}

int main() {
	thrust::host_vector<int> h_a;
	ReadFromBinary(h_a, base_dir + "3e4_int.dat");
	thrust::host_vector<int> h_b = h_a;
	thrust::device_vector<int> d_a = h_a;
	thrust::device_vector<int> d_b = h_b;
	thrust::device_vector<int> d_c(h_a.size());
	const size_t N = sqrt(h_a.size());
	std::cout << "Elements: " << h_a.size() << '\n';
	dim3 blockSize(16, 16);
	dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

	std::pair<int, int> time_count = {0,0};

	for(int i = 10 - 1; i >= 0; i--) {
		auto start = std::chrono::high_resolution_clock::now();
		SqMatrixMulKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_c.data()), thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), N);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
		std::cout << elapsed_time.count() << '\n';
		time_count.first += elapsed_time.count();
		time_count.second++;
	}

	std::cout << "Average time of " << time_count.second << " calculations: " << static_cast<double>(time_count.first) / time_count.second;

	return 0;
}