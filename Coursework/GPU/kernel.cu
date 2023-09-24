#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include "Headers/matrix_add.cuh"
#include "Headers/matrix_mul.cuh"
#include <chrono>

void print(const thrust::host_vector<int>& m) {
	for(auto& i : m) {
		std::cout << i << ' ';
	}
	std::cout << '\n';
}

int main() {
	const size_t N = 3e4;
	const size_t size = N * N;
	std::cout << "elements: " << size << '\n';
	thrust::default_random_engine rng(2324);
	thrust::uniform_int_distribution<int> dist(1, 10);

	thrust::host_vector<int> h_a(size);
	thrust::host_vector<int> h_b(size);

	auto s = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < size; ++i) {
		h_a[i] = dist(rng);
		h_b[i] = dist(rng);
	}
	auto e = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s);
	std::cout << time.count() << "  " << "\n";


	thrust::device_vector<int> d_a = h_a;
	thrust::device_vector<int> d_b = h_b;
	thrust::device_vector<int> d_c(size);

	dim3 blockSize(16, 16);
	dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

	auto start = std::chrono::high_resolution_clock::now();
	SqMatrixAddKernel << <numBlocks, blockSize >> > (thrust::raw_pointer_cast(d_c.data()), thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), N);
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cout << elapsed_time.count() << "  ";
	return 0;
}