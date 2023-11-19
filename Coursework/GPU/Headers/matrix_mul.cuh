#pragma once
#include "device_launch_parameters.h"

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