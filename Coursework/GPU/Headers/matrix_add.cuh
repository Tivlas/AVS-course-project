#pragma once
#include "device_launch_parameters.h"

template <typename T>
__global__ void matrixAddKernel(T* c, const T* a, const T* b, size_t N) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t j = threadIdx.y + blockIdx.y * blockDim.y;
	if(i < N && j < N) {
		auto idx = i * N + j;
		c[idx] = a[idx] + b[idx];
	}
}