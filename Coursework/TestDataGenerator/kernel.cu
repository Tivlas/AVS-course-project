﻿#include "cuda_runtime.h"
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "Headers/file.cuh"

struct GenRandInt {
	__device__
		int operator () (int idx) {
		thrust::default_random_engine rand_eng;
		thrust::uniform_int_distribution<int> uni_dist;
		rand_eng.discard(idx);
		return uni_dist(rand_eng);
	}
};

int main(int argc, char* argv[]) {
	size_t N = 3;
	if(argc != 3) {
		std::cout << "Requires 2 args\n";
		return 0;
	}
	else {
		N = std::atoi(argv[2]);
		if(N < 2 || N > 30000) {
			std::cout << "Invalid size parameter (2 <= size <= 30000).\n";
			return 0;
		}
	}
	std::string fileName = argv[1];
	if(!std::ofstream(fileName).good()) {
		std::cout << "Invalid file";
		return 0;
	}
	const size_t size = N * N;
	thrust::device_vector<int> a(size);
	thrust::transform(thrust::make_counting_iterator(0ULL), thrust::make_counting_iterator(size), a.begin(), GenRandInt());
	thrust::host_vector<int> a_copy = a;
	try {
		SaveToBinary(a_copy, fileName);
	}
	catch(const std::exception& e) {
		std::cout << e.what();
	}
	return 0;
}