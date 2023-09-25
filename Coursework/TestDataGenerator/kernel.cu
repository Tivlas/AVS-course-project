#include "cuda_runtime.h"
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

const std::string base_dir = "D:\\University\\Courseworks\\AVS\\AVS-course-project\\Coursework\\TestDataGenerator\\data\\";


int main() {
	const size_t N = 30000;
	const size_t size = N * N;
	thrust::device_vector<int> a(size);
	thrust::transform(
   thrust::make_counting_iterator(0ULL),
   thrust::make_counting_iterator(size),
   a.begin(),
   GenRandInt());
	thrust::host_vector<int> a_copy = a;
	auto filename = base_dir + "3e1_int.dat";
	SaveToBinary(a_copy, filename);
	return 0;
}