#include "cuda_runtime.h"
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <type_traits>

template <typename T>
void SaveToBinary(const thrust::host_vector<T>& v, const std::string& filename) {
	std::ofstream file(filename, std::ios::binary | std::ios::trunc | std::ios::out);
	file.seekp(0);
	size_t size = v.size();
	file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(v.data()), size * sizeof(T));
	file.close();
}

template <typename T>
using DistributionType = typename std::conditional<std::is_integral_v<T>, thrust::uniform_int_distribution<T>, thrust::uniform_real_distribution<T>>::type;

template <typename T>
struct GenRand {
	thrust::default_random_engine rand_eng;
	DistributionType<T> uni_dist;

	__device__
		GenRand(): rand_eng(thrust::default_random_engine{static_cast<unsigned int>(clock())}) {
	}

	__device__
		T operator () (int idx) {
		rand_eng.discard(idx);
		return uni_dist(rand_eng);
	}
};

template <typename T>
void generate(const std::string& fileName, size_t size) {
	thrust::device_vector<T> a(size);
	thrust::transform(thrust::make_counting_iterator(0ULL), thrust::make_counting_iterator(size), a.begin(), GenRand<T>());
	thrust::host_vector<T> a_copy = a;
	SaveToBinary<T>(a_copy, fileName);
}

int main(int argc, char* argv[]) {
	size_t N = 3;
	if(argc != 4) {
		std::cout << "Requires 3 args\n";
		return 0;
	}
	else {
		N = std::atoi(argv[2]);
		if(N < 2 || N > 30000) {
			std::cout << "Invalid size parameter (2 <= size <= 30000).\n";
			return 0;
		}
		std::string dataType = argv[3];
		if(dataType != "f" && dataType != "i") {
			std::cout << "Invalid data type (f or i).\n";
			return 0;
		}
	}
	std::string fileName = argv[1];
	std::string dataType = argv[3];
	const size_t size = N * N;
	if(dataType == "f") {
		generate<float>(fileName, size);
	}
	else if (dataType == "i") {
		generate<int>(fileName, size);
	}
	else {
		std::cout << "Nothing was generated! Invalid data type.";
	}
	return 0;
}