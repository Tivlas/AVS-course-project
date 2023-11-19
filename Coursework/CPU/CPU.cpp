#include <iostream>
#include <omp.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>
#include <atomic>
#include <exception>

template <typename T>
void readFromBinary(std::vector<T>& v, const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	if(!file.good()) throw std::invalid_argument("Invalid file!");
	size_t size = 0;
	file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	v.resize(size);
	file.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
	file.close();
}


void matrixMul(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result) {
	size_t size = sqrt(a.size());
	int j, k;
#pragma omp parallel for private(j, k)
	for(int i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			result[i * size + j] = 0;
			for(k = 0; k < size; k++) {
				result[i * size + j] += a[i * size + k] * b[k * size + j];
			}
		}
	}
}

void matrixAdd(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result) {
	size_t size = a.size();
	int i;
#pragma omp parallel for private(i) 
	for(i = 0; i < size; i++) {
		result[i] = a[i] + b[i];
	}
}

void print(const std::vector<int> m) {
	for(auto i : m) {
		std::cout << i << ' ';
	}
	std::cout << '\n';
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

auto measureTime(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result,
				  std::function<void(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result)> matrix_func) {
	std::atomic<bool> isCalculating;
	isCalculating.store(true);
	std::thread progressThread(printProgressIndicator, std::ref(isCalculating));
	auto start = std::chrono::high_resolution_clock::now();
	matrix_func(a, b, result);
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
	std::vector<int> a;
	try {
		readFromBinary(a, argv[1]);
	}
	catch(const std::exception& e) {
		std::cout << e.what();
		return 0;
	}
	std::vector<int> b = a;
	std::vector<int> result(a.size(), 0);
	omp_set_num_threads(omp_get_max_threads());

	std::string op = argv[2];
	std::function<void(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result)> matrix_func;
	matrix_func = op == "m" ? matrixMul : matrixAdd;
	auto time = measureTime(a, b, result, matrix_func);
	std::cout << time;
}

