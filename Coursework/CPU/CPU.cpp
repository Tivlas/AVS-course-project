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

template <typename T>
void matrixMul(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
	size_t size = sqrt(a.size());
	int j, k, i;
#pragma omp parallel for private(j, k)
	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			result[i * size + j] = 0;
			for(k = 0; k < size; k++) {
				result[i * size + j] += a[i * size + k] * b[k * size + j];
			}
		}
	}
}

template <typename T>
void matrixAdd(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result) {
	size_t size = a.size();
	int i;
#pragma omp parallel for private(i) 
	for(i = 0; i < size; i++) {
		result[i] = a[i] + b[i];
	}
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
auto measureTime(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result,
				  std::function<void(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result)> matrix_func) {
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

template <typename T>
void calculate(std::string op, const std::string& fileName) {
	std::vector<T> a;
	try {
		readFromBinary(a, fileName);
	}
	catch(const std::exception& e) {
		std::cout << e.what();
		return;
	}
	std::vector<T> b = a;
	std::vector<T> result(a.size(), 0);
	std::function<void(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& result)> matrix_func;
	matrix_func = op == "m" ? matrixMul<T> : matrixAdd<T>;
	auto time = measureTime<T>(a, b, result, matrix_func);
	std::cout << time;
}

int main(int argc, char* argv[]) {
	if(argc != 4) {
		std::cout << "Requires 3 args\n";
		return 0;
	}
	else {
		std::string op = argv[2];
		if(op != "a" && op != "m") {
			std::cout << "Usage: <file_path> <function (m or a)>\n";
			return 0;
		}
		std::string dataType = argv[3];
		if(dataType != "i" && dataType != "f") {
			std::cout << "Usage: <file_path> <function (m or a)> <dataType (i or f)\n";
			return 0;
		}
	}
	omp_set_num_threads(omp_get_max_threads());
	std::string op = argv[2];
	std::string dataType = argv[3];
	std::string fileName = argv[1];
	if(dataType == "i") {
		calculate<int>(op, fileName);
	}
	else if(dataType == "f") {
		calculate<float>(op, fileName);
	}
}

