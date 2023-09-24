#include <thrust/host_vector.h>
#include <fstream>

#pragma region TXT
template <typename T>
void SaveToTxt(const thrust::host_vector<T>& v, const std::string& filename) {
	std::ofstream file(filename);
	file << v.size() << "\n";
	for(const auto& val : v) {
		file << val << "\n";
	}
	file.close();
}

template <typename T>
void ReadFromTxt(T& vector, const std::string& filename) {
	std::ifstream file(filename);
	size_t size;
	file >> size;
	vector.reserve(size);
	typename T::value_type value;
	while(file >> value) {
		vector.push_back(value);
	}
	file.close();
}
#pragma endregion

#pragma region BINARY
template <typename T>
void SaveToBinary(const thrust::host_vector<T>& v, const std::string& filename) {
	std::ofstream file(filename, std::ios::binary | std::ios::trunc);
	file.seekp(0);
	size_t size = v.size();
	file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
	file.write(reinterpret_cast<const char*>(v.data()), size * sizeof(T));
	file.close();
}


template <typename T>
void ReadFromBinary(T& vector, const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	size_t size = 0;
	file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	vector.resize(size);
	file.read(reinterpret_cast<char*>(vector.data()), size * sizeof(typename T::value_type));
	file.close();
}
#pragma endregion


