#include <thrust/host_vector.h>
#include <fstream>

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
#pragma endregion


