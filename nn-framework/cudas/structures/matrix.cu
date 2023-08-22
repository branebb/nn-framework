#include "nn-framework/headers/structures/matrix.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

Matrix::Matrix(size_t x_dim, size_t y_dim) :
	dims(x_dim, y_dim), 
    deviceData(nullptr), 
    hostData(nullptr),
	isDeviceAllocated(false), 
    isHostAllocated(false)
{ }

Matrix::Matrix(Dimensions dims) :
	Matrix(dims.x, dims.y)
{ }

void Matrix::allocateCudaMemory() 
{
	if (!isDeviceAllocated) 
    {
		float* device_memory = nullptr;
		cudaMalloc(&device_memory, dims.x * dims.y * sizeof(float));
		cuda_check(cudaDeviceSynchronize());
		deviceData = std::shared_ptr<float>(device_memory, [&](float* ptr) { cudaFree(ptr); });
		isDeviceAllocated = true;
	}
}

void Matrix::allocateHostMemory() 
{
	if (!isHostAllocated) 
    {
		hostData = std::shared_ptr<float>(new float[dims.x * dims.y], [&](float* ptr) { delete[] ptr; });
		isHostAllocated = true;
	}
}

void Matrix::allocateMemory() 
{
	allocateCudaMemory();
	allocateHostMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Dimensions dims) 
{
	if (!isDeviceAllocated && !isHostAllocated) 
    {
		this->dims = dims;
		allocateMemory();
	}
}

void Matrix::copyHostToDevice() 
{
	if (isDeviceAllocated && isHostAllocated) 
    {
		cudaMemcpy(deviceData.get(), hostData.get(), dims.x * dims.y * sizeof(float), cudaMemcpyHostToDevice);
		cuda_check(cudaDeviceSynchronize());
	}
}

void Matrix::copyDeviceToHost() 
{
	if (isDeviceAllocated && isHostAllocated) 
    {
		cudaMemcpy(hostData.get(), deviceData.get(), dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToHost);
		cuda_check(cudaDeviceSynchronize());
	}
}

float& Matrix::operator[](const int index) 
{
	return hostData.get()[index];
}

const float& Matrix::operator[](const int index) const 
{
	return hostData.get()[index];
}