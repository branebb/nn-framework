#pragma once

#include "dimensions.hh"
#include <memory>

class Matrix 
{
private:

	bool isDeviceAllocated;
	bool isHostAllocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:
	Dimensions dims;

	std::shared_ptr<float> deviceData;
	std::shared_ptr<float> hostData;

	Matrix(size_t x_dim = 1, size_t y_dim = 1);
	Matrix(Dimensions dims);

	void allocateMemory();
	void allocateMemoryIfNotAllocated(Dimensions dims);

	void copyHostToDevice();
	void copyDeviceToHost();

	float& operator[](const int index);
	const float& operator[](const int index) const;

	void oneHotEncoding();

	bool deviceAllocation();
};