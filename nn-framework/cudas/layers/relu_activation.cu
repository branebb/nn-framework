#include "nn-framework/headers/layers/relu_activation.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

#include <math.h>

__global__ void reluActivationForward(float* Z, float *A, int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) 
    {
		A[index] = fmaxf(Z[index], 0);
	}
}

__global__ void reluActivationBackprop(float* Z, float *dA, float* dZ, int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) 
    {
		if (Z[index] > 0) 
			dZ[index] = dA[index];
		else
			dZ[index] = 0;
	}
}

ReLUActivation::ReLUActivation(std::string name) { this->name = name; }

ReLUActivation::~ReLUActivation() { }

Matrix& ReLUActivation::forward(Matrix& Z) 
{
	this->Z = Z;

	A.allocateMemoryIfNotAllocated(Z.dims);

	dim3 block_size(1024);
	dim3 num_of_blocks((Z.dims.y * Z.dims.x + block_size.x - 1) / block_size.x);

	reluActivationForward<<<num_of_blocks, block_size>>>(Z.deviceData.get(), A.deviceData.get(), Z.dims.x, Z.dims.y);
	
    cuda_check(cudaDeviceSynchronize());

	return A;
}

Matrix& ReLUActivation::backprop(Matrix& dA, float learning_rate) 
{
	dZ.allocateMemoryIfNotAllocated(Z.dims);

	dim3 block_size(1024);
	dim3 num_of_blocks((Z.dims.y * Z.dims.x + block_size.x - 1) / block_size.x);

	reluActivationBackprop<<<num_of_blocks, block_size>>>(Z.deviceData.get(), dA.deviceData.get(), dZ.deviceData.get(), Z.dims.x, Z.dims.y);
	
    cuda_check(cudaDeviceSynchronize());

	return dZ;
}