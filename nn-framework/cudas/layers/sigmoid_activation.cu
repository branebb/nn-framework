#include "nn-framework/headers/layers/sigmoid_activation.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

__device__ float sigmoid(float x)
{
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) 
		A[index] = sigmoid(Z[index]);

}

__global__ void sigmoidActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) 
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim)
		dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
}

SigmoidActivation::SigmoidActivation(std::string name) { this->name = name; }

SigmoidActivation::~SigmoidActivation(){ }

Matrix& SigmoidActivation::forward(Matrix& Z) 
{
	this->Z = Z;

	A.allocateMemoryIfNotAllocated(Z.dims);

	dim3 block_size(1024);
	dim3 num_of_blocks((Z.dims.y * Z.dims.x + block_size.x - 1) / block_size.x);

	sigmoidActivationForward<<<num_of_blocks, block_size>>>(Z.deviceData.get(), A.deviceData.get(), Z.dims.x, Z.dims.y);
	
    // cuda_check(cudaDeviceSynchronize());

	return A;
}

Matrix& SigmoidActivation::backprop(Matrix& dA, float learning_rate) 
{
	dZ.allocateMemoryIfNotAllocated(Z.dims);

	dim3 block_size(1024);
	dim3 num_of_blocks((Z.dims.y * Z.dims.x + block_size.x - 1) / block_size.x);
	
    sigmoidActivationBackprop<<<num_of_blocks, block_size>>>(Z.deviceData.get(), dA.deviceData.get(), dZ.deviceData.get(), Z.dims.x, Z.dims.y);
	
    // cuda_check(cudaDeviceSynchronize());

	return dZ;
}