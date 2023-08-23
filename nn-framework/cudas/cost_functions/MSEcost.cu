#include "nn-framework/headers/cost_functions/MSEcost.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

#include <assert.h>

__global__ void meanSquareErrorCost(float* predictions, float* target, int size, float* cost)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) 
    {
        float diff = predictions[index] - target[index];
        atomicAdd(cost, diff * diff);
    }

    if (index == size - 1) 
        atomicExch(cost, *cost / (2 * size));
}

__global__ void dMeanSquareErrorCost(float* predictions, float* target, float* dY, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size)
        dY[index] = predictions[index] - target[index];
}

float MSECost::cost(Matrix predictions, Matrix target) 
{
    // Checking if dimensions are same
	assert(predictions.dims.x == target.dims.x);

	float* cost;

	cudaMallocManaged(&cost, sizeof(float));
	
    *cost = 0.0f;

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.dims.x + block_size.x - 1) / block_size.x);

	meanSquareErrorCost<<<num_of_blocks, block_size>>>(predictions.deviceData.get(), target.deviceData.get(), predictions.dims.x, cost);
	
    cuda_check(cudaDeviceSynchronize());

	float cost_value = *cost;

	cudaFree(cost);

	return cost_value;
}

Matrix MSECost::dCost(Matrix predictions, Matrix target, Matrix dY) 
{
    // Checking if dimensions are same
	assert(predictions.dims.x == target.dims.x);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.dims.x + block_size.x - 1) / block_size.x);

	dMeanSquareErrorCost<<<num_of_blocks, block_size>>>(predictions.deviceData.get(), target.deviceData.get(), dY.deviceData.get(), predictions.dims.x);
	
    cuda_check(cudaDeviceSynchronize());

	return dY;
}