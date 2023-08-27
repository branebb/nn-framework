#include "nn-framework/headers/cost_functions/MSEcost.hh"
#include "nn-framework/utils/error_check_cuda.hpp"
#include "nn-framework/headers/regularization/regularization.hh"

#include <assert.h>

__global__ void meanSquareErrorCost(float* predictions, float* target, int features, int data, float* cost)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < data) 
    {
        float col_cost = 0.0f;
        for (int row = 0; row < features; row++)
        {
            float diff = predictions[row * data + col] - target[row * data + col];
            col_cost += diff * diff;
        }

        atomicAdd(cost, col_cost);
    }

    if (col == data - 1) 
        atomicExch(cost, *cost / (2 * data));
}

__global__ void dMeanSquareErrorCost(float* predictions, float* target, float* dY, int features, int data)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < data) 
    {
        for (int row = 0; row < features; row++)
        {
            int index = row * data + col;
            dY[index] = ((predictions[index] - target[index]) / data);
        }
    }
}

float MSECost::cost(Matrix& predictions, Matrix& target, Matrix& W) 
{
    // Checking if dimensions are same
    // X number of data
    // Y number of features 
	assert(predictions.dims.x == target.dims.x && predictions.dims.y == target.dims.y);

	float* cost;

	cudaMallocManaged(&cost, sizeof(float));
	
    *cost = 0.0f;

	dim3 block_size(1024);
	dim3 num_of_blocks((predictions.dims.x + block_size.x - 1) / block_size.x);

	meanSquareErrorCost<<<num_of_blocks, block_size>>>(predictions.deviceData.get(), target.deviceData.get(), predictions.dims.y, predictions.dims.x, cost);
	
	cuda_check(cudaDeviceSynchronize());

    float cost_value = *cost;

	cudaFree(cost);

    if(regularization != nullptr)
    {
        float regTerm  = 0.0f;

        regTerm = regularization->costRegularization(W);

        cost_value += regTerm;
    }

	return cost_value;
}

Matrix MSECost::dCost(Matrix& predictions, Matrix& target, Matrix& dY)
{
    // Checking if dimensions are same
    // X number of data
    // Y number of features 
	assert(predictions.dims.x == target.dims.x && predictions.dims.y == target.dims.y);

	dim3 block_size(1024);
	dim3 num_of_blocks((predictions.dims.x + block_size.x - 1) / block_size.x);

	dMeanSquareErrorCost<<<num_of_blocks, block_size>>>(predictions.deviceData.get(), target.deviceData.get(), dY.deviceData.get(), predictions.dims.y, predictions.dims.x);
	
    cuda_check(cudaDeviceSynchronize());

	return dY;
}

MSECost::MSECost(Regularization* regularization): regularization(regularization) { }