#include "nn-framework/headers/cost_functions/CrossEntropy.hh"
#include "nn-framework/utils/error_check_cuda.hpp"
#include "nn-framework/headers/regularization/regularization.hh"

#include <assert.h>

__global__ void crossEntropyCost(float* predictions, float* target, int features, int data, float* cost)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < data) 
    {
        float col_cost = 0.0f;
        for (int row = 0; row < features; row++)
        {
            float epsilon = 1e-7f; // same in keras
            col_cost += (-target[row * data + col] * log(predictions[row * data + col] + epsilon));
        }

        atomicAdd(cost, col_cost);
    }

    if (col == data - 1) 
        atomicExch(cost, *cost / data);
}

__global__ void dCrossEntropyCost(float* predictions, float* target, float* dY, int features, int data)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < data) 
    {
        for (int row = 0; row < features; row++)
        {
            int index = row * data + col;
            dY[index] = (predictions[index] - target[index]) / data;
        }
    }
}

float CrossEntropyCost::cost(Matrix& predictions, Matrix& target, Matrix& W) 
{
    assert(predictions.dims.x == target.dims.x && predictions.dims.y == target.dims.y);

    float* cost;

    cudaMallocManaged(&cost, sizeof(float));

    *cost = 0.0f;

    dim3 block_size(1024);
    dim3 num_of_blocks((predictions.dims.x + block_size.x - 1) / block_size.x);

    crossEntropyCost<<<num_of_blocks, block_size>>>(predictions.deviceData.get(), target.deviceData.get(), predictions.dims.y, predictions.dims.x, cost);

    cuda_check(cudaDeviceSynchronize());

    float cost_value = *cost;

    cudaFree(cost);

    if (regularization != nullptr)
    {
        float regTerm = 0.0f;

        regTerm = regularization->costRegularization(W);

        cost_value += regTerm;
    }

    return cost_value;
}

Matrix CrossEntropyCost::dCost(Matrix& predictions, Matrix& target, Matrix& dY)
{
    assert(predictions.dims.x == target.dims.x && predictions.dims.y == target.dims.y);

    dim3 block_size(1024);
    dim3 num_of_blocks((predictions.dims.x + block_size.x - 1) / block_size.x);

    dCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.deviceData.get(), target.deviceData.get(), dY.deviceData.get(), predictions.dims.y, predictions.dims.x);

    cuda_check(cudaDeviceSynchronize());

    return dY;
}

CrossEntropyCost::CrossEntropyCost(Regularization* regularization): regularization(regularization) { }