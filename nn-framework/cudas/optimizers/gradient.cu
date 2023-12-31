#include "nn-framework/headers/optimizers/gradient.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

__global__ void updateWcuda(float* dW, float* W, float learning_rate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        W[idx] -= learning_rate * dW[idx];
    }
}

__global__ void updateBcuda(float* db, float* b, float learning_rate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        b[idx] -= learning_rate * db[idx];
    }
}

void Gradient::updateW(Matrix &dW, Matrix &W, float learning_rate)
{
    dim3 block_size(1024);
    dim3 num_of_blocks((dW.dims.y * dW.dims.x + block_size.x - 1) / block_size.x);

    updateWcuda<<<num_of_blocks, block_size>>>(dW.deviceData.get(), W.deviceData.get(), learning_rate, dW.dims.y * dW.dims.x);
}

void Gradient::updateB(Matrix &db, Matrix &b, float learning_rate)
{

    dim3 block_size(1024);
    dim3 num_of_blocks((db.dims.y * db.dims.x + block_size.x - 1) / block_size.x);

    updateWcuda<<<num_of_blocks, block_size>>>(db.deviceData.get(), b.deviceData.get(), learning_rate, db.dims.y * db.dims.x);

    // cuda_check(cudaDeviceSynchronize());
}

void Gradient::updateStep(Matrix &dW, Matrix &W, Matrix &db, Matrix &b, float learning_rate)
{
    updateW(dW, W, learning_rate);
    updateB(db, b, learning_rate);
}