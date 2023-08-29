#include "nn-framework/headers/layers/tanh_activation.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

__device__ float myTanh(float x)
{
    return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

__global__ void tanhActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) 
        A[index] = myTanh(Z[index]);
}

__global__ void tanhActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim)
        dZ[index] = dA[index] * (1 - (myTanh(Z[index]) * myTanh(Z[index])));
}

TanhActivation::TanhActivation(std::string name) { this->name = name; }

TanhActivation::~TanhActivation() { }

Matrix& TanhActivation::forward(Matrix& Z) 
{
    this->Z = Z;

    A.allocateMemoryIfNotAllocated(Z.dims);

    dim3 block_size(1024);
    dim3 num_of_blocks((Z.dims.y * Z.dims.x + block_size.x - 1) / block_size.x);

    tanhActivationForward<<<num_of_blocks, block_size>>>(Z.deviceData.get(), A.deviceData.get(), Z.dims.x, Z.dims.y);
    
    // cuda_check(cudaDeviceSynchronize());

    return A;
}

Matrix& TanhActivation::backprop(Matrix& dA, float learning_rate) 
{
    dZ.allocateMemoryIfNotAllocated(Z.dims);

    dim3 block_size(1024);
    dim3 num_of_blocks((Z.dims.y * Z.dims.x + block_size.x - 1) / block_size.x);
    
    tanhActivationBackprop<<<num_of_blocks, block_size>>>(Z.deviceData.get(), dA.deviceData.get(), dZ.deviceData.get(), Z.dims.x, Z.dims.y);
    
    // cuda_check(cudaDeviceSynchronize());

    return dZ;
}