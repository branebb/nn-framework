#include "nn-framework/headers/layers/softmax_activation.hh"
#include "nn-framework/utils/error_check_cuda.hpp"


__global__ void softmaxActivationForward(float *Z, float *A, int Z_x_dim, int Z_y_dim)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < Z_x_dim)
    {
        float exp_sum = 0.0f;
        for (int i = 0; i < Z_y_dim; i++)
            exp_sum += exp(Z[i * Z_x_dim + col]);

        for (int i = 0; i < Z_y_dim; i++)
            A[i * Z_x_dim + col] = exp(Z[i * Z_x_dim + col]) / exp_sum;
    }
}

SoftmaxActivation::SoftmaxActivation(std::string name) { this->name = name; }

SoftmaxActivation::~SoftmaxActivation() { }

Matrix &SoftmaxActivation::forward(Matrix &Z)
{
    this->Z = Z;

    A.allocateMemoryIfNotAllocated(Z.dims);

    dim3 block_size(8, 8);
    dim3 num_of_blocks((Z.dims.x + block_size.x - 1) / block_size.x, (Z.dims.y + block_size.y - 1) / block_size.y);

    softmaxActivationForward<<<num_of_blocks, block_size>>>(Z.deviceData.get(), A.deviceData.get(), Z.dims.x, Z.dims.y);

    cuda_check(cudaDeviceSynchronize());

    return A;
}

Matrix& SoftmaxActivation::backprop(Matrix& dA, float learning_rate)
{
    return dA;
}