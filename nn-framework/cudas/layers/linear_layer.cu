#include "nn-framework/headers/layers/linear_layer.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

#include <string>
#include <random>
#include <assert.h>

__global__ void linearLayerForward(float *W, float *A, float *Z, float *b, int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if (row < Z_y_dim && col < Z_x_dim)
    {
        for (int i = 0; i < W_x_dim; i++)
            Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];

        Z[row * Z_x_dim + col] = Z_value + b[row];
    }
}

__global__ void linearLayerBackprop(float *W, float *dZ, float *dA, int W_x_dim, int W_y_dim, int dZ_x_dim, int dZ_y_dim)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // W.T
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_x_dim;

    float dA_value = 0.0f;

    if (row < dA_y_dim && col < dA_x_dim)
    {
        for (int i = 0; i < W_y_dim; i++)
            dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];

        dA[row * dA_x_dim + col] = dA_value;
    }
}

__global__ void linearLayerUpdateWeights(float *dZ, float *A, float *W, int dZ_x_dim, int dZ_y_dim, int A_x_dim, int A_y_dim, float learning_rate)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // A.T
    int W_x_dim = A_y_dim;
    int W_y_dim = dZ_y_dim;

    float dW_value = 0.0f;

    if (row < W_y_dim && col < W_x_dim)
    {
        for (int i = 0; i < dZ_x_dim; i++)
            dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];

        W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
    }
}

__global__ void linearLayerUpdateBias(float *dZ, float *b, int dZ_x_dim, int dZ_y_dim, int b_x_dim, float learning_rate)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dZ_x_dim * dZ_y_dim)
    {
        int dZ_x = index % dZ_x_dim;
        int dZ_y = index / dZ_x_dim;

        atomicAdd(&b[dZ_y], -learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
    }
}

LinearLayer::LinearLayer(std::string name, Dimensions W_dims) : W(W_dims), b(W_dims.y, 1)
{
    this->name = name;
    b.allocateMemory();
    W.allocateMemory();
    initializeBiasWithZeros();
    initializeWeightsRandomly();
    // cuda_check(cudaDeviceSynchronize());
}

LinearLayer::~LinearLayer() {}

void LinearLayer::initializeWeightsRandomly()
{
    std::mt19937 rng(std::random_device{}());
     std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (int x = 0; x < W.dims.x; x++)
        for (int y = 0; y < W.dims.y; y++)
            W[y * W.dims.x + x] = dist(rng);

    W.copyHostToDevice();
}

void LinearLayer::initializeBiasWithZeros()
{
    for (int x = 0; x < b.dims.x; x++)
        b[x] = 0;

    b.copyHostToDevice();
}

Matrix &LinearLayer::forward(Matrix &A)
{
    // Check if W and A are chained
    assert(W.dims.x == A.dims.y);

    this->A = A;

    Dimensions Z_dims(A.dims.x, W.dims.y);

    Z.allocateMemoryIfNotAllocated(Z_dims);

    computeStoreLayerOutput(A);

    // cuda_check(cudaDeviceSynchronize());

    return Z;
}

void LinearLayer::computeStoreLayerOutput(Matrix &A)
{
    dim3 block_size(8, 8);
    dim3 num_of_blocks((Z.dims.x + block_size.x - 1) / block_size.x, (Z.dims.y + block_size.y - 1) / block_size.y);

    linearLayerForward<<<num_of_blocks, block_size>>>(W.deviceData.get(), A.deviceData.get(), Z.deviceData.get(), b.deviceData.get(), W.dims.x, W.dims.y, A.dims.x, A.dims.y);
}

Matrix &LinearLayer::backprop(Matrix &dZ, float learning_rate)
{
    dA.allocateMemoryIfNotAllocated(A.dims);

    computeStoreBackpropError(dZ);
    // cuda_check(cudaDeviceSynchronize());

    updateBias(dZ, learning_rate);
    // cuda_check(cudaDeviceSynchronize());

    updateWeights(dZ, learning_rate);
    // cuda_check(cudaDeviceSynchronize());

    return dA;
}

void LinearLayer::computeStoreBackpropError(Matrix &dZ)
{
    dim3 block_size(8, 8);
    dim3 num_of_blocks((A.dims.x + block_size.x - 1) / block_size.x, (A.dims.y + block_size.y - 1) / block_size.y);

    linearLayerBackprop<<<num_of_blocks, block_size>>>(W.deviceData.get(), dZ.deviceData.get(), dA.deviceData.get(), W.dims.x, W.dims.y, dZ.dims.x, dZ.dims.y);
}

void LinearLayer::updateWeights(Matrix &dZ, float learning_rate)
{
    dim3 block_size(8, 8);
    dim3 num_of_blocks((W.dims.x + block_size.x - 1) / block_size.x, (W.dims.y + block_size.y - 1) / block_size.y);

    linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(dZ.deviceData.get(), A.deviceData.get(), W.deviceData.get(), dZ.dims.x, dZ.dims.y, A.dims.x, A.dims.y, learning_rate);
}

void LinearLayer::updateBias(Matrix &dZ, float learning_rate)
{
    dim3 block_size(256);
    dim3 num_of_blocks((dZ.dims.y * dZ.dims.x + block_size.x - 1) / block_size.x);

    linearLayerUpdateBias<<<num_of_blocks, block_size>>>(dZ.deviceData.get(), b.deviceData.get(),dZ.dims.x, dZ.dims.y, b.dims.x, learning_rate);
}

int LinearLayer::getXDim() const { return W.dims.x; }

int LinearLayer::getYDim() const { return W.dims.y; }

Matrix LinearLayer::getWeightsMatrix() const { return W; }

Matrix LinearLayer::getBiasVector() const { return b; }