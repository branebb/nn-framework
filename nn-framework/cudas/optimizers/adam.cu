#include "nn-framework/headers/optimizers/adam.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

__global__ void updateWAdam(float *dW, float *W, float *mW, float *vW, float beta1, float beta2, float epsilon, float learning_rate, int size, int t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    { 
        mW[idx] = beta1 * mW[idx] + (1 - beta1) * dW[idx];
        vW[idx] = beta2 * vW[idx] + (1 - beta2) * dW[idx] * dW[idx];

        float beta1_t = pow(beta1, t);
        float beta2_t = pow(beta2, t);

        float m_hat = mW[idx] / (1 - beta1_t);
        float v_hat = vW[idx] / (1 - beta2_t);

        W[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}

__global__ void updateBAdam(float *db, float *b, float *mb, float *vb, float beta1, float beta2, float epsilon, float learning_rate, int size, int t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        // Update m and v for biases
        mb[idx] = beta1 * mb[idx] + (1 - beta1) * db[idx];
        vb[idx] = beta2 * vb[idx] + (1 - beta2) * db[idx] * db[idx];

        // Bias correction
        float beta1_t = pow(beta1, t);
        float beta2_t = pow(beta2, t);

        // Update b using m and v
        float m_hat = mb[idx] / (1 - beta1_t);
        float v_hat = vb[idx] / (1 - beta2_t);
        b[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}

AdamOptimizer::AdamOptimizer(float beta1, float beta2, float epsilon): 
    beta1(beta1), 
    beta2(beta2), 
    epsilon(epsilon),
    mW(wDims),
    vW(wDims),
    mb(bDims),
    vb(bDims),
    t(1)
{ 
    mW.allocateMemory();   
    vW.allocateMemory();   
    mb.allocateMemory();   
    vb.allocateMemory();

    setMatricesToZero();
}

void AdamOptimizer::setMatricesToZero()
{
    for (int x = 0; x < mW.dims.x; x++)
        for (int y = 0; y < mW.dims.y; y++)
            mW[y * mW.dims.x + x] = vW[y * mW.dims.x + x] = 0.0f;
    
    for (int x = 0; x < mb.dims.x; x++)
        for (int y = 0; y < mb.dims.y; y++)
            mb[y * mb.dims.x + x] = vb[y * mb.dims.x + x] = 0.0f;          
}

void AdamOptimizer::initialize(Dimensions weightDims, Dimensions biasDims)
{
    *this = AdamOptimizer(beta1, beta2, epsilon);
}

void AdamOptimizer::updateW(Matrix &dW, Matrix &W, float learning_rate)
{
    dim3 block_size(1024);
    dim3 num_of_blocks((dW.dims.y * dW.dims.x + block_size.x - 1) / block_size.x);

    updateWAdam<<<num_of_blocks, block_size>>>(dW.deviceData.get(), W.deviceData.get(), mW.deviceData.get(), vW.deviceData.get(), beta1, beta2, epsilon, learning_rate, dW.dims.y * dW.dims.x, t);
    
    cuda_check(cudaDeviceSynchronize());
}

void AdamOptimizer::updateB(Matrix &db, Matrix &b, float learning_rate)
{ 
    dim3 block_size(1024);
    dim3 num_of_blocks((db.dims.y * db.dims.x + block_size.x - 1) / block_size.x);

    updateBAdam<<<num_of_blocks, block_size>>>(db.deviceData.get(), b.deviceData.get(), mb.deviceData.get(), vb.deviceData.get(), beta1, beta2, epsilon, learning_rate, db.dims.y * db.dims.x, t);

    cuda_check(cudaDeviceSynchronize());
}

void AdamOptimizer::updateStep(Matrix &dW, Matrix &W, Matrix &db, Matrix &b, float learning_rate)
{ 
    updateW(dW, W, learning_rate);
    updateB(db, b, learning_rate);
    increaseT();
}

void AdamOptimizer::increaseT() { t++; }