#include "nn-framework/headers/regularization/L2.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

__global__ void applyRegularization(float *dW, float *W, float lambda, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        dW[idx] += lambda * W[idx];
    }
}

__global__ void calculateRegularizationTerm(float* W, float lambda, int size, float* regularizationTerm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float reg_cost = 0.0f;

    if (idx < size)
    {
        float w_i = W[idx];
        reg_cost += w_i * w_i;
    }

    atomicAdd(regularizationTerm, reg_cost);

}

void L2::gradientRegularization(Matrix& W, Matrix &dW, int size)
{
    dim3 block_size(1024);
	dim3 num_of_blocks((W.dims.x + block_size.x - 1) / block_size.x);

    applyRegularization<<<num_of_blocks, block_size>>>(dW.deviceData.get(), W.deviceData.get(), lambda, dW.dims.x * dW.dims.y);
    
    // cuda_check(cudaDeviceSynchronize());
}

float L2::costRegularization(Matrix &W)
{
    float* regularizationTerm;

	cudaMallocManaged(&regularizationTerm, sizeof(float));
	
    *regularizationTerm = 0.0f;

	dim3 block_size(1024);
	dim3 num_of_blocks((W.dims.x + block_size.x - 1) / block_size.x);

    calculateRegularizationTerm<<<num_of_blocks, block_size>>>(W.deviceData.get(), lambda, W.dims.x * W.dims.y, regularizationTerm);
    
    cuda_check(cudaDeviceSynchronize());

    float regularizationTermValue = *regularizationTerm;
    
	cudaFree(regularizationTerm);

    regularizationTermValue *= (lambda / (2 * W.dims.x));

	return regularizationTermValue;
}

L2::L2(float lambda): lambda(lambda) { }