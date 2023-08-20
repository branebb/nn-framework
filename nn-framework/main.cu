// Standard library
#include <iostream>
#include <stdio.h>

// Nvidia
#include <driver_types.h>
#include <cuda_runtime.h>

// Project headers
#include "nn-framework/utils/error_check_cuda.hpp"

__global__ void myKernelTest(void)
{
    int idx = threadIdx.x;

    printf("Thread ID: %d\n", idx);

    return;
}

int main()
{
    myKernelTest<<<1, 128>>>();
    //1024 max

    cuda_check(cudaDeviceSynchronize());

    printf("CUDA radi!\n");
    
    return 0;
}