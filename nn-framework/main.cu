#include <iostream>

__global__ void kernelWithPrint() {
    printf("Thread %d in block %d\n", threadIdx.x, blockIdx.x);
}


int main()
{
    std::cout << "Radi" << std::endl;
    kernelWithPrint<<<1, 1>>>();

    return 0;
}