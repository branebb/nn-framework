#include <iostream>

__global__ void myKernelTest(void){

}

int main()
{
    myKernelTest<<<1, 1>>>();
    printf("Radi bkt mazo\n");
    return 0;
}