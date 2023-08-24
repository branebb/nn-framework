// Standard library
#include <iostream>
#include <stdio.h>
#include <time.h>

// Nvidia
#include <driver_types.h>
#include <cuda_runtime.h>

// Project headers
#include "nn-framework/utils/error_check_cuda.hpp"
#include "nn-framework/headers/structures/neural_network.hh"
#include "nn-framework/headers/cost_functions/MSEcost.hh"
#include "nn-framework/headers/layers/linear_layer.hh"
#include "nn-framework/headers/layers/tanh_activation.hh"
#include "nn-framework/headers/layers/relu_activation.hh"
#include "nn-framework/headers/layers/sigmoid_activation.hh"
#include "nn-framework/headers/structures/matrix.hh"
#include "nn-framework/coordinates_test.hh"
#include "nn-framework/headers/layers/softmax_activation.hh"


int main()
{
    srand(time(NULL));

    MSECost MSE;
    NeuralNetwork nn(&MSE);

    nn.addLayer(new LinearLayer("linear1", Dimensions(2, 10)));
    nn.addLayer(new ReLUActivation("softmaxtest"));
    nn.addLayer(new LinearLayer("linear2", Dimensions(10, 2)));
    nn.addLayer(new SoftmaxActivation("sigmoid"));

    

    cuda_check(cudaDeviceSynchronize());

    return 0;
}