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

    CoordinatesDataset dataset(100, 20);

    Matrix Y;
    for (int epoch = 0; epoch < 1001; epoch++) 
    {
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) 
        {
            Y = nn.forward(dataset.getBatches().at(batch));
            nn.backprop(Y, dataset.getTargets().at(batch));
            cost += MSE.cost(Y, dataset.getTargets().at(batch));
        }

        if (epoch % 100 == 0) 
        {
            std::cout << "Epoch: " << epoch
                << ", Cost: " << cost / dataset.getNumOfBatches()
                << std::endl;
        }
    }

    cuda_check(cudaDeviceSynchronize());

    return 0;
}