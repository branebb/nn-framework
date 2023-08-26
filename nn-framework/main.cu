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
#include "nn-framework/headers/optimizers/optimizer.hh"
#include "nn-framework/headers/optimizers/gradient.hh"
#include "nn-framework/headers/optimizers/adam.hh"
#include "nn-framework/headers/regularization/L2.hh"

int main()
{
    float lambda = 0.0f;
    L2 l2(lambda);
    MSECost MSE(nullptr);
    AdamOptimizer adam(0.9, 0.999, 1e-8);
    Gradient grad;
    float lr = 0.01;
    NeuralNetwork nn(&MSE, &adam, nullptr, lr);

    nn.addLayer(new LinearLayer("linear1", Dimensions(2, 30)));
    nn.addLayer(new ReLUActivation("relu"));
    nn.addLayer(new LinearLayer("linear2", Dimensions(30, 2)));
    nn.addLayer(new SoftmaxActivation("softmax"));

    CoordinatesDataset dataset(100, 20);

    Matrix Y;

    for (int epoch = 0; epoch < 101; epoch++) 
    {
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) 
        {
            Y = nn.forward(dataset.getBatches().at(batch));
            nn.backprop(Y, dataset.getTargets().at(batch));
        
            LinearLayer* linearLayer = dynamic_cast<LinearLayer*>(nn.getLayers()[2]);
            Matrix layerW = linearLayer->getWeightsMatrix();
            cost += MSE.cost(Y, dataset.getTargets().at(batch), layerW);
        }

        if (epoch % 10 == 0) 
                std::cout << "Epoch: " << epoch << ", Cost: " << cost / dataset.getNumOfBatches() << std::endl;
    }

    Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));

    float accuracy = nn.computeAccuracy(Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));

    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}