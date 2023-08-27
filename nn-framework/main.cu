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
#include "nn-framework/mnist.hh"

int main()
{
    MNIST traindata(128, 200, "datasets/mnist_train.csv");
    // CoordinatesDataset test(1, 1);
    std:: cout << "Data prepared successfully!\n";

    float lambda = 0.0f;
    L2 l2(lambda);

    MSECost MSE(&l2);

    Gradient grad;

    float lr = 0.1;

    NeuralNetwork nn(&MSE, &grad, &l2, lr);

    nn.addLayer(new LinearLayer("linear1", Dimensions(784, 256)));
    nn.addLayer(new ReLUActivation("relu"));
    nn.addLayer(new LinearLayer("linear2", Dimensions(256, 256)));
    nn.addLayer(new ReLUActivation("relu"));
    nn.addLayer(new LinearLayer("linear2", Dimensions(256, 10)));
    nn.addLayer(new SoftmaxActivation("softmax"));

    Matrix Y;

    for (int epoch = 0; epoch < 21; epoch++) 
    {
        float cost = 0.0;

        for (int batch = 0; batch < traindata.getNumOfBatches(); batch++) 
        {
            Y = nn.forward(traindata.getBatches().at(batch));
            nn.backprop(Y, traindata.getTargets().at(batch));
        
            LinearLayer* linearLayer = dynamic_cast<LinearLayer*>(nn.getLayers()[2]);
            Matrix layerW = linearLayer->getWeightsMatrix();
            cost += MSE.cost(Y, traindata.getTargets().at(batch), layerW);
        }

        if (epoch % 10 == 0) 
                std::cout << "Epoch: " << epoch << ", Cost: " << cost / traindata.getNumOfBatches() << std::endl;
    }

    return 0;

}