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
    MNIST traindata(32, 1875, "datasets/mnist_train.csv");

    float lambda = 0.01f;
    L2 l2(lambda);

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-4f;

    AdamOptimizer adam(beta1, beta2, epsilon);

    MSECost MSE(&l2);

    Gradient grad;

    float lr = 0.001f;

    NeuralNetwork nn(&MSE, &adam, &l2, lr);

    nn.addLayer(new LinearLayer("linear1", Dimensions(784, 128)));
    nn.addLayer(new ReLUActivation("relu"));
    nn.addLayer(new LinearLayer("linear2", Dimensions(128, 64)));
    nn.addLayer(new ReLUActivation("relu"));
    nn.addLayer(new LinearLayer("linear2", Dimensions(64, 10)));
    nn.addLayer(new SoftmaxActivation("softmax"));

    Matrix Y;

    for (int epoch = 0; epoch < 11; epoch++) 
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

        if (epoch % 1 == 0) 
                std::cout << "Epoch: " << epoch << ", Cost: " << cost / traindata.getNumOfBatches() << std::endl;
    }

    MNIST testdata(32, 312, "datasets/mnist_test.csv");
    Matrix A;

    float accuracy = 0.0f;
    for (int i = 0; i < testdata.getNumOfBatches(); i++)
    {
        A = nn.forward(testdata.getBatches().at(i));
        accuracy += nn.computeAccuracy(A, testdata.getTargets().at(i));
    }
    
    std::cout << "Accuracy on test data: " << accuracy / testdata.getNumOfBatches();

    // Epoch: 0, Cost: 0.273616
    // Epoch: 1, Cost: 0.243005
    // Epoch: 2, Cost: 0.222543
    // Epoch: 3, Cost: 0.207329
    // Epoch: 4, Cost: 0.18212
    // Epoch: 5, Cost: 0.176571
    // Epoch: 6, Cost: 0.153203
    // Epoch: 7, Cost: 0.149846
    // Epoch: 8, Cost: 0.140936
    // Epoch: 9, Cost: 0.134518
    // Epoch: 10, Cost: 0.129513
    // Accuracy on test data: 0.88111
    // Last run on this network with MNIST digits

    return 0;

}