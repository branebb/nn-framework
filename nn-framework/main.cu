// Standard library
#include <iostream>
#include <stdio.h>

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
#include "nn-framework/mnist/mnist.hh"
#include "nn-framework/headers/cost_functions/CrossEntropy.hh"

int main()
{
    float lambda = 0.00f;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    float learning_rate = 0.01f;

    MNIST traindata(32, 1875, "datasets/mnist_train.csv");

    L2 L2(lambda);

    CrossEntropyCost crossEntropy(&L2);

    AdamOptimizer adam(beta1, beta2, epsilon);

    MSECost MSE(&L2);

    NeuralNetwork nn(&crossEntropy, &adam, &L2, learning_rate);

    nn.addLayer(new LinearLayer("linear1", Dimensions(784, 128)));
    nn.addLayer(new ReLUActivation("relu"));
    nn.addLayer(new LinearLayer("linear2", Dimensions(128, 64)));
    nn.addLayer(new ReLUActivation("relu2"));
    nn.addLayer(new LinearLayer("linear3", Dimensions(64, 10)));
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
            cost += crossEntropy.cost(Y, traindata.getTargets().at(batch), layerW);
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

    return 0;
}