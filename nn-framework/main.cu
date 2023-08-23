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


int main()
{
    MSECost MSE;
    NeuralNetwork nn(&MSE);
    nn.addLayer(new LinearLayer("linear1", Dimensions(2, 10)));
    nn.addLayer(new TanhActivation("tanh1"));
    // nn.addLayer(new LinearLayer("linear2", Dimensions(10, 10)));
    // nn.addLayer(new ReLUActivation("relu2"));
    nn.addLayer(new LinearLayer("linear3", Dimensions(10, 2)));
    nn.addLayer(new SigmoidActivation("sigmoidOut"));

    CoordinatesDataset dataset(100, 21);

    Matrix Y;
	// for (int epoch = 0; epoch < 1001; epoch++) 
    // {
	// 	float cost = 0.0;

	// 	for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) 
    //     {
	// 		Y = nn.forward(dataset.getBatches().at(batch));
	// 		nn.backprop(Y, dataset.getTargets().at(batch));
	// 		cost += MSE.cost(Y, dataset.getTargets().at(batch));
	// 	}

	// 	if (epoch % 100 == 0) 
    //     {
	// 		std::cout 	<< "Epoch: " << epoch
	// 					<< ", Cost: " << cost / dataset.getNumOfBatches()
	// 					<< std::endl;
	// 	}
	// }

    auto el = dataset.getBatches().at(0);

    std::cout << el[0] << std::endl;

    // std::vector<NNLayer*> layers = nn.getLayers();


    // for (NNLayer* layer : layers)
    // {
    //     printf("Layer name: %s\n", layer->getName().c_str());

    //     if (LinearLayer* linearLayer = dynamic_cast<LinearLayer*>(layer)) 
    //     {
    //         Matrix w = linearLayer->getBiasVector();
    //         printf("Layer dimensions: (%d, %d)\n", linearLayer->getXDim(), linearLayer->getYDim());
    //         for (int row = 0; row < w.dims.y; row++) 
    //         {
    //             for (int col = 0; col < w.dims.x; col++) 
    //             {
    //                 printf("%f ", w[row * w.dims.x + col]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }


    cuda_check(cudaDeviceSynchronize());

    return 0;
}