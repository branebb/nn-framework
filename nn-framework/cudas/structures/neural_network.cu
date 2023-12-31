#include "nn-framework/headers/structures/neural_network.hh"
#include "nn-framework/utils/error_check_cuda.hpp"
#include "nn-framework/headers/layers/linear_layer.hh"
#include "nn-framework/headers/optimizers/adam.hh"

#include <assert.h>

NeuralNetwork::NeuralNetwork(CostFunction* costFunction, Optimizer* optimizer, Regularization* regularization , float learning_rate) : 
	costFunction(costFunction),
	optimizer(optimizer),
	regularization(regularization),
	learning_rate(learning_rate) 
{ }

NeuralNetwork::~NeuralNetwork() 
{
	for (auto layer : layers)
		delete layer;
}

void NeuralNetwork::setCostFunction(CostFunction* costFunction) 
{
    this->costFunction = costFunction;
}

void NeuralNetwork::addLayer(NNLayer* layer) 
{
	this->layers.push_back(layer);

	LinearLayer* linearLayer = dynamic_cast<LinearLayer*>(layer);


	if (linearLayer)
	{
		Optimizer* newOptimizer = nullptr;
		AdamOptimizer* adamOptimizer = dynamic_cast<AdamOptimizer*>(optimizer);

		if(adamOptimizer)
		{
			newOptimizer = new AdamOptimizer(adamOptimizer->getBeta1(), adamOptimizer->getBeta2(), adamOptimizer->getEpsilon());
			newOptimizer->initialize(Dimensions(linearLayer->getXDim(), linearLayer->getYDim()), Dimensions(linearLayer->getXDim(), 1));
		}
		else
		{
			newOptimizer = optimizer;
		}
		
        linearLayer->setOptimizer(newOptimizer);
		linearLayer->setRegularization(regularization);
	}
}

Matrix NeuralNetwork::forward(Matrix X) 
{
	Matrix Z = X;

	for (auto layer : layers) 
		Z = layer->forward(Z);

	Y = Z;

	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) 
{
	dY.allocateMemoryIfNotAllocated(predictions.dims);

	Matrix error = costFunction->dCost(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) 
		error = (*it)->backprop(error, learning_rate);
	
	cuda_check(cudaDeviceSynchronize());
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const { return layers; }

float NeuralNetwork::computeAccuracy(Matrix& predictions, Matrix& target)
{	
	assert(predictions.dims.x == target.dims.x && predictions.dims.y == target.dims.y);

	if(predictions.deviceAllocation())
		predictions.copyDeviceToHost();
	
	predictions.oneHotEncoding();

	int correct = 0;

	for(int col = 0; col < target.dims.x; col++)
	{	
		int flag = 1;
		for(int row = 0; row < target.dims.y; row++)
		{
			if(predictions[col + target.dims.x * row] != target[col + target.dims.x * row])
				flag = 0;
		}

		if(flag)
			correct++;
	}

	return static_cast<float>(correct) / target.dims.x;
}