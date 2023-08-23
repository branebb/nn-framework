#include "nn-framework/headers/structures/neural_network.hh"
#include "nn-framework/utils/error_check_cuda.hpp"

NeuralNetwork::NeuralNetwork(CostFunction* costFunction, float learning_rate) : 
	costFunction(costFunction),
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