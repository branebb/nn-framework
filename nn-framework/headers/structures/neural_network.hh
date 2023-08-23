#pragma once

#include <vector>
#include "nn-framework/headers/layers/nn_layer.hh"
#include "nn-framework/headers/cost_functions/cost_function.hh"
class NeuralNetwork
{
private:
    std::vector<NNLayer*> layers;
    
    Matrix Y, dY;

    float learning_rate;

	CostFunction* costFunction;

public:
    NeuralNetwork(CostFunction* costFunction, float learning_rate = 0.01);
	~NeuralNetwork();

	void setCostFunction(CostFunction* costFunction);

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};