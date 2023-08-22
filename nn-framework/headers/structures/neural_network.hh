#pragma once

#include <vector>
#include "layers/linear_layer.hh"
#include "MSEcost.hh"

class NeuralNetwork
{
private:
    std::vector<NNLayer*> layers;
    
    Matrix Y, dY;

    float learning_rate;

public:
    NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};