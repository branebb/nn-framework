#pragma once

#include "nn_layer.hh"

class LinearLayer : public NNLayer
{
private:
    Matrix W, b, Z, A, dA;

    void initializeBiasWithZeros();
	void initializeWeightsRandomly();

	void computeStoreBackpropError(Matrix& dZ);
	void computeStoreLayerOutput(Matrix& A);
	void updateWeights(Matrix& dZ, float learning_rate);
	void updateBias(Matrix& dZ, float learning_rate);

public:
    LinearLayer(std::string name, Dimensions W_shape);
	~LinearLayer();

	Matrix& forward(Matrix& A);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;

};