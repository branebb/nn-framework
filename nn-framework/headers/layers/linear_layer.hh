#pragma once

#include "nn_layer.hh"
#include "nn-framework/headers/optimizers/optimizer.hh"

class LinearLayer : public NNLayer
{
private:
    Matrix W, b, Z, A, dA, dW, db;

	Optimizer* optimizer;

    void initializeBiasWithZeros();
	void initializeWeightsRandomly();

	void computeStoreBackpropError(Matrix& dZ);
	void computeStoreLayerOutput(Matrix& A);
	void computeStoreWGradient(Matrix& dZ);
	void computeStoreBGradient(Matrix& dZ);
	void updateWeights(Matrix& dZ, float learning_rate);
	void updateBias(Matrix& dZ, float learning_rate);

public:
    LinearLayer(std::string name, Dimensions W_dims);
	~LinearLayer();

	Matrix& forward(Matrix& A);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;

	void setOptimizer(Optimizer* optimizer);

};