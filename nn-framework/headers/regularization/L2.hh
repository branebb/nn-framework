#pragma once

#include "regularization.hh"

class L2 : public Regularization 
{
private:
    float lambda;
public:
    L2(float lambda);

    void gradientRegularization(Matrix& W, Matrix &dW, int size);
    float costRegularization(Matrix& W);
};