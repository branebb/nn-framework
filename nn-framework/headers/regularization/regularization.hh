#pragma once

#include "nn-framework/headers/structures/matrix.hh"

class Regularization
{
public:
    virtual void gradientRegularization(Matrix& W, Matrix &dW, int size) = 0;
    virtual float costRegularization(Matrix &W) = 0;
};