#pragma once

#include "cost_function.hh"
#include "nn-framework/headers/structures/matrix.hh"
#include "nn-framework/headers/regularization/regularization.hh"

class MSECost : public CostFunction 
{
private:
    Regularization* regularization;
public:
    MSECost(Regularization* regularization = nullptr);

    float cost(Matrix& target, Matrix& predicted, Matrix& W);

    Matrix dCost(Matrix& predicted, Matrix& target, Matrix& dY);
};
