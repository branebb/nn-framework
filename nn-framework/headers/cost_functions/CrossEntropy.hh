#pragma once

#include "cost_function.hh"
#include "nn-framework/headers/structures/matrix.hh"
#include "nn-framework/headers/regularization/regularization.hh"

class CrossEntropyCost : public CostFunction 
{
private:
    Regularization* regularization;
public:
    CrossEntropyCost(Regularization* regularization = nullptr);

    float cost(Matrix& target, Matrix& predicted, Matrix& W);

    Matrix dCost(Matrix& predicted, Matrix& target, Matrix& dY);
};
