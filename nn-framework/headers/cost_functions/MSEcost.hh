#pragma once

#include "cost_function.hh"
#include "nn-framework/headers/structures/matrix.hh"

class MSECost : public CostFunction 
{
public:
    float cost(Matrix target, Matrix predicted);

    Matrix dCost(Matrix predicted, Matrix target, Matrix dY);
};
