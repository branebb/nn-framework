#pragma once

#include "nn-framework/headers/structures/matrix.hh"

class CostFunction 
{
public:
    virtual ~CostFunction() = 0;

    virtual float cost(Matrix target, Matrix predicted) = 0;
    virtual Matrix dCost(Matrix predicted, Matrix target, Matrix dY) = 0;
};
