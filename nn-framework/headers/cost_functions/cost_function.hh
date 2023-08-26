#pragma once

#include "nn-framework/headers/structures/matrix.hh"

class CostFunction 
{
public:
    virtual float cost(Matrix& target, Matrix& predicted, Matrix& W) = 0;
    virtual Matrix dCost(Matrix& predicted, Matrix& target, Matrix& dY) = 0;
};
