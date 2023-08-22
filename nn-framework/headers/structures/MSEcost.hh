#pragma once

#include "matrix.hh"

class MSEcost
{
public:
    float cost(Matrix target, Matrix predicted);

    Matrix dCost(Matrix predicted, Matrix target, Matrix dY);
};