#pragma once

#include "nn-framework/headers/structures/matrix.hh"

class Optimizer 
{
public:
    virtual void updateW(Matrix &dW, Matrix &W, float learning_rate) = 0;
    virtual void updateB(Matrix &db, Matrix &b, float learning_rate) = 0;
};