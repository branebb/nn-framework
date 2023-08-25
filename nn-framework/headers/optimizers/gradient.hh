#pragma once

#include "optimizer.hh"

class Gradient : public Optimizer 
{
public:
    void updateW(Matrix &dW, Matrix &W, float learning_rate);
    void updateB(Matrix &db, Matrix &b, float learning_rate);

    void updateStep(Matrix &dW, Matrix &W, Matrix &db, Matrix &b, float learning_rate);
};