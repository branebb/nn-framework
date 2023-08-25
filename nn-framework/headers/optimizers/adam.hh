#pragma once

#include "optimizer.hh"

class AdamOptimizer : public Optimizer
{
private:
    float beta1, beta2, epsilon;
    Matrix mW, vW, mb, vb;
    int t;

public:
    AdamOptimizer(float beta1, float beta2, float epsilon);

    void initialize(Dimensions weightDims, Dimensions biasDims);

    void updateW(Matrix &dW, Matrix &W, float learning_rate);
    void updateB(Matrix &db, Matrix &b, float learning_rate);

    void updateStep(Matrix &dW, Matrix &W, Matrix &db, Matrix &b, float learning_rate);

    void setMatricesToZero();

    void increaseT();
};