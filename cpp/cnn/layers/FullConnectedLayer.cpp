#include"FullConnectedLayer.hpp"

#include<iostream>

using namespace Eigen;
using namespace std;

FullConnectedLayer::FullConnectedLayer(int inputSize, int outputSize):
    inputSize(inputSize), outputSize(outputSize),
    W(MatrixXd::Random(outputSize, inputSize)),
    b(VectorXd::Random(outputSize))
{}

std::pair<int, int> FullConnectedLayer::getInputSize() const { return {inputSize, 1}; }
std::pair<int, int> FullConnectedLayer::getOutputSize() const { return {outputSize, 1}; }

MatrixXd FullConnectedLayer::forward(MatrixXd input) const {
    return W * input + b;
}

MatrixXd FullConnectedLayer::backward(MatrixXd, MatrixXd error) const {
    return W.transpose() * error;
}

void FullConnectedLayer::update(MatrixXd input, MatrixXd error, double rate) {
    W -= error * input.transpose() * rate;
    b -= error * rate;
}