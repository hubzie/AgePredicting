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

MatrixXd FullConnectedLayer::process(MatrixXd input) const {
    return W * input + b;
}

MatrixXd FullConnectedLayer::improve(MatrixXd input, MatrixXd output, double step) {
    // TODO: debug
    MatrixXd dW = step * input * output.transpose();
    auto db = output;
    output = W.transpose() * output;
    W += dW.transpose();
    b += db;
    return output;
}