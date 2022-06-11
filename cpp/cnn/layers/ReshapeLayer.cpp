#include"ReshapeLayer.hpp"

using namespace Eigen;
using namespace std;

ReshapeLayer::ReshapeLayer(pair<int,int> inputSize, pair<int,int> outputSize):
    inputSize(inputSize), outputSize(outputSize) {
    assert(inputSize.first * inputSize.second == outputSize.first * outputSize.second);
}

pair<int, int> ReshapeLayer::getInputSize() const { return inputSize; }
pair<int, int> ReshapeLayer::getOutputSize() const { return outputSize; }



MatrixXd ReshapeLayer::forward(MatrixXd input) const {
    return input.reshaped(outputSize.first, outputSize.second);
}

MatrixXd ReshapeLayer::backward(MatrixXd, MatrixXd error) const {
    return error.reshaped(inputSize.first, inputSize.second);
}

void ReshapeLayer::update(MatrixXd, MatrixXd, double) {}