#include"SigmoidLayer.hpp"

using namespace Eigen;
using namespace std;

SigmoidLayer::SigmoidLayer(pair<int,int> size): size(size) {}

pair<int, int> SigmoidLayer::getInputSize() const { return size; }
pair<int, int> SigmoidLayer::getOutputSize() const { return size; }



static constexpr double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static constexpr double sigmoid_derivative(double x) { return x * (1-x); }



MatrixXd SigmoidLayer::forward(MatrixXd input) const {
    for(int i=0;i<size.first;i++)
        for(int j=0;j<size.second;j++)
            input(i,j) = sigmoid(input(i,j));
    return input;
}

MatrixXd SigmoidLayer::backward(MatrixXd input, MatrixXd error) const {
    auto output = forward(input);
    for(int i=0;i<size.first;i++)
        for(int j=0;j<size.second;j++)
            error(i,j) = error(i,j) * sigmoid_derivative(output(i,j));
    return error;
}

void SigmoidLayer::update(MatrixXd, MatrixXd, double) {}