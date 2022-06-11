#include"SigmoidLayer.hpp"

using namespace Eigen;
using namespace std;

SigmoidLayer::SigmoidLayer(int size): size(size) {}

pair<int, int> SigmoidLayer::getInputSize() const { return {size, 1}; }
pair<int, int> SigmoidLayer::getOutputSize() const { return {size, 1}; }



static constexpr double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static constexpr double sigmoid_derivative(double x) {
    double t = exp(-x);
    return t / ((t+1) * (t+1));
}



MatrixXd SigmoidLayer::forward(MatrixXd input) const {
    for(int i=0;i<input.size();i++)
        input(i) = sigmoid(input(i));
    return input;
}

MatrixXd SigmoidLayer::backward(MatrixXd input, MatrixXd error) const {
    for(int i=0;i<error.size();i++)
        error(i) = error(i) * sigmoid_derivative(input(i));
    return error;
}

void SigmoidLayer::update(MatrixXd input, MatrixXd error, double rate) {}