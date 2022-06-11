#include"ReLuLayer.hpp"

using namespace Eigen;
using namespace std;

ReLuLayer::ReLuLayer(pair<int,int> size): size(size) {}

pair<int, int> ReLuLayer::getInputSize() const { return size; }
pair<int, int> ReLuLayer::getOutputSize() const { return size; }



static constexpr double relu(double x) { return max(0.0, x); }
static constexpr double relu_derivative(double x) { return (x > 0 ? 1.0 : 0.0); }



MatrixXd ReLuLayer::forward(MatrixXd input) const {
    for(int i=0;i<size.first;i++)
        for(int j=0;j<size.second;j++)
            input(i,j) = relu(input(i,j));
    return input;
}

MatrixXd ReLuLayer::backward(MatrixXd input, MatrixXd error) const {
    auto output = forward(input);
    for(int i=0;i<size.first;i++)
        for(int j=0;j<size.second;j++)
            error(i,j) = error(i,j) * relu_derivative(output(i,j));
    return error;
}

void ReLuLayer::update(MatrixXd, MatrixXd, double) {}