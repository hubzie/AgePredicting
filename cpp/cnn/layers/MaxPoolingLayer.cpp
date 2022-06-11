#include"MaxPoolingLayer.hpp"

using namespace Eigen;
using namespace std;

MaxPoolingLayer::MaxPoolingLayer(pair<int,int> inputSize, pair<int,int> poolSize):
    inputSize(inputSize), poolSize(poolSize) {

}

pair<int, int> MaxPoolingLayer::getInputSize() const { return inputSize; }
pair<int, int> MaxPoolingLayer::getOutputSize() const {
    return {inputSize.first / poolSize.first, inputSize.second / poolSize.second};
}



MatrixXd MaxPoolingLayer::forward(MatrixXd input) const {
    auto [x,y] = getOutputSize();
    MatrixXd output(x,y);

    auto [pw, ph] = poolSize;

    for(int i=0; i<x; i++)
        for(int j=0; j<y; j++) {
            double val = numeric_limits<double>::min();
            for(int a=0; a<pw; a++)
                for(int b=0; b<ph; b++)
                    val = max(val, input(i*pw+a, j*ph+b));
            output(i,j) = val;
        }

    return output;
}

MatrixXd MaxPoolingLayer::backward(MatrixXd input, MatrixXd error) const {
    auto output = forward(input);
    auto [x, y] = getInputSize();
    auto [pw, ph] = poolSize;

    MatrixXd result = MatrixXd::Zero(x, y);

    for(int i=0; i<x; i++)
        for(int j=0; j<y; j++)
            if(input(i,j) == output(i/pw, j/ph)) {
                output(i/pw, j/ph) = numeric_limits<double>::min();
                result(i,j) = error(i/pw, i/ph);
            }

    return result;
}

void MaxPoolingLayer::update(MatrixXd, MatrixXd, double) {}