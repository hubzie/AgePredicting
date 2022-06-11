#include"AveragePoolingLayer.hpp"

#include<numeric>

using namespace Eigen;
using namespace std;

AveragePoolingLayer::AveragePoolingLayer(pair<int,int> inputSize, pair<int,int> poolSize):
    inputSize(inputSize), poolSize(poolSize) {

}

pair<int, int> AveragePoolingLayer::getInputSize() const { return inputSize; }
pair<int, int> AveragePoolingLayer::getOutputSize() const {
    return {inputSize.first / poolSize.first, inputSize.second / poolSize.second};
}



MatrixXd AveragePoolingLayer::forward(MatrixXd input) const {
    auto [x,y] = getOutputSize();
    MatrixXd output(x,y);

    auto [pw, ph] = poolSize;

    for(int i=0; i<x; i++)
        for(int j=0; j<y; j++) {
            double val = 0.0;
            for(int a=0; a<pw; a++)
                for(int b=0; b<ph; b++)
                    val += input(i*pw+a, j*ph+b);
            output(i,j) = val / (pw * ph);
        }

    return output;
}

MatrixXd AveragePoolingLayer::backward(MatrixXd input, MatrixXd error) const {
    auto [x, y] = getInputSize();
    auto [pw, ph] = poolSize;

    MatrixXd result = MatrixXd::Zero(x, y);

    for(int i=0; i<x; i++)
        for(int j=0; j<y; j++)
            result(i, j) = error(i/pw, i/ph) / (pw * ph);

    return result;
}

void AveragePoolingLayer::update(MatrixXd, MatrixXd, double) {}