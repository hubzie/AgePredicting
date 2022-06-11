#include"AveragePoolingLayer.hpp"

using namespace Eigen;
using namespace std;

AveragePoolingLayer::AveragePoolingLayer(pair<int,int> inputSize, pair<int,int> poolSize):
    inputSize(inputSize), poolSize(poolSize) {
    assert(inputSize.first % poolSize.first == 0);
    assert(inputSize.second % poolSize.second == 0);
}

pair<int, int> AveragePoolingLayer::getInputSize() const { return inputSize; }
pair<int, int> AveragePoolingLayer::getOutputSize() const {
    return {inputSize.first / poolSize.first,
            inputSize.second / poolSize.second};
}



MatrixXd AveragePoolingLayer::forward(MatrixXd input) const {
    auto [x,y] = getOutputSize();
    MatrixXd output = MatrixXd::Zero(x,y);

    auto [pw, ph] = poolSize;
    double cnt = pw*ph;

    for(int i=0; i<x; i++)
        for(int j=0; j<y; j++)
            for(int p=0; p<pw; p++)
                for(int q=0; q<ph; q++)
                    output(i,j) += input(i*pw+p, j*ph+q) / cnt;

    return output;
}

MatrixXd AveragePoolingLayer::backward(MatrixXd, MatrixXd error) const {
    auto [x, y] = inputSize;
    auto [pw, ph] = poolSize;

    MatrixXd result = MatrixXd::Zero(x, y);

    for(int i=0; i<x; i++)
        for(int j=0; j<y; j++)
            result(i, j) = error(i/pw, j/ph) / (pw * ph);

    return result;
}

void AveragePoolingLayer::update(MatrixXd, MatrixXd, double) {}