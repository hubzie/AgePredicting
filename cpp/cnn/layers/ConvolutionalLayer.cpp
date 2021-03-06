#include"ConvolutionalLayer.hpp"

#include<fstream>
#include<random>

using namespace Eigen;
using namespace std;

const std::string ConvolutionalLayer::NAME = "CONVOLUTIONAL_LAYER";

void ConvolutionalLayer::load(const string& path) {
    ifstream file(path);
    assert(file.is_open());

    string name;
    assert(file >> name);
    assert(name == NAME);

    assert(file >> inputSize.first >> inputSize.second);
    assert(file >> kernelSize.first >> kernelSize.second);

    W = MatrixXd(kernelSize.first, kernelSize.second);
    for(int i=0;i<W.rows();i++)
        for(int j=0;j<W.cols();j++)
            assert(file >> W(i,j));
    assert(file >> b);
}

void ConvolutionalLayer::save(const string& path) const {
    ofstream file(path);
    file << NAME << "\n";
    file << inputSize.first << " " << inputSize.second << "\n";
    file << kernelSize.first << " " << kernelSize.second << "\n";
    file << W << "\n";
    file << b << "\n";
}

string ConvolutionalLayer::getName() const { return NAME; }

ConvolutionalLayer::ConvolutionalLayer(pair<int,int> inputSize, pair<int,int> kernelSize):
        inputSize(inputSize), kernelSize(kernelSize),
        W(MatrixXd::Random(kernelSize.first, kernelSize.second)) {
    static random_device rd;
    static mt19937 g(rd());
    b = uniform_real_distribution<double>(-1.0, 1.0)(g);
}

pair<int, int> ConvolutionalLayer::getInputSize() const { return inputSize; }
pair<int, int> ConvolutionalLayer::getOutputSize() const {
    return {inputSize.first - kernelSize.first + 1, inputSize.second - kernelSize.second + 1};
}



MatrixXd ConvolutionalLayer::forward(MatrixXd input) const {
    auto [x,y] = getOutputSize();
    MatrixXd output(x,y);

    auto [pw, ph] = kernelSize;

    for(int i=0; i<x; i++)
        for(int j=0; j<y; j++) {
            output(i,j) = b;
            for(int p=0; p<pw; p++)
                for(int q=0; q<ph; q++)
                    output(i,j) += W(p,q) * input(i+p, j+q);
        }

    return output;
}

MatrixXd ConvolutionalLayer::backward(MatrixXd, MatrixXd error) const {
    auto [x,y] = inputSize;
    auto [pw, ph] = kernelSize;

    MatrixXd result = MatrixXd::Zero(inputSize.first, inputSize.second);

    for(int i=0; i<x-pw+1; i++)
        for(int j=0; j<y-ph+1; j++)
            for(int p=0; p<pw; p++)
                for(int q=0; q<ph; q++)
                    result(i+p,j+q) += W(pw-p-1,ph-q-1) * error(i, j);

    return result;
}

void ConvolutionalLayer::update(MatrixXd input, MatrixXd error, double step) {
    for(int i=0;i<error.rows();i++)
        for(int j=0;j<error.cols();j++)
            b -= step * error(i,j);

    auto [x,y] = inputSize;
    auto [pw, ph] = kernelSize;

    MatrixXd dW = MatrixXd::Zero(pw,ph);
    for(int p=0; p<pw; p++)
        for(int q=0; q<ph; q++)
            for(int i=0; i<x-pw+1; i++)
                for(int j=0; j<y-ph+1; j++)
                    W(p,q) -= step * error(i,j) * input(i+p, j+q);
}