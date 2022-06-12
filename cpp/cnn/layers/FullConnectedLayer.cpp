#include"FullConnectedLayer.hpp"

#include<fstream>
#include<iostream>

using namespace Eigen;
using namespace std;

const std::string FullConnectedLayer::NAME = "FULL_CONNECTED_LAYER";

void FullConnectedLayer::load(const string& path) {
    ifstream file(path);
    assert(file.is_open());

    string name;
    assert(file >> name);
    assert(name == NAME);

    assert(file >> inputSize >> outputSize);
    W = MatrixXd(outputSize, inputSize);
    b = VectorXd(outputSize);

    for(int i=0;i<W.rows();i++)
        for(int j=0;j<W.cols();j++)
            assert(file >> W(i,j));
    for(int i=0;i<b.size();i++)
        assert(file >> b(i));
}

void FullConnectedLayer::save(const string& path) const {
    ofstream file(path);
    file << NAME << "\n";
    file << inputSize << " " << outputSize << "\n";
    file << W << "\n";
    file << b.transpose() << "\n";
}

string FullConnectedLayer::getName() const { return NAME; }

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