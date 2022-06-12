#include"ReshapeLayer.hpp"

#include<fstream>

using namespace Eigen;
using namespace std;

const std::string ReshapeLayer::NAME = "RESHAPE_LAYER";

void ReshapeLayer::load(const string& path) {
    ifstream file(path);
    assert(file.is_open());

    string name;
    assert(file >> name);
    assert(name == NAME);

    assert(file >> inputSize.first >> inputSize.second);
    assert(file >> outputSize.first >> outputSize.second);
}

void ReshapeLayer::save(const string& path) const {
    ofstream file(path);
    file << NAME << "\n";
    file << inputSize.first << " " << inputSize.second << "\n";
    file << outputSize.first << " " << outputSize.second << "\n";
}

string ReshapeLayer::getName() const { return NAME; }

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