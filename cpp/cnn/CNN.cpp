#include"CNN.hpp"

#include<Eigen/Dense>

#include<cassert>
#include<iostream>

using namespace Eigen;
using namespace std;

CNN::CNN(pair<int, int> inputSize): inputSize(inputSize), outputSize(inputSize) {}
CNN::~CNN() { for(auto l : layers) delete l; }

std::pair<int, int> CNN::getInputSize() const { return inputSize; }
std::pair<int, int> CNN::getOutputSize() const { return outputSize; }

void CNN::addLayer(Layer* layer) {
    assert(outputSize == layer->getInputSize());
    outputSize = layer->getOutputSize();
    layers.push_back(layer);
}

void CNN::train(const std::vector<Data>& data) {
    cerr << "CNN: Layers count = " << layers.size() << endl;
    cerr << "CNN: Training data set size = " << data.size() << endl;

    random_device rd;
    mt19937 g(rd());
    auto getIdx = [&g](int a, int b) {
        return uniform_int_distribution<int>(a,b)(g);
    };

    // Forward
    double step = 1.0;
    int size = layers.size();
    vector<MatrixXd> input(size + 1), error(size + 1);

    for(int it=1;it<=100*1000;it++) {
        if(it%1000 == 0) {
            cerr << "CNN: Iteration #" << it << endl;
            step *= 0.9;
        }

        auto& d = data[getIdx(0, data.size()-1)];

        input[0] = d.x;
        for (int i = 0; i < size; i++)
            input[i + 1] = layers[i]->forward(input[i]);

        error[size] = input[size];
        error[size](d.y) -= 1;

        for (int i = size - 1; i >= 0; i--)
            error[i] = layers[i]->backward(input[i], error[i + 1]);

        for (int i = 0; i < size; i++)
            layers[i]->update(input[i], error[i + 1], step);
    }
}

short CNN::predict(MatrixXd x) const {
    for(auto l : layers)
        x = l->forward(x);

    int idx = 0;
    for(int i=0;i<x.size();i++)
        if(x(idx) < x(i)) idx = i;
    return idx;
}