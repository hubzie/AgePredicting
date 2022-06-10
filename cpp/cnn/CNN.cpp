#include"CNN.hpp"

#include<Eigen/Dense>

#include<cassert>
#include<iostream>

using namespace Eigen;
using namespace std;

CNN::CNN(pair<int, int> inputSize): inputSize(inputSize), outputSize(inputSize) {}

std::pair<int, int> CNN::getInputSize() const { return inputSize; }
std::pair<int, int> CNN::getOutputSize() const { return outputSize; }

void CNN::addLayer(Layer* layer) {
    assert(outputSize == layer->getInputSize());
    outputSize = layer->getOutputSize();
    layers.push_back(layer);
}

void CNN::train(const std::vector<Data>& data) {
    vector<MatrixXd> tmp;

    // TODO: debug
    for(int it=0;it<10;it++)
        for(const auto& d : data) {
            auto x = d.x;

            tmp.clear();
            for(auto l : layers) {
                tmp.emplace_back(x);
                x = l->process(x);
            }

            for(int i=0;i<x.size();i++)
                x(i) = exp(x(i));
            x /= x.sum();
            x(d.y == 2 ? 0 : 1) -= 1.0; // TODO: demo

            for(int i=layers.size()-1;i>=0;i--)
                x = layers[i]->improve(tmp[i], x, 0.001);
        }

    // Evaluate
    int acc = 0;
    for(const auto& d : data) {
        auto x = d.x;

        tmp.clear();
        for(auto l : layers) {
            tmp.emplace_back(x);
            x = l->process(x);
        }

        if (d.y == (x(0) > x(1) ? 2 : 50)) acc++;
    }

    cout << "Accuracy " << acc << " / " << data.size() << "\n";
}