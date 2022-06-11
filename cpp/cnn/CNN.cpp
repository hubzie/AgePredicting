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
    // Forward
    double step = 1.0;
    for(int it=1;it<=10000;it++) {
        if(it%1000 == 0) {
            cerr << "CNN: Iteration #" << it << endl;
            step *= 0.9;
        }

        int size = layers.size();
        vector<MatrixXd> input(size + 1), error(size + 1);
        for (auto &d: data) {
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

    vector<int> ok(outputSize.first);
    for(auto& d : data) {
        auto x = d.x;
        for(auto l : layers)
            x = l->forward(x);

        int idx = 0;
        for(int i=0;i<x.size();i++)
            if(x(idx) < x(i)) idx = i;

        if(idx == d.y) ok[idx]++;
    }

    int acc = accumulate(ok.begin(), ok.end(), 0);
    cout << "Accuracy = " << 100.0*acc / data.size() << "%\n";
    cout << "\t" << acc << "/" << data.size() << " (" << ok[0];
    for(int i=1;i<ok.size();i++) cout << "," << ok[i];
    cout << ")\n";
}