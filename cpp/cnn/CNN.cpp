#include"CNN.hpp"

#include<Eigen/Dense>

#include<cassert>
#include<fstream>
#include<iostream>

using namespace Eigen;
using namespace std;

const std::string CNN::NAME = "CNN_SAVE_FILE";

CNN::CNN(pair<int, int> inputSize): inputSize(inputSize), outputSize(inputSize) {}
CNN::~CNN() { for(auto l : layers) delete l; }

CNN::CNN(const string& path) {
    ifstream net(path+"net.txt");
    assert(net.is_open());

    string line;
    assert(net >> line);
    assert(line == NAME);
    assert(net >> inputSize.first >> inputSize.second);
    outputSize = inputSize;

    string name, file;
    while(net >> name >> file) {
        Layer* layer;
        if(name == AveragePoolingLayer::NAME) layer = new AveragePoolingLayer();
        else if(name == ConvolutionalLayer::NAME) layer = new ConvolutionalLayer();
        else if(name == FullConnectedLayer::NAME) layer = new FullConnectedLayer();
        else if(name == MaxPoolingLayer::NAME) layer = new MaxPoolingLayer();
        else if(name == ReLuLayer::NAME) layer = new ReLuLayer();
        else if(name == ReshapeLayer::NAME) layer = new ReshapeLayer();
        else if(name == SigmoidLayer::NAME) layer = new SigmoidLayer();
        else assert(false);

        layer->load(path+file);
        addLayer(layer);
    }
}

void CNN::save(const string& path) {
    ofstream net(path+"net.txt");
    assert(net.is_open());

    net << NAME << "\n";
    net << inputSize.first << " " << inputSize.second << "\n";
    for(int i=0;i<layers.size();i++) {
        auto l = layers[i];
        net << l->getName() << " " << "layer_" << i << ".txt" << "\n";
        l->save(path+"layer_"+to_string(i)+".txt");
    }
}

std::pair<int, int> CNN::getInputSize() const { return inputSize; }
std::pair<int, int> CNN::getOutputSize() const { return outputSize; }

void CNN::addLayer(Layer* layer) {
    assert(outputSize == layer->getInputSize());
    outputSize = layer->getOutputSize();
    layers.push_back(layer);
}

void CNN::train(const std::vector<Data>& data) {
    //cerr << "CNN: Layers count = " << layers.size() << endl;
    //cerr << "CNN: Training data set size = " << data.size() << endl;

    random_device rd;
    mt19937 g(rd());
    auto getIdx = [&g](int a, int b) {
        return uniform_int_distribution<int>(a,b)(g);
    };

    // Forward
    double step = 0.1;
    int size = layers.size();
    vector<MatrixXd> input(size + 1), error(size + 1);

    for(int it=1;it<=50*1000;it++) {
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

        if(it%1000 == 0) {
            step *= 0.9;

            int cnt = 0;
            for(auto& d : data)
                if(d.y == predict(d.x)) cnt++;

            double accuracy = 100.0 * cnt / data.size();
            //cerr << "CNN: Iteration #" << it << ", accuracy = " << accuracy << "%" << endl;
        }
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