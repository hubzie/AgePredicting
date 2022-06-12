#pragma once

#include<vector>

#include"Utils.hpp"

#include"Layer.hpp"
#include"layers/AveragePoolingLayer.hpp"
#include"layers/ConvolutionalLayer.hpp"
#include"layers/FullConnectedLayer.hpp"
#include"layers/ReLuLayer.hpp"
#include"layers/ReshapeLayer.hpp"
#include"layers/SigmoidLayer.hpp"
#include"layers/MaxPoolingLayer.hpp"

class CNN {
    static const std::string NAME;

    std::pair<int,int> inputSize, outputSize;
    std::vector<Layer*> layers;

public:

    CNN(std::pair<int,int> inputSize);
    CNN(const std::string& path);
    virtual ~CNN();

    void save(const std::string& path);

    std::pair<int,int> getInputSize() const;
    std::pair<int,int> getOutputSize() const;

    void addLayer(Layer*);

    void train(const std::vector<Data>&);
    short predict(Eigen::MatrixXd) const;
};