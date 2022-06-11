#pragma once

#include<vector>

#include"Utils.hpp"

#include"Layer.hpp"
#include"layers/FullConnectedLayer.hpp"
#include"layers/SigmoidLayer.hpp"

class CNN {
    std::pair<int,int> inputSize, outputSize;
    std::vector<Layer*> layers;

public:

    CNN(std::pair<int,int> inputSize);

    std::pair<int,int> getInputSize() const;
    std::pair<int,int> getOutputSize() const;

    void addLayer(Layer*);

    void train(const std::vector<Data>&);
};
