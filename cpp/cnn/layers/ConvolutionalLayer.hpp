#pragma once

#include<Eigen/Dense>

#include"Layer.hpp"

class ConvolutionalLayer : public Layer {
    std::pair<int,int> inputSize, kernelSize;
    Eigen::MatrixXd W;
    double b;

public:

    ConvolutionalLayer(std::pair<int,int> inputSize, std::pair<int,int> kernelSize);

    std::pair<int,int> getInputSize() const override;
    std::pair<int,int> getOutputSize() const override;

    Eigen::MatrixXd forward(Eigen::MatrixXd) const override;
    Eigen::MatrixXd backward(Eigen::MatrixXd input, Eigen::MatrixXd error) const override;
    void update(Eigen::MatrixXd input, Eigen::MatrixXd error, double rate) override;
};