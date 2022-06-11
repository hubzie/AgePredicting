#pragma once

#include<Eigen/Dense>

#include"../Layer.hpp"

class FullConnectedLayer : public Layer {
    int inputSize, outputSize;

    Eigen::MatrixXd W;
    Eigen::VectorXd b;

public:

    FullConnectedLayer(int inputSize, int outputSize);

    std::pair<int,int> getInputSize() const override;
    std::pair<int,int> getOutputSize() const override;

    Eigen::MatrixXd forward(Eigen::MatrixXd) const override;
    Eigen::MatrixXd backward(Eigen::MatrixXd input, Eigen::MatrixXd error) const override;
    void update(Eigen::MatrixXd input, Eigen::MatrixXd error, double rate) override;
};
