#pragma once

#include<Eigen/Dense>

#include"../Layer.hpp"

class FullConnectedLayer : public Layer {
    friend class CNN;
    static const std::string NAME;

    int inputSize, outputSize;

    Eigen::MatrixXd W;
    Eigen::VectorXd b;

    FullConnectedLayer() = default;

    void load(const std::string&) override;
    void save(const std::string&) const override;

    std::string getName() const override;

public:

    FullConnectedLayer(int inputSize, int outputSize);

    std::pair<int,int> getInputSize() const override;
    std::pair<int,int> getOutputSize() const override;

    Eigen::MatrixXd forward(Eigen::MatrixXd) const override;
    Eigen::MatrixXd backward(Eigen::MatrixXd input, Eigen::MatrixXd error) const override;
    void update(Eigen::MatrixXd input, Eigen::MatrixXd error, double rate) override;
};
