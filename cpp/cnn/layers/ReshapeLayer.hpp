#pragma once

#include"Layer.hpp"

class ReshapeLayer : public Layer {
    friend class CNN;
    static const std::string NAME;

    std::pair<int,int> inputSize, outputSize;

    ReshapeLayer() = default;

    void load(const std::string&) override;
    void save(const std::string&) const override;

    std::string getName() const override;

public:

    ReshapeLayer(std::pair<int,int> inputSize, std::pair<int,int> outputSize);

    std::pair<int,int> getInputSize() const override;
    std::pair<int,int> getOutputSize() const override;

    Eigen::MatrixXd forward(Eigen::MatrixXd) const override;
    Eigen::MatrixXd backward(Eigen::MatrixXd input, Eigen::MatrixXd error) const override;
    void update(Eigen::MatrixXd input, Eigen::MatrixXd error, double rate) override;
};