#pragma once

#include"Layer.hpp"

class ReLuLayer : public Layer {
    std::pair<int,int> size;

public:

    ReLuLayer(std::pair<int,int> size);

    std::pair<int,int> getInputSize() const override;
    std::pair<int,int> getOutputSize() const override;

    Eigen::MatrixXd forward(Eigen::MatrixXd) const override;
    Eigen::MatrixXd backward(Eigen::MatrixXd input, Eigen::MatrixXd error) const override;
    void update(Eigen::MatrixXd input, Eigen::MatrixXd error, double rate) override;
};
