#pragma once

#include<Eigen/Dense>

class Layer {
public:
    virtual std::pair<int,int> getInputSize() const = 0;
    virtual std::pair<int,int> getOutputSize() const = 0;

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd) const = 0;
    virtual Eigen::MatrixXd backward(Eigen::MatrixXd input, Eigen::MatrixXd error) const = 0;
    virtual void update(Eigen::MatrixXd input, Eigen::MatrixXd error, double rate) = 0;
};