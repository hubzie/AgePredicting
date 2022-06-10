#pragma once

#include<Eigen/Dense>

class Layer {
public:
    virtual std::pair<int,int> getInputSize() const = 0;
    virtual std::pair<int,int> getOutputSize() const = 0;

    virtual Eigen::MatrixXd process(Eigen::MatrixXd) const = 0;
    virtual Eigen::MatrixXd improve(Eigen::MatrixXd input, Eigen::MatrixXd output, double step) = 0;
};