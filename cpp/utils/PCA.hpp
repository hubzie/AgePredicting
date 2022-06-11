#pragma once

#include<vector>

#include"Utils.hpp"

class PCA {
    Eigen::MatrixXd V;
    Eigen::VectorXd mean;

public:

    PCA(const std::vector<Data>&, double);

    Eigen::VectorXd transform(const Eigen::VectorXd&) const;
    Eigen::MatrixXd getV() const;
};
