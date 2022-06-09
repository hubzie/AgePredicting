#pragma once

#include<vector>

#include"Utils.hpp"

const static double COMPRESSION = 0.95;

class PCA {
    Eigen::MatrixXd V;
    Eigen::VectorXd mean;

public:

    PCA(const std::vector<Data>&);

    Eigen::VectorXd transform(const Eigen::VectorXd&) const;
    Eigen::MatrixXd getV() const;
};
