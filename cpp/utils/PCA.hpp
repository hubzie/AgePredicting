#pragma once

#include<vector>

#include"Utils.hpp"

const static double COMPRESSION = 0.9;

class PCA {
    Eigen::MatrixXd V;

public:

    PCA(const std::vector<Data>&);

    Eigen::VectorXd transform(const Eigen::VectorXd&) const;
};
