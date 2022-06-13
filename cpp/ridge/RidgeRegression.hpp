#pragma once

#include <string>

#include "Utils.hpp"

class RidgeRegression {
    Eigen::VectorXd theta;
    double minC, maxC, step, C{};
public:
    double calc(const Data &input) const;
    static double trainingError(const std::vector<Data> &input, const Eigen::VectorXd &t) ;
    [[nodiscard]] double error(const std::vector<Data> &input) const;
    void train(const std::vector<Data> &train, const std::vector<Data> &val);
    void save(const std::string &path) const;
    RidgeRegression(const double &minC = .25, const double &maxC = 257, const double &step = 2);
};
