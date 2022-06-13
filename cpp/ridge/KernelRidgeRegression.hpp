#pragma once

#include <string>
#include <functional>

#include "Utils.hpp"

typedef std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)> Kernel;

struct regData {
    Eigen::VectorXd x;
    double y;
};

class KernelRidgeRegression {
    double minC, maxC, step, C{};
    Eigen::VectorXd alfa;
    std::vector<regData> tr;
    Kernel K;
    double offset{};

    void setOffset();
public:
    double calc(const Data &input) const;
    static double trainingError(const Eigen::MatrixXd &cache, const Eigen::VectorXd &target, const Eigen::VectorXd &t) ;
    [[nodiscard]] double error(const std::vector<Data> &input) const;
    void train(const std::vector<Data> &train, const std::vector<Data> &val);
    void save(const std::string &path) const;
    KernelRidgeRegression(Kernel K, const double &minC = .25, const double &maxC = 257, const double &step = 2);
};
