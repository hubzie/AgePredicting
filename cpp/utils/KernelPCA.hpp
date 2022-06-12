#pragma once

#include<Eigen/Dense>
#include<functional>
#include<vector>

#include"Utils.hpp"

using KernelFunc = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

class KernelPCA {
    static const std::string NAME;

    enum { NOT_INITIALIZED, INITIALIZED } status = NOT_INITIALIZED;

    Eigen::MatrixXd V;
    Eigen::MatrixXd base;
    KernelFunc kernel;

    Eigen::VectorXd apply(const Eigen::VectorXd&) const;

public:

    void fromFile(const std::string&, KernelFunc);
    void prepare(const std::vector<Data>&, KernelFunc, double compression);

    Eigen::VectorXd transform(const Eigen::VectorXd&) const;

    void save(const std::string&);
};