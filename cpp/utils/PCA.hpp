#pragma once

#include<vector>

#include"Utils.hpp"

class PCA {
    static const std::string NAME;

    enum { NOT_INITIALIZED, INITIALIZED } status = NOT_INITIALIZED;

    Eigen::MatrixXd V;
    Eigen::VectorXd mean;

public:

    void fromFile(const std::string&);
    void prepare(const std::vector<Data>&, double compression);

    Eigen::VectorXd transform(const Eigen::VectorXd&) const;

    void save(const std::string&);
};

const std::string PCA::NAME = "PCA_SAVE_FILE";