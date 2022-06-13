#include <fstream>

#include "RidgeRegression.hpp"

double RidgeRegression::error(const std::vector<Data> &input) const {
    return std::transform_reduce(input.cbegin(), input.cend(), 0.0, std::plus<>(), [&](const Data &d) -> double {
        return std::abs(theta.dot(d.x) - d.y);
    }) / (double)input.size();
}

double RidgeRegression::trainingError(const std::vector<Data> &input, const Eigen::VectorXd &t) {
    return std::transform_reduce(input.cbegin(), input.cend(), 0.0, std::plus<>(), [&](const Data &d) -> double {
        const double save = t.dot(d.x) - d.y;
        return save * save;
    });
}


static std::pair<Eigen::MatrixXd, Eigen::VectorXd> toMatrix(const std::vector<Data> &X) {
    const int n = X.size(), m = X[0].x.size();
    Eigen::MatrixXd A(n, m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            A(i, j) = X[i].x[j];
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i)
        y[i] = X[i].y;
    return std::make_pair(A, y);
}

void RidgeRegression::train(const std::vector<Data> &train, const std::vector<Data> &val) {
    auto [X, y] = toMatrix(train);
    C = minC;
    Eigen::MatrixXd A = X.transpose() * X, I = Eigen::MatrixXd::Identity(X.cols(), X.cols());
    X.transposeInPlace();
    X *= y;
    I(I.rows() - 1, I.cols() - 1) = 0;

    theta = (A + C * I).inverse() * X;

    for (C = minC * step; C < maxC; C *= step) {
        Eigen::VectorXd t = (A + C * I).inverse() * X;
        if (trainingError(val, theta) > trainingError(val, t))
            theta = t;
    }
}

RidgeRegression::RidgeRegression(const double &minC, const double &maxC, const double &step) : minC(minC), maxC(maxC), step(step) {}

void RidgeRegression::save(const std::string &path) const {
    std::ofstream file(path);
    file << theta;
    file.close();
}
