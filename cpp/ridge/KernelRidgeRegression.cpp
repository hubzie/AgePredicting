#include <fstream>

#include "KernelRidgeRegression.hpp"

std::ostream& operator<< (std::ostream& out, const regData& d) {
    return out << d.y << " " << d.x.transpose();
}

double KernelRidgeRegression::error(const std::vector<Data> &input) const {
    Eigen::VectorXd kap(tr.size());
    return std::transform_reduce(input.cbegin(), input.cend(), 0.0, std::plus<>(), [&](const Data &d) -> double {
        for (int i = 0; i < kap.size(); ++i)
            kap[i] = K(tr[i].x, d.x);
        return std::abs(alfa.dot(kap) - (double)d.y + offset);
    }) / (double)input.size();
}

double KernelRidgeRegression::calc(const Data &input) const {
    Eigen::VectorXd kap(tr.size());
    for (int i = 0; i < kap.size(); ++i)
        kap[i] = K(tr[i].x, input.x);
    return alfa.dot(kap) + offset;
}

double KernelRidgeRegression::trainingError(const Eigen::MatrixXd &cache, const Eigen::VectorXd &target, const Eigen::VectorXd &a) {
    return (a * cache - target).squaredNorm();
}

static std::pair<Eigen::MatrixXd, Eigen::VectorXd> toMatrix(const std::vector<Data> &X, const Kernel &K) {
    const int n = X.size();
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = K(X[i].x, X[j].x);
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i)
        y[i] = X[i].y;
    y.transposeInPlace();
    return std::make_pair(A, y);
}

void KernelRidgeRegression::setOffset() {
    offset = 0;
    for (auto &i : tr)
        offset += i.y;
    offset /= (double)tr.size();
    for (auto &i : tr)
        i.y -= offset;
//    std::cout << "OFFSET: " << offset << std::endl;
}

void KernelRidgeRegression::train(const std::vector<Data> &train, const std::vector<Data> &val) {
    tr.resize(train.size());
    for (size_t i = 0; i < tr.size(); ++i) {
        tr[i].x = train[i].x;
        tr[i].y = (double)train[i].y;
    }
    setOffset();

    Eigen::MatrixXd cache(train.size(), val.size());
    for (int i = 0; i < cache.rows(); ++i)
        for (int j = 0; j < cache.cols(); ++j)
            cache(i, j) = K(train[i].x, val[j].x);

    Eigen::VectorXd target(val.size());
    for (int i = 0; i < target.rows(); ++i)
        target[i] = val[i].y - offset;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(train.size(), train.size());
    auto [A, y] = toMatrix(train, K);
    C = minC;

    alfa = y * (A + C * I).inverse();

    for (C = minC * step; C < maxC; C *= step) {
        Eigen::VectorXd a = y * (A + C * I).inverse();
        if (trainingError(cache, target, alfa) > trainingError(cache, target, a))
            alfa = a;
    }
}

KernelRidgeRegression::KernelRidgeRegression(Kernel K, const double &minC, const double &maxC, const double &step) : minC(minC), maxC(maxC), step(step), K(std::move(K)) {}

void KernelRidgeRegression::save(const std::string &path) const {
    std::ofstream file(path);
    file << offset << '\n' << alfa << '\n';
    for (auto &i : tr)
        file << i << '\n';
    file.close();
}
