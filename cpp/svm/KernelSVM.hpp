#pragma once

#include <Eigen/Dense>
#include <random>
#include <list>
#include <functional>

#include "Model.hpp"

typedef std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)> Kernel;

class KernelSVM : public Model {
    const Kernel K;
    double C, b;
    std::mt19937 gen;
    std::vector<double> a;
    std::list<int> non_zero;
    std::list<std::pair<int, double>> unbound;
    std::vector<Data> X;
    int zeros;

    inline void clean();
    [[nodiscard]] inline double place(const Data &input) const;
    [[nodiscard]] inline bool onBound(const double &a) const;
    bool update(const int &i1, const int &i2, const double &E1, const double &E2, const std::vector<Data> &training);
    bool examineExample(const int &i2, const double &E2, const std::vector<Data> &training);
    void refreshCache(const std::vector<Data> &training);

    void _train(const std::vector<Data> &training, const std::vector<Data> &validation) override;
    [[nodiscard]] int _call(const Data &input) const override;
    void _save(const std::string &filename) const override;
    void _load(std::ifstream &file) override;

public:
    [[nodiscard]] inline const std::vector<double> & getA() const {
        return a;
    }
    [[nodiscard]] inline const double & getB() const {
        return b;
    }

    explicit KernelSVM(Kernel K, const double &C = 1);
    ~KernelSVM() override = default;
};
