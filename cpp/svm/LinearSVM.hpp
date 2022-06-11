#pragma once

#include <Eigen/Dense>
#include <random>
#include <list>

#include "../model/Model.hpp"

class LinearSVM : public Model {
    Eigen::VectorXd w;
    std::list<int> unbound;
    double C, b;
    std::vector<double> a;
    std::mt19937 gen;

    [[nodiscard]] inline double place(const Data &input) const;
    [[nodiscard]] inline bool onBound(const double &a) const;
    bool update(const int &i1, const int &i2, const Data &d1, const Data &d2);
    bool examineExample(const int &i2, const std::vector<Data> &d2);

    void _train(const std::vector<Data> &training, const std::vector<Data> &validation) override;
    [[nodiscard]] int _call(const Data &input) const override;
    void _save(const std::string &filename) const override;
public:

    [[nodiscard]] inline const Eigen::VectorXd & getW() const {
        return w;
    }
    [[nodiscard]] inline const double & getB() const {
        return b;
    }
    explicit LinearSVM(const int &width = 48 * 48, const double &C = 50);
    ~LinearSVM() override = default;
};