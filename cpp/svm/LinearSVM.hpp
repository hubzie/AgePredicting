#pragma once

#include <Eigen/Dense>
#include <random>
#include <list>

#include "../model/Model.hpp"

class LinearSVM : public Model {
    Eigen::VectorXd w, _w;
    std::list<std::pair<int, double>> unbound;
    double C{}, b, _b, minC, maxC, step;
    std::vector<double> a;
    std::mt19937 gen;

    inline double bestError(const std::vector<Data> &input) const;
    [[nodiscard]] inline double place(const Data &input) const;
    [[nodiscard]] inline bool onBound(const double &a) const;
    bool update(const int &i1, const int &i2, const double &E1, const double &E2, const std::vector<Data> &training);
    bool examineExample(const int &i2, const double &E2, const std::vector<Data> &training);
    void refreshCache(const std::vector<Data> &training);
    void run(const std::vector<Data> &training);
    void _train(const std::vector<Data> &training, const std::vector<Data> &validation) override;
    [[nodiscard]] int _call(const Data &input) const override;
    void _save(const std::string &filename) const override;
    void _load(std::ifstream &filename) override;
public:

    [[nodiscard]] inline const Eigen::VectorXd & getW() const {
        return w;
    }
    [[nodiscard]] inline const double & getB() const {
        return b;
    }
    explicit LinearSVM(const int &width = 48 * 48, const double &minC = 1, const double &maxC = 100, const double &step = 1);
    ~LinearSVM() override = default;
};