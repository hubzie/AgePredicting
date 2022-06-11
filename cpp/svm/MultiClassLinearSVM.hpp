#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "Model.hpp"
#include "LinearSVM.hpp"
#include "Utils.hpp"

class MultiClassLinearSVM : public Model {
    const int classes, width;
    std::vector<LinearSVM> machines;
    std::vector<std::pair<int, int>> tree;

    void split(std::vector<Data> training, const std::vector<Data> &validation, const int &l, const int &r);

    void _train(const std::vector<Data> &training, const std::vector<Data> &validation) override;
    void _save(const std::string &filename) const override;
    [[nodiscard]] int _call(const Data &input) const override;
public:
    explicit MultiClassLinearSVM(const int &classes = 2, const int &width = 48*48);
};
