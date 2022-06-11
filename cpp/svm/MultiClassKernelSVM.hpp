#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "Model.hpp"
#include "KernelSVM.hpp"
#include "Utils.hpp"

class MultiClassKernelSVM : public Model {
    Kernel K;
    const int classes;
    std::vector<KernelSVM> machines;
    std::vector<std::pair<int, int>> tree;

    void split(std::vector<Data> training, const std::vector<Data> &validation, const int &l, const int &r);

    void _train(const std::vector<Data> &training, const std::vector<Data> &validation) override;
    void _save(const std::string &filename) const override;
    [[nodiscard]] int _call(const Data &input) const override;
public:
    explicit MultiClassKernelSVM(Kernel K, const int &classes = 2);
};
