#include <iostream>
#include <chrono>

#include "svm/MultiClassLinearSVM.hpp"
#include "svm/MultiClassKernelSVM.hpp"
#include "utils/Utils.hpp"

constexpr double GAMMA = 3;

double gauss(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    if (&a == &b)
        return 1;
    return exp(-(a - b).squaredNorm() * GAMMA);
}

int main() {
    auto data = fromParsedFile("../../demo/pca.data");
    standardize(data, mean(data), deviation(data));

    for(auto& d : data)
        d.y = (d.y == 2 ? 0 : 1);

    auto [train, test] = split(data, .6);

    auto start = std::chrono::high_resolution_clock::now();

//    MultiClassLinearSVM multi_svm(2, train[0].x.rows());
//    multi_svm.train(train);
//    multi_svm.save("../../out/multi_svm.params");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
//    std::cout << "After " << diff.count() << "s Linear SVM achieved " << multi_svm.error(test) << " training error\n";

    start = std::chrono::high_resolution_clock::now();

    MultiClassKernelSVM k_svm(gauss);
    k_svm.train(train);
    k_svm.save("../../out/k_svm.params");

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "After " << diff.count() << "s Kernel SVM achieved " << k_svm.error(test) << " training error\n";
}
