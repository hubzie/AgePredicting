#include <iostream>
#include <chrono>

#include "svm/MultiClassLinearSVM.hpp"
#include "utils/Utils.hpp"

int main() {
    auto data = fromParsedFile("../../demo/pca.data");
    standardize(data, mean(data), deviation(data));

    for(auto& d : data)
        d.y = (d.y == 2 ? -1 : 1);

    auto start = std::chrono::high_resolution_clock::now();

    MultiClassLinearSVM multi_svm(2, 111);
    multi_svm.train(data);
    multi_svm.save("../../out/multi_svm.params");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "After " << diff.count() << "s Linear SVM achieved " << multi_svm.error(data) << " training error\n";
}
