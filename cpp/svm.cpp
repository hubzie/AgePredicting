#include <iostream>
#include <chrono>

#include "svm/MultiClassLinearSVM.hpp"
#include "utils/Utils.hpp"

int main() {
    std::vector<Data> training = fromFile("../../data/age_gender.csv");
    std::cout << "Fetched data " << training.size() << std::endl;
    std::vector<Data> subset;
    for (Data &d : training) {
        if (d.y != 3 && d.y != 20 && d.y != 50 && d.y != 90)
            continue;
        subset.push_back(d);
        subset.back().x /= 256.0;
        if (d.y == 3)
            subset.back().y = 0;
        else if (d.y == 20)
            subset.back().y = 1;
        else if (d.y == 50)
            subset.back().y = 2;
        else if (d.y == 90)
            subset.back().y = 3;
    }
    std::cout << "Cropped to " << subset.size() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    MultiClassLinearSVM multi_svm(4);
    multi_svm.train(subset);
    multi_svm.save("../../out/multi_svm.params");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "After " << diff.count() << " Linear SVM achieved " << multi_svm.error(subset) << " training error\n";
}
