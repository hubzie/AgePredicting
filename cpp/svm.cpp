#include <iostream>

#include "svm/LinearSVM.hpp"
#include "utils/Utils.hpp"

int main() {
    std::vector<Data> training = fromFile("../../data/age_gender.csv");
    std::cout << "Fetched data " << training.size() << std::endl;
    std::vector<Data> subset;
    for (Data &d : training) {
        if (d.y == 2) {
            subset.push_back(d);
            subset.back().y = -1;
            for (int i = 0; i < 48*48; ++i)
                subset.back().x(i) -= 128.0;
            subset.back().x /= 256.0;
        } else if (d.y == 50) {
            subset.push_back(d);
            for (int i = 0; i < 48*48; ++i)
                subset.back().x(i) -= 128.0;
            subset.back().y = 1;
            subset.back().x /= 256.0;
        }
    }
    std::cout << "Cropped to " << subset.size() << std::endl;

    LinearSVM svm;
    svm.train(subset);
    svm.save("../../out/svm.params");
    std::cout << svm.error(subset) << std::endl;
}
