#include <iostream>
#include <chrono>

#include "MultiClassLinearSVM.hpp"
#include "MultiClassKernelSVM.hpp"
#include "Utils.hpp"
#include "PCA.hpp"
#include "LinearSVM.hpp"
#include "KernelSVM.hpp"

constexpr double GAMMA = 2;

double gauss(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    if (&a == &b)
        return 1;
    return exp(-(a - b).squaredNorm() * GAMMA);
}

short ageGroup(const short &age) {
    if (age <= 3)
        return 0;
    else if (age <= 8)
        return 1;
    else if (age <= 15)
        return 2;
    else if (age <= 25)
        return 3;
    else if (age <= 40)
        return 4;
    else if (age <= 60)
        return 5;
    else if (age <= 80)
        return 6;
    return 7;
}

void assignGroups(std::vector<Data> &data) {
    for (auto &r : data)
        r.y = ageGroup(r.y);
}

std::vector<Data> readData() {
    std::cerr << "Reading data..." << std::endl;
    try {
        auto data = fromParsedFile("../../demo/svm.data");
        return data;
    } catch (const FileNotFound &e) {
        auto data = fromFile("../../data/age_gender.csv");

        std::cerr << "Shuffling..." << std::endl;
        shuffle(data);

        std::cerr << "Preparing..." << std::endl;
        PCA pca;
        pca.prepare(data, 0.95);

        std::cerr << "Compressing..." << std::endl;
        for (auto &r : data) {
            r.x = pca.transform(r.x);
        }
        standardize(data, mean(data), deviation(data));
        {
            std::ofstream demo("../../demo/svm.data");
            for (auto &r: data)
                demo << r << "\n";
        }
        return data;
    }

}

void load_lsvm() {
    auto data = readData();

    auto [train, test] = split(data, .1);

    MultiClassLinearSVM l_svm(ageGroup(120) + 1, train[0].x.rows());
    l_svm.load("../../out/multi_svm.params");

    std::cerr << "Linear SVM training accuracy: " << 1 - l_svm.error(train) << std::endl;
    std::cerr << "Linear SVM test accuracy: " << 1 - l_svm.error(test) << std::endl;
    std::cerr << "Linear SVM training average distance: " << l_svm.distance(train) << std::endl;
    std::cerr << "Linear SVM test average distance: " << l_svm.distance(test) << std::endl;
}

void lsvm() {
    auto data = readData();
//    std::vector<int> groups = equalGroups(data);
//    assignGroups(data);
//    for (auto &r : data)
//        r.y = (short)(std::lower_bound(groups.begin(), groups.end(), r.y) - groups.begin());
    for (auto &r : data)
        --r.y;
    auto [train, data2] = split(data, .3);
    auto [val, test] = split(data2, .5);

    auto start = std::chrono::high_resolution_clock::now();

//    MultiClassLinearSVM l_svm(ageGroup(120) + 1, train[0].x.rows());
    MultiClassLinearSVM l_svm(116, train[0].x.rows(), .25, pow(2, 8), 2);
    l_svm.train(train, val);
    l_svm.save("../../out/l_svm.params");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cerr << "Linear SVM took " << diff.count() << "s to train" << std::endl;
    std::cerr << "Linear SVM training accuracy: " << 1 - l_svm.error(train) << std::endl;
    std::cerr << "Linear SVM test accuracy: " << 1 - l_svm.error(test) << std::endl;
    std::cerr << "Linear SVM training average distance: " << l_svm.distance(train) << std::endl;
    std::cerr << "Linear SVM test average distance: " << l_svm.distance(test) << std::endl;
}

void ksvm() {
    auto data = readData();
    std::vector<int> groups = equalGroups(data);
//    assignGroups(data);
    for (auto &r : data)
        r.y = (short)(std::lower_bound(groups.begin(), groups.end(), r.y) - groups.begin());
    auto [train, test] = split(data, .1);

    auto start = std::chrono::high_resolution_clock::now();

    MultiClassKernelSVM k_svm(gauss, ageGroup(120) + 1);
    k_svm.train(train);
    k_svm.save("../../out/k_svm.params");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cerr << "Kernel SVM took " << diff.count() << "s to train" << std::endl;
    std::cerr << "Kernel SVM training accuracy: " << 1 - k_svm.error(train) << std::endl;
    std::cerr << "Kernel SVM test accuracy: " << 1 - k_svm.error(test) << std::endl;
    std::cerr << "Kernel SVM training average distance: " << k_svm.distance(train) << std::endl;
    std::cerr << "kernel SVM test average distance: " << k_svm.distance(test) << std::endl;
}

int main() {
    lsvm();
//    load_lsvm();
//    ksvm();
}
