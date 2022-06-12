#include <fstream>

#include "MultiClassKernelSVM.hpp"

void MultiClassKernelSVM::split(std::vector<Data> training, const std::vector<Data> &validation, const int &l, const int &r) {
    training.erase(std::remove_if(training.begin(), training.end(), [&](const Data &x) -> bool {
        return x.y < l || x.y >= r;
    }), training.end());

    const int m = (l + r) >> 1;
    if (r - l <= 3) {
        if (r - l == 3)
            split(training, validation, m, r);
        tree.emplace_back(-1 - l, r - l == 3 ? tree.size() - 1 : -r);

        std::for_each(training.begin(), training.end(), [&](Data &x) -> void {
            x.y = x.y < m ? -1 : 1;
        });
        machines.emplace_back(K, minC, maxC, step);
        machines.back().train(training, validation);
        return;
    }

    split(training, validation, l, m);
    const int leftChild = (int)tree.size() - 1;
    split(training, validation, m, r);
    tree.emplace_back(leftChild, tree.size() - 1);

    std::for_each(training.begin(), training.end(), [&](Data &x) -> void {
        x.y = x.y < m ? -1 : 1;
    });
    machines.emplace_back(K, minC, maxC, step);
    machines.back().train(training, validation);
}

void MultiClassKernelSVM::_train(const std::vector<Data> &training, const std::vector<Data> &validation) {
    split(training, validation, 0, classes);
}

void MultiClassKernelSVM::_save(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file.is_open())
        throw FileNotFound();

    file << classes << ' ' << tree.size() << '\n';
    for (auto &[x, y] : tree)
        file << x << ' ' << y << '\n';
    for (const KernelSVM &svm : machines) {
        for (auto &c: svm.getA())
            file << c << ' ';
        file << svm.getB() << '\n';
    }
    file.close();
}

void MultiClassKernelSVM::_load(std::ifstream &file) {
    if (!file.is_open())
        throw FileNotFound();

    int size;

    file >> classes >> size;
    tree.resize(size);
    for (auto &[x, y] : tree)
        file >> x >> y;
    machines.clear();
    std::string line;
    std::getline(file, line);
    for (int i = 0; i < tree.size(); ++i) {
        machines.emplace_back(K, minC, maxC, step);
        machines.back().load(file);
    }
}

int MultiClassKernelSVM::_call(const Data &input) const {
    int root = (int)tree.size() - 1;
    while (root >= 0)
        root = machines[root](input) < 0 ? tree[root].first : tree[root].second;
    return -1 - root;
}

MultiClassKernelSVM::MultiClassKernelSVM(Kernel K, const int &classes, const double &minC, const double &maxC, const double &step) : K(std::move(K)), classes(classes), minC(minC), maxC(maxC), step(step) {

}
