#include<Eigen/Core>
#include<iostream>
#include<filesystem>

#include"KernelPCA.hpp"
#include"PCA.hpp"
#include"Utils.hpp"

using namespace Eigen;
using namespace std;

double kernel(const VectorXd& a, const VectorXd& b) {
    if(&a == &b) return 1;
    return exp(- (a-b).squaredNorm() / (2.0 * 1000.0 * 1000.0));
}

int main() {
    cerr << "Thread count: " << nbThreads() << endl;

    cerr << "Reading data..." << endl;
    auto data = fromFile("../../data/age_gender.csv");

    cerr << "Shuffling..." << endl;
    shuffle(data);

    cerr << "Splitting data..." << endl;
    auto [fit, test] = split(data, 0.66);

    fit.resize(1000);

    cerr << "Preparing..." << endl;

    const string save_location = "../../demo/kpca/";
    const string kpca_save_location = save_location+"kernel_pca.data";
    const string pca_save_location = save_location+"pca.data";

    {
        KernelPCA kpca;
        if (filesystem::exists(kpca_save_location)) kpca.fromFile(kpca_save_location, kernel);
        else {
            kpca.prepare(fit, kernel, 0.9);
            kpca.save(kpca_save_location);
        }

        cerr << "Compressing..." << endl;
        for (auto &r: data)
            r.x = kpca.transform(r.x);
        for (auto &r: fit)
            r.x = kpca.transform(r.x);
    }

    {
        PCA pca;
        if (filesystem::exists(pca_save_location)) pca.fromFile(pca_save_location);
        else {
            pca.prepare(fit, 0.9);
            pca.save(pca_save_location);
        }
        for (auto &r: data)
            r.x = pca.transform(r.x);
        for (auto &r: fit)
            r.x = pca.transform(r.x);
    }

    save(save_location+"data.data", data);

    return 0;
}
