#include"Utils.hpp"
#include"PCA.hpp"
#include"KernelPCA.hpp"

#include<filesystem>
#include<iostream>
#include<vector>

using namespace Eigen;
using namespace std;

double kernel(double tau, const VectorXd& a, const VectorXd& b) {
    if(&a == &b) return 1;
    return exp(- (a-b).squaredNorm() / (2.0 * tau * tau));
}

void preparePureData(vector<Data> train, vector<Data> test) {
    cerr << "Save pure data..." << endl;
    const string path = "../../data/pure_data/";
    filesystem::create_directories(path);
    save(path + "train.data", train);
    save(path + "test.data", test);
}

void preparePCAData(vector<Data> train, vector<Data> test) {
    cerr << "Prepare PCA..." << endl;

    PCA pca;
    pca.prepare(train, 0.95);

    cerr << "Transforming..." << endl;
    for(auto& d : train) d.x = pca.transform(d.x);
    for(auto& d : test) d.x = pca.transform(d.x);

    cerr << "Save PCA data..." << endl;
    const string path = "../../data/pca_data/";
    filesystem::create_directories(path);
    save(path + "train.data", train);
    save(path + "test.data", test);
}

void prepareKernelPCAData(vector<Data> train, vector<Data> test, double tau, double frac) {
    cerr << "Prepare kernel PCA with tau = " << tau << "..." << endl;

    auto train_part = train;
    train_part.resize(frac * train.size());

    KernelPCA pca;
    pca.prepare(train_part, [tau](auto a, auto b) { return kernel(tau, a, b); }, 0.95);

    cerr << "Transforming..." << endl;
    for(auto& d : train) d.x = pca.transform(d.x);
    for(auto& d : test) d.x = pca.transform(d.x);

    cerr << "Save kernel PCA data..." << endl;
    const string path = "../../data/kernel_pca_data_"+to_string((int) tau)+"/";
    filesystem::create_directories(path);
    save(path + "train.data", train);
    save(path + "test.data", test);
}

int main() {
    cerr << fixed << setprecision(3);
    cerr << "Load data..." << endl;

    auto data = fromFile("../../data/age_gender.csv");
    shuffle(data);
    auto [train, test] = split(data, 0.66);

    preparePureData(train, test);
    preparePCAData(train, test);
    prepareKernelPCAData(train, test, 500.0, 0.1);
    prepareKernelPCAData(train, test, 1000.0, 0.1);
    prepareKernelPCAData(train, test, 2000.0, 0.1);

    return 0;
}