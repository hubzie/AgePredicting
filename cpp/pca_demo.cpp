#include<Eigen/Core>
#include<iostream>
#include<fstream>
using namespace std;

#include"Utils.hpp"
#include"PCA.hpp"

int main() {
    cerr << "Thread count: " << Eigen::nbThreads() << endl;

    cerr << "Reading data..." << endl;
    auto data = fromFile("../../data/age_gender.csv");

    cerr << "Shuffling..." << endl;
    shuffle(data);

    cerr << "Splitting data..." << endl;
    auto [fit, test] = split(data, 0.66);
    fit.resize(5000);
    test.resize(100);

    cerr << "Compressing... " << endl;
    PCA pca(fit);

    for(auto& r : test)
        r.x = pca.transform(r.x);

    {
        ofstream demo("../../demo/pca.data");
        for (auto &r: test)
            demo << r.x.transpose() << "\n";
    }

    {
        ofstream demoV("../../demo/pca_v.data");
        demoV << pca.getV();
    }

    return 0;
}
