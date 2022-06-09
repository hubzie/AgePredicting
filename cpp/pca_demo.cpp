#include<Eigen/Core>
#include<iostream>
#include<fstream>
using namespace std;

#include"Utils.hpp"
#include"PCA.hpp"

int main() {
    cerr << "Thread count: " << Eigen::nbThreads() << "\n";

    auto data = fromFile("../../data/age_gender.csv");
    data.resize(1000);
    PCA pca(data);

    for(auto& r : data)
        r.x = pca.transform(r.x);

    {
        ofstream demo("../../demo/pca.data");
        for (auto &r: data)
            demo << r.x.transpose() << "\n";
    }

    {
        ofstream demoV("../../demo/pca_v.data");
        demoV << pca.getV();
    }

    return 0;
}
