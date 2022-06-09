#include<Eigen/Core>
#include<iostream>
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

//    for(auto& r : data)
//        cout << r.x.transpose() << "\n";

    return 0;
}