#include<Eigen/Core>
#include<iostream>
#include<filesystem>

using namespace Eigen;
using namespace std;

#include"Utils.hpp"
#include"PCA.hpp"

int main() {
    cerr << "Thread count: " << nbThreads() << endl;

    cerr << "Reading data..." << endl;
    auto data = fromFile("../../data/age_gender.csv");

    cerr << "Shuffling..." << endl;
    shuffle(data);

    cerr << "Splitting data..." << endl;
    auto [fit, test] = split(data, 0.66);

    cerr << "Preparing..." << endl;
    PCA pca;

    const string pca_save_location = "../../demo/pca.data";

    if(filesystem::exists(pca_save_location))
        pca.fromFile(pca_save_location);
    else
        pca.prepare(fit, 0.95);

    cerr << "Compressing..." << endl;
    for(auto& r : data)
        r.x = pca.transform(r.x);

    save("../../demo/pca_data.data", data);
    pca.save(pca_save_location);

    return 0;
}
