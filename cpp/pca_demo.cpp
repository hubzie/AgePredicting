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

    // Extract some data
/*    data.erase(remove_if(data.begin(),
                         data.end(),
                         [](const Data& x){ return x.y != 2 && x.y != 50; }),
               data.end());
*/
    cerr << "Splitting data..." << endl;
    auto [fit, test] = split(data, 0.66);

    cerr << "Preparing..." << endl;
    PCA pca(fit, 0.95);

    cerr << "Compressing..." << endl;
    for(auto& r : data)
        r.x = pca.transform(r.x);

    {
        ofstream demo("../../demo/pca.data");
        for (auto &r: data)
            demo << r << "\n";
    }

    {
        ofstream demoV("../../demo/pca_v.data");
        demoV << pca.getV();
    }

    return 0;
}
