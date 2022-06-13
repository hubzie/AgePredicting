#include"RegressionCNN.hpp"

#include<iomanip>
#include<iostream>
#include<map>

using namespace std;

void show(const string& model_path, const string& data_path) {
    RegressionCNN model(model_path);
    auto data = fromParsedFile(data_path);

    map<int,double> error;
    map<int,int> cnt;
    for(auto& d : data) {
        error[d.y] += abs(d.y - model.predict(d.x));
        cnt[d.y]++;
    }

    cout << model_path << "\n";
    for(auto [k,v] : error)
        cout << v / cnt[k] << " ";
    cout << "\n";
}

int main() {
    cout << fixed << setprecision(3);
/*
    show("../../models/regression_nn/pca/reg_nn1/", "../../data/pca_data/test.data");
    show("../../models/regression_nn/kpca_1000/reg_nn1/", "../../data/kernel_pca_data_1000/test.data");
    show("../../models/regression_nn/pure/reg_cnn1/", "../../data/pure_data/test.data");
*/
    auto data = fromParsedFile("../../data/pca_data/test.data");
    map<int,int> M;
    for(auto& d : data) M[d.y]++;

    for(auto& [k,v] : M) cout << k << ", ";
    cout << "\n";

    return 0;
}