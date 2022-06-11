#include"CNN.hpp"
#include"Utils.hpp"

#include<iostream>
#include<iomanip>

using namespace std;

int main() {
    cout << fixed << setprecision(3);
    srand(time(nullptr));

    auto data = fromParsedFile("../../demo/pca.data");
    standardize(data, mean(data), deviation(data));

    for(auto& d : data)
        d.y = (d.y == 2 ? 0 : 1);

    CNN model({111, 1});
    model.addLayer(new FullConnectedLayer(111, 40));
    model.addLayer(new SigmoidLayer(40));
    model.addLayer(new FullConnectedLayer(40, 8));
    model.addLayer(new SigmoidLayer(8));
    model.addLayer(new FullConnectedLayer(8, 2));
    model.addLayer(new SigmoidLayer(2));

    model.train(data);

    return 0;
}