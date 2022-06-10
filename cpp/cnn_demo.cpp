#include"CNN.hpp"
#include"Utils.hpp"

int main() {
    srand(time(nullptr));

    auto data = fromParsedFile("../../demo/pca.data");
    standardize(data, mean(data), deviation(data));

    CNN model({112, 1});
    model.addLayer(new FullConnectedLayer(112, 2));
//    model.addLayer(new FullConnectedLayer(112, 40));
//    model.addLayer(new FullConnectedLayer(40, 10));
//    model.addLayer(new FullConnectedLayer(10, 2));

    model.train(data);

    return 0;
}