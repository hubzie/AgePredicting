#include"CNN.hpp"
#include"Utils.hpp"

#include<algorithm>
#include<iostream>
#include<iomanip>
#include<map>
#include<vector>

using namespace std;

void pure_nn() {
    cerr << "Loading data..." << endl;
    auto data = fromParsedFile("../../demo/pca.data");
    standardize(data, mean(data), deviation(data));

    data.erase(remove_if(data.begin(),
                         data.end(),
                         [](const Data& x){ return x.y != 10 && x.y != 30; }),
               data.end());

    map<int,int> cnt;
    for(auto& d : data) cnt[d.y]++;

    cerr << "### DATASET ###\n";
    for(auto [k,v] : cnt)
        cerr << "\t" << k << " : " << v << "\n";
    cerr << "### DATASET ###\n" << flush;

    map<int,int> M;
    {
        int it = 0;
        for(auto [k,v] : cnt) M[k] = it++;
    }

    for(auto& d : data)
        d.y = M[d.y];

    auto [train, test] = split(data, 0.66);

    cerr << "Building CNN..." << endl;
    int inputSize = data[0].x.size();
    CNN model({inputSize, 1});
    model.addLayer(new FullConnectedLayer(inputSize, 50));
    model.addLayer(new ReLuLayer({50, 1}));
    model.addLayer(new FullConnectedLayer(50, cnt.size()));
    model.addLayer(new SigmoidLayer({cnt.size(), 1}));

    cerr << "Training..." << endl;
    model.train(train);

    cerr << "Evaluating..." << endl;
    {
        double acc = 0;
        for(auto& d : train)
            if(d.y == model.predict(d.x)) acc++;
        cerr << "Training set accuracy = " << 100.0 * acc / train.size() << "%\n";
    }

    {
        double acc = 0;
        for(auto& d : test)
            if(d.y == model.predict(d.x)) acc++;
        cerr << "Test set accuracy = " << 100.0 * acc / test.size() << "%\n";
    }
}

void cnn() {
    cerr << "Loading data..." << endl;
    auto data = fromFile("../../data/age_gender.csv");
    for(auto& d : data)
        for(int i=0;i<d.x.size();i++)
            d.x(i) /= 256.0;

    data.erase(remove_if(data.begin(),
                         data.end(),
                         [](const Data& x){ return x.y != 10 && x.y != 30; }),
               data.end());

    map<int,int> cnt;
    for(auto& d : data) cnt[d.y]++;

    cerr << "### DATASET ###\n";
    for(auto [k,v] : cnt)
        cerr << "\t" << k << " : " << v << "\n";
    cerr << "### DATASET ###\n" << flush;

    map<int,int> M;
    {
        int it = 0;
        for(auto [k,v] : cnt) M[k] = it++;
    }

    for(auto& d : data)
        d.y = M[d.y];

    auto [train, test] = split(data, 0.66);

    cerr << "Building CNN..." << endl;
    CNN model({48*48, 1});
    model.addLayer(new ReshapeLayer({48*48, 1}, {48, 48}));
    model.addLayer(new MaxPoolingLayer({48, 48}, {2, 2}));
    model.addLayer(new SigmoidLayer({24, 24}));
    model.addLayer(new AveragePoolingLayer({24, 24}, {2, 2}));
    model.addLayer(new SigmoidLayer({12, 12}));
    model.addLayer(new ReshapeLayer({12, 12}, {12*12, 1}));
    model.addLayer(new FullConnectedLayer(12*12, cnt.size()));
    model.addLayer(new SigmoidLayer({cnt.size(), 1}));

    cerr << "Training..." << endl;
    model.train(train);

    cerr << "Evaluating..." << endl;
    {
        double acc = 0;
        for(auto& d : train)
            if(d.y == model.predict(d.x)) acc++;
        cerr << "Training set accuracy = " << 100.0 * acc / train.size() << "%\n";
    }

    {
        double acc = 0;
        for(auto& d : test)
            if(d.y == model.predict(d.x)) acc++;
        cerr << "Test set accuracy = " << 100.0 * acc / test.size() << "%\n";
    }
}

int main() {
    cerr << fixed << setprecision(3);
    srand(time(nullptr));

    //pure_nn();
    cnn();

    return 0;
}