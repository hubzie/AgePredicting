#include"CNN.hpp"
#include"Utils.hpp"

#include<algorithm>
#include<fstream>
#include<iostream>
#include<iomanip>
#include<map>
#include<vector>

using namespace std;

const int CLASS_ZERO = 23;
const int CLASS_ONE = 34;

static void evaluate(const CNN& model, const vector<Data>& train, const vector<Data>& test) {
    cerr << "Evaluating..." << endl;
    {
        double acc = 0;
        for(auto& d : train)
            if(d.y == model.predict(d.x)) acc++;
        cout << "Training set accuracy = " << 100.0 * acc / train.size() << "%\n";
    }

    {
        double acc = 0;
        for(auto& d : test)
            if(d.y == model.predict(d.x)) acc++;
        cout << "Test set accuracy = " << 100.0 * acc / test.size() << "%\n";
    }
}

vector<Data> load_pca() {
    cerr << "Loading data..." << endl;

    auto data = fromParsedFile("../../demo/pca_data.data");
    standardize(data, mean(data), deviation(data));

    data.erase(remove_if(data.begin(),
                         data.end(),
                         [](const Data& x){ return x.y != CLASS_ZERO && x.y != CLASS_ONE; }),
               data.end());
    return data;
}

vector<Data> load_kpca() {
    cerr << "Loading data..." << endl;

    vector<Data> data;
    try {
        data = fromParsedFile("../../demo/kpca/cnn.data");
    } catch(const exception&) {
        cout << "Prepare data..." << endl;

        data = fromParsedFile("../../demo/kpca/data.data");

        data.erase(remove_if(data.begin(),
                             data.end(),
                             [](const Data &x) { return x.y != CLASS_ZERO && x.y != CLASS_ONE; }),
                   data.end());

        {
            cout << "Save for future runs..." << endl;
            ofstream file("../../demo/kpca/cnn.data");
            for (auto &d: data)
                file << d << "\n";
        }
    }
    return data;
}

void pure_nn(vector<Data> data) {
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
    CNN model(make_pair(inputSize, 1));
    model.addLayer(new FullConnectedLayer(inputSize, 200));
    model.addLayer(new ReLuLayer({200, 1}));
    model.addLayer(new FullConnectedLayer(200, 50));
    model.addLayer(new ReLuLayer({50, 1}));
    model.addLayer(new FullConnectedLayer(50, cnt.size()));
    model.addLayer(new SigmoidLayer({cnt.size(), 1}));

    cerr << "Training..." << endl;
    model.train(train);

    evaluate(model, train, test);
}

void cnn(bool loadBest) {
    cerr << "Loading data..." << endl;

    vector<Data> data;

    try {
        data = fromParsedFile("../../demo/cnn.data");
    } catch(const exception&) {
        cout << "Prepare data..." << endl;

        data = fromFile("../../data/age_gender.csv");
        for (auto &d: data)
            for (int i = 0; i < d.x.size(); i++)
                d.x(i) /= 256.0;

        data.erase(remove_if(data.begin(),
                             data.end(),
                             [](const Data &x) { return x.y != CLASS_ZERO && x.y != CLASS_ONE; }),
                   data.end());

        {
            cout << "Save for future runs..." << endl;
            ofstream file("../../demo/cnn.data");
            for (auto &d: data)
                file << d << "\n";
        }
    }

    map<int,int> cnt;
    for(auto& d : data) cnt[d.y]++;

    map<int,int> M;
    {
        int it = 0;
        for(auto [k,v] : cnt)
            M[k] = it++;
    }

    for(auto& d : data)
        d.y = M[d.y];

    cnt.clear();
    for(auto& d : data) cnt[d.y]++;

    cerr << "### DATASET ###\n";
    for(auto [k,v] : cnt)
        cerr << "\t" << k << " : " << v << "\n";
    cerr << "### DATASET ###\n" << flush;

    auto [train, test] = split(data, 0.66);

    if (loadBest) {
        cout << "Loading best..." << endl;
        CNN model("../../demo/best_cnn/");
        evaluate(model, train, test);
    } else {
        cerr << "Building CNN..." << endl;
        CNN model(make_pair(48 * 48, 1));

        model.addLayer(new ReshapeLayer({48 * 48, 1}, {48, 48}));

        model.addLayer(new ConvolutionalLayer({48, 48}, {5, 5}));
        model.addLayer(new SigmoidLayer({44, 44}));

        model.addLayer(new ConvolutionalLayer({44, 44}, {5, 5}));
        model.addLayer(new MaxPoolingLayer({40, 40}, {2, 2}));
        model.addLayer(new SigmoidLayer({20, 20}));

        model.addLayer(new ConvolutionalLayer({20, 20}, {5, 5}));
        model.addLayer(new AveragePoolingLayer({16, 16}, {2, 2}));
        model.addLayer(new SigmoidLayer({8, 8}));

        model.addLayer(new ReshapeLayer({8, 8}, {8 * 8, 1}));

        model.addLayer(new FullConnectedLayer(8 * 8, 30));
        model.addLayer(new ReLuLayer({30, 1}));

        model.addLayer(new FullConnectedLayer(30, 16));
        model.addLayer(new SigmoidLayer({16, 1}));

        model.addLayer(new FullConnectedLayer(16, M.size()));

        cerr << "Training..." << endl;
        model.train(train);

        evaluate(model, train, test);

        cerr << "Saving..." << endl;
        model.save("../../demo/cnn_dir/");
    }
}

int main() {
    cerr << fixed << setprecision(3);
    cout << fixed << setprecision(3);
    srand(time(nullptr));

    //pure_nn(load_pca());
    //pure_nn(load_kpca());
    cnn(false);
    //cnn(true);

    return 0;
}