#include"RegressionCNN.hpp"
#include"Utils.hpp"

#include<fstream>
#include<iostream>
#include<iomanip>
#include<map>
#include<vector>

using namespace std;

static void evaluate(const RegressionCNN& model, const vector<Data>& train, const vector<Data>& test) {
    cerr << "Evaluating..." << endl;
    {
        double score = 0.0;
        for(auto& d : train)
            score += abs(d.y - model.predict(d.x));
        cout << "Training set score = " << score / train.size() << "\n";
    }

    {
        double score = 0.0;
        for(auto& d : test)
            score += abs(d.y - model.predict(d.x));
        cout << "Test set score = " << score / test.size() << "\n";
    }
}

vector<Data> load_pca() {
    cerr << "Loading data..." << endl;

    auto data = fromParsedFile("../../demo/pca_data.data");
    standardize(data, mean(data), deviation(data));

    return data;
}

vector<Data> load_kpca() {
    cerr << "Loading data..." << endl;

    vector<Data> data;
    try {
        data = fromParsedFile("../../demo/kpca/reg_cnn.data");
    } catch(const exception&) {
        cout << "Prepare data..." << endl;

        data = fromParsedFile("../../demo/kpca/data.data");

        {
            cout << "Save for future runs..." << endl;
            ofstream file("../../demo/kpca/reg_cnn.data");
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

    auto [train, test] = split(data, 0.66);

    cerr << "Building CNN..." << endl;
    int inputSize = data[0].x.size();
    RegressionCNN model(make_pair(inputSize, 1));
    model.addLayer(new FullConnectedLayer(inputSize, 100));
    model.addLayer(new ReLuLayer({100, 1}));
    model.addLayer(new FullConnectedLayer(100, 30));
    model.addLayer(new SigmoidLayer({30, 1}));
    model.addLayer(new FullConnectedLayer(30, 1));

    cerr << "Training..." << endl;
    model.train(train);

    evaluate(model, train, test);
}

void cnn(bool loadBest) {
    cerr << "Loading data..." << endl;

    vector<Data> data;

    try {
        data = fromParsedFile("../../demo/reg_cnn.data");
    } catch(const exception&) {
        cout << "Prepare data..." << endl;

        data = fromFile("../../data/age_gender.csv");
        for (auto &d: data)
            for (int i = 0; i < d.x.size(); i++)
                d.x(i) /= 256.0;

        {
            cout << "Save for future runs..." << endl;
            ofstream file("../../demo/reg_cnn.data");
            for (auto &d: data)
                file << d << "\n";
        }
    }

    map<int,int> cnt;
    for(auto& d : data) cnt[d.y]++;

    cerr << "### DATASET ###\n";
    for(auto [k,v] : cnt)
        cerr << "\t" << k << " : " << v << "\n";
    cerr << "### DATASET ###\n" << flush;

    auto [train, test] = split(data, 0.66);

    if (loadBest) {
        cout << "Loading best..." << endl;
        RegressionCNN model("../../demo/reg_best_cnn/");
        evaluate(model, train, test);
    } else {
        cerr << "Building CNN..." << endl;
        RegressionCNN model(make_pair(48 * 48, 1));

        model.addLayer(new ReshapeLayer({48 * 48, 1}, {48, 48}));

        model.addLayer(new ConvolutionalLayer({48, 48}, {5, 5}));
        model.addLayer(new SigmoidLayer({44, 44}));

        model.addLayer(new ConvolutionalLayer({44, 44}, {5, 5}));
        model.addLayer(new MaxPoolingLayer({40, 40}, {2, 2}));
        model.addLayer(new SigmoidLayer({20, 20}));

        model.addLayer(new ConvolutionalLayer({20, 20}, {5, 5}));
        model.addLayer(new AveragePoolingLayer({16, 16}, {2, 2}));
        model.addLayer(new ReLuLayer({8, 8}));

        model.addLayer(new ReshapeLayer({8, 8}, {8 * 8, 1}));

        model.addLayer(new FullConnectedLayer(8 * 8, 40));
        model.addLayer(new ReLuLayer({40, 1}));

        model.addLayer(new FullConnectedLayer(40, 20));
        model.addLayer(new SigmoidLayer({20, 1}));

        model.addLayer(new FullConnectedLayer(20, 1));

        cerr << "Training..." << endl;
        model.train(train);

        evaluate(model, train, test);

        cerr << "Saving..." << endl;
        model.save("../../demo/reg_cnn_dir/");
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