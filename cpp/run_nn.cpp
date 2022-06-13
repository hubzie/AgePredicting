#include"CNN.hpp"
#include"Utils.hpp"

#include<filesystem>
#include<fstream>
#include<functional>
#include<map>
#include<iostream>

using namespace std;

const int REPS = 10;
const double step = 0.1;

double evaluate(const CNN& model, const vector<Data>& data) {
    int acc = 0;
    for(auto& d : data)
        if(model.predict(d.x) == d.y) acc++;
    return (100.0 * acc) / data.size();
}

void saveScore(const string& path, const vector<double>& score) {
    ofstream file(path);
    assert(file.is_open());
    for(auto i : score) file << i << " ";
    file << "\n";
}

void filter(vector<Data>& data, function<int(int)> func) {
    for(auto& d : data)
        d.y = func(d.y);

    data.erase(remove_if(
            data.begin(),
            data.end(),
            [](auto d){ return d.y == -1; }
            ), data.end());
}



CNN* buildNN_1(int inputSize, int outputSize) {
    CNN* model = new CNN(make_pair(inputSize, 1));

    model->addLayer(new FullConnectedLayer(inputSize, 40));
    model->addLayer(new SigmoidLayer({40, 1}));

    model->addLayer(new FullConnectedLayer(40, outputSize));

    return model;
}

CNN* buildNN_2(int inputSize, int outputSize) {
    CNN* model = new CNN(make_pair(inputSize, 1));

    model->addLayer(new FullConnectedLayer(inputSize, 50));
    model->addLayer(new SigmoidLayer({50, 1}));

    model->addLayer(new FullConnectedLayer(50, 30));
    model->addLayer(new ReLuLayer({30, 1}));

    model->addLayer(new FullConnectedLayer(30, 10));
    model->addLayer(new SigmoidLayer({10, 1}));

    model->addLayer(new FullConnectedLayer(10, outputSize));

    return model;
}

CNN* buildCNN_1(int outputSize) {
    CNN* model = new CNN(make_pair(48*48, 1));

    model->addLayer(new ReshapeLayer({48 * 48, 1}, {48, 48}));

    model->addLayer(new ConvolutionalLayer({48, 48}, {5, 5}));
    model->addLayer(new AveragePoolingLayer({44, 44}, {4, 4}));
    model->addLayer(new ReLuLayer({11, 11}));

    model->addLayer(new ReshapeLayer({11, 11}, {11 * 11, 1}));

    model->addLayer(new FullConnectedLayer(11 * 11, 50));
    model->addLayer(new ReLuLayer({50, 1}));

    model->addLayer(new FullConnectedLayer(50, 20));
    model->addLayer(new SigmoidLayer({20, 1}));

    model->addLayer(new FullConnectedLayer(20, outputSize));

    return model;
}

CNN* buildCNN_2(int outputSize) {
    CNN* model = new CNN(make_pair(48*48, 1));

    model->addLayer(new ReshapeLayer({48 * 48, 1}, {48, 48}));

    model->addLayer(new ConvolutionalLayer({48, 48}, {5, 5}));
    model->addLayer(new SigmoidLayer({44, 44}));

    model->addLayer(new ConvolutionalLayer({44, 44}, {5, 5}));
    model->addLayer(new MaxPoolingLayer({40, 40}, {2, 2}));
    model->addLayer(new SigmoidLayer({20, 20}));

    model->addLayer(new ConvolutionalLayer({20, 20}, {5, 5}));
    model->addLayer(new AveragePoolingLayer({16, 16}, {2, 2}));
    model->addLayer(new ReLuLayer({8, 8}));

    model->addLayer(new ReshapeLayer({8, 8}, {8 * 8, 1}));

    model->addLayer(new FullConnectedLayer(8 * 8, 40));
    model->addLayer(new ReLuLayer({40, 1}));

    model->addLayer(new FullConnectedLayer(40, 20));
    model->addLayer(new SigmoidLayer({20, 1}));

    model->addLayer(new FullConnectedLayer(20, outputSize));

    return model;
}



void test_nn(const string& path, const string& data_path, function<int(int)> func, int outputSize) {
    cerr << "### TEST NN ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto d_train = fromParsedFile(data_path+"train.data");
    auto test = fromParsedFile(data_path+"test.data");

    filter(d_train, func);
    filter(test, func);

    int inputSize = d_train[0].x.size();

    vector<double> curve_train_nn1, curve_train_nn2;
    vector<double> curve_test_nn1, curve_test_nn2;

    CNN* best_nn1 = nullptr;
    CNN* best_nn2 = nullptr;

    for(double frac=step;frac<=1.0;frac+=step) {
        double score_nn1, score_nn2;
        score_nn1 = score_nn2 = numeric_limits<double>::min();

        delete best_nn1;
        delete best_nn2;
        best_nn1 = best_nn2 = nullptr;

        for (int i = 1; i <= REPS; i++) {
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);

            {
                auto nn1 = buildNN_1(inputSize, outputSize);
                nn1->train(train);
                double score = evaluate(*nn1, train);
                if (score > score_nn1) {
                    swap(nn1, best_nn1);
                    score_nn1 = score;
                }
                delete nn1;
            }

            {
                auto nn2 = buildNN_2(inputSize, outputSize);
                nn2->train(train);
                double score = evaluate(*nn2, train);
                if (score > score_nn2) {
                    swap(nn2, best_nn2);
                    score_nn2 = score;
                }
                delete nn2;
            }
        }

        cerr << "Tested frac = " << 100.0 * frac << "%" << endl;
        cerr << "Score NN1 = " << score_nn1 << ", NN2 = " << score_nn2 << endl;
        curve_train_nn1.push_back(score_nn1);
        curve_train_nn2.push_back(score_nn2);
        curve_test_nn1.push_back(evaluate(*best_nn1, test));
        curve_test_nn2.push_back(evaluate(*best_nn2, test));
    }

    filesystem::create_directories(path+"nn1/");
    best_nn1->save(path+"nn1/");
    saveScore(path+"nn1/train_curve.data", curve_train_nn1);
    saveScore(path+"nn1/test_curve.data", curve_test_nn1);

    filesystem::create_directories(path+"nn2/");
    best_nn2->save(path+"nn2/");
    saveScore(path+"nn2/train_curve.data", curve_train_nn2);
    saveScore(path+"nn2/test_curve.data", curve_test_nn2);

    delete best_nn1;
    delete best_nn2;
}



void test_cnn(const string& path, const string& data_path, function<int(int)> func, int outputSize) {
    cerr << "### TEST CNN ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto d_train = fromParsedFile(data_path+"train.data");
    auto test = fromParsedFile(data_path+"test.data");
    filter(d_train, func);
    filter(test, func);

    int inputSize = d_train[0].x.size();

    vector<double> curve_train_cnn1, curve_train_cnn2;
    vector<double> curve_test_cnn1, curve_test_cnn2;

    CNN* best_cnn1 = nullptr;
    CNN* best_cnn2 = nullptr;

    for(double frac=step;frac<=1.0;frac+=step) {
        double score_cnn1, score_cnn2;
        score_cnn1 = score_cnn2 = numeric_limits<double>::min();

        delete best_cnn1;
        delete best_cnn2;
        best_cnn1 = best_cnn2 = nullptr;

        for (int i = 1; i <= REPS; i++) {
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);

            {
                auto cnn1 = buildCNN_1(outputSize);
                cnn1->train(train);
                double score = evaluate(*cnn1, train);
                if (score > score_cnn1) {
                    swap(cnn1, best_cnn1);
                    score_cnn1 = score;
                }
                delete cnn1;
            }

            {
                auto cnn2 = buildCNN_2(outputSize);
                cnn2->train(train);
                double score = evaluate(*cnn2, train);
                if (score > score_cnn2) {
                    swap(cnn2, best_cnn2);
                    score_cnn2 = score;
                }
                delete cnn2;
            }
        }

        cerr << "Tested frac = " << 100.0 * frac << "%" << endl;
        cerr << "Score CNN1 = " << score_cnn1 << ", CNN2 = " << score_cnn2 << endl;
        curve_train_cnn1.push_back(score_cnn1);
        curve_train_cnn2.push_back(score_cnn2);
        curve_test_cnn1.push_back(evaluate(*best_cnn1, test));
        curve_test_cnn2.push_back(evaluate(*best_cnn2, test));
    }

    filesystem::create_directories(path+"cnn1/");
    best_cnn1->save(path+"cnn1/");
    saveScore(path+"cnn1/train_curve.data", curve_train_cnn1);
    saveScore(path+"cnn1/test_curve.data", curve_test_cnn1);

    filesystem::create_directories(path+"cnn2/");
    best_cnn2->save(path+"cnn2/");
    saveScore(path+"cnn2/train_curve.data", curve_train_cnn2);
    saveScore(path+"cnn2/test_curve.data", curve_test_cnn2);

    delete best_cnn1;
    delete best_cnn2;
}



int filter_2_50(int x) {
    if(x == 2) return 0;
    if(x == 50) return 1;
    return -1;
}

int filter_23_34(int x) {
    if(x == 23) return 0;
    if(x == 34) return 1;
    return -1;
}

int filter_bucket(int age) {
    if (age <= 3)
        return 0;
    else if (age <= 8)
        return 1;
    else if (age <= 15)
        return 2;
    else if (age <= 25)
        return 3;
    else if (age <= 40)
        return 4;
    else if (age <= 60)
        return 5;
    else if (age <= 80)
        return 6;
    return 7;
}

static map<int,int> RENUM = {
        { 1, 0 }, { 2, 1 }, { 3, 2 }, { 4, 3 }, { 5, 4 }, { 6, 5 }, { 7, 6 }, { 8, 7 }, { 9, 8 }, { 10, 9 }, { 11, 10 }, { 12, 11 }, { 13, 12 }, { 14, 13 }, { 15, 14 }, { 16, 15 }, { 17, 16 }, { 18, 17 }, { 19, 18 }, { 20, 19 }, { 21, 20 }, { 22, 21 }, { 23, 22 }, { 24, 23 }, { 25, 24 }, { 26, 25 }, { 27, 26 }, { 28, 27 }, { 29, 28 }, { 30, 29 }, { 31, 30 }, { 32, 31 }, { 33, 32 }, { 34, 33 }, { 35, 34 }, { 36, 35 }, { 37, 36 }, { 38, 37 }, { 39, 38 }, { 40, 39 }, { 41, 40 }, { 42, 41 }, { 43, 42 }, { 44, 43 }, { 45, 44 }, { 46, 45 }, { 47, 46 }, { 48, 47 }, { 49, 48 }, { 50, 49 }, { 51, 50 }, { 52, 51 }, { 53, 52 }, { 54, 53 }, { 55, 54 }, { 56, 55 }, { 57, 56 }, { 58, 57 }, { 59, 58 }, { 60, 59 }, { 61, 60 }, { 62, 61 }, { 63, 62 }, { 64, 63 }, { 65, 64 }, { 66, 65 }, { 67, 66 }, { 68, 67 }, { 69, 68 }, { 70, 69 }, { 71, 70 }, { 72, 71 }, { 73, 72 }, { 74, 73 }, { 75, 74 }, { 76, 75 }, { 77, 76 }, { 78, 77 }, { 79, 78 }, { 80, 79 }, { 81, 80 }, { 82, 81 }, { 83, 82 }, { 84, 83 }, { 85, 84 }, { 86, 85 }, { 87, 86 }, { 88, 87 }, { 89, 88 }, { 90, 89 }, { 91, 90 }, { 92, 91 }, { 93, 92 }, { 95, 93 }, { 96, 94 }, { 99, 95 }, { 100, 96 }, { 101, 97 }, { 105, 98 }, { 110, 99 }, { 115, 100 }, { 116, 101 }
};

int filter_all(int x) {
    return RENUM[x];
}

int main() {
    cerr << fixed << setprecision(3);

    const string path = "../../models/nn/";
    filesystem::create_directories(path);

    test_nn(path+"pca/2_50/", "../../data/pca_data/", filter_2_50, 2);
    test_nn(path+"kpca_500/2_50/", "../../data/kernel_pca_data_500/", filter_2_50, 2);
    test_nn(path+"kpca_1000/2_50/", "../../data/kernel_pca_data_1000/", filter_2_50, 2);
    test_nn(path+"kpca_2000/2_50/", "../../data/kernel_pca_data_2000/", filter_2_50, 2);
    test_cnn(path+"pure/2_50/", "../../data/pure_data/", filter_2_50, 2);

    test_nn(path+"pca/23_34/", "../../data/pca_data/", filter_23_34, 2);
    test_nn(path+"kpca_500/23_34/", "../../data/kernel_pca_data_500/", filter_23_34, 2);
    test_nn(path+"kpca_1000/23_34/", "../../data/kernel_pca_data_1000/", filter_23_34, 2);
    test_nn(path+"kpca_2000/23_34/", "../../data/kernel_pca_data_2000/", filter_23_34, 2);
    test_cnn(path+"pure/23_34/", "../../data/pure_data/", filter_2_50, 2);

    test_nn(path+"pca/bucket/", "../../data/pca_data/", filter_bucket, 8);
    test_nn(path+"kpca_500/bucket/", "../../data/kernel_pca_data_500/", filter_bucket, 8);
    test_nn(path+"kpca_1000/bucket/", "../../data/kernel_pca_data_1000/", filter_bucket, 8);
    test_nn(path+"kpca_2000/bucket/", "../../data/kernel_pca_data_2000/", filter_bucket, 8);
    test_cnn(path+"pure/bucket/", "../../data/pure_data/", filter_bucket, 8);

    test_nn(path+"pca/all/", "../../data/pca_data/", filter_all, 102);
    test_nn(path+"kpca_500/all/", "../../data/kernel_pca_data_500/", filter_all, 102);
    test_nn(path+"kpca_1000/all/", "../../data/kernel_pca_data_1000/", filter_all, 102);
    test_nn(path+"kpca_2000/all/", "../../data/kernel_pca_data_2000/", filter_all, 102);
    test_cnn(path+"pure/all/", "../../data/pure_data/", filter_all, 102);

    return 0;
}
