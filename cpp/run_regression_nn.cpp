#include"RegressionCNN.hpp"
#include"Utils.hpp"

#include<filesystem>
#include<fstream>
#include<iostream>

using namespace std;

const int REPS = 50;
const double step = 0.1;

double evaluate(const RegressionCNN& model, const vector<Data>& data) {
    double score;
    for(auto& d : data)
        score += abs(model.predict(d.x) - d.y);
    return score / data.size();
}

void saveScore(const string& path, const vector<double>& score) {
    ofstream file(path);
    assert(file.is_open());
    for(auto i : score) file << i << " ";
    file << "\n";
}



RegressionCNN* buildRegressionNN_1(int inputSize) {
    RegressionCNN* model = new RegressionCNN(make_pair(inputSize, 1));

    model->addLayer(new FullConnectedLayer(inputSize, 40));
    model->addLayer(new SigmoidLayer({40, 1}));

    model->addLayer(new FullConnectedLayer(40, 1));

    return model;
}

RegressionCNN* buildRegressionNN_2(int inputSize) {
    RegressionCNN* model = new RegressionCNN(make_pair(inputSize, 1));

    model->addLayer(new FullConnectedLayer(inputSize, 50));
    model->addLayer(new SigmoidLayer({50, 1}));

    model->addLayer(new FullConnectedLayer(50, 30));
    model->addLayer(new ReLuLayer({30, 1}));

    model->addLayer(new FullConnectedLayer(30, 10));
    model->addLayer(new SigmoidLayer({10, 1}));

    model->addLayer(new FullConnectedLayer(10, 1));

    return model;
}

RegressionCNN* buildRegressionCNN_1() {
    RegressionCNN* model = new RegressionCNN(make_pair(48*48, 1));

    model->addLayer(new ReshapeLayer({48 * 48, 1}, {48, 48}));

    model->addLayer(new ConvolutionalLayer({48, 48}, {5, 5}));
    model->addLayer(new AveragePoolingLayer({44, 44}, {4, 4}));
    model->addLayer(new ReLuLayer({11, 11}));

    model->addLayer(new ReshapeLayer({11, 11}, {11 * 11, 1}));

    model->addLayer(new FullConnectedLayer(11 * 11, 50));
    model->addLayer(new ReLuLayer({50, 1}));

    model->addLayer(new FullConnectedLayer(50, 20));
    model->addLayer(new SigmoidLayer({20, 1}));

    model->addLayer(new FullConnectedLayer(20, 1));

    return model;
}

RegressionCNN* buildRegressionCNN_2() {
    RegressionCNN* model = new RegressionCNN(make_pair(48*48, 1));

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

    model->addLayer(new FullConnectedLayer(20, 1));

    return model;
}



void test_nn(const string& path, const string& data_path) {
    cerr << "### TEST NN ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto d_train = fromParsedFile(data_path+"train.data");
    auto test = fromParsedFile(data_path+"test.data");

    int inputSize = d_train[0].x.size();

    vector<double> curve_train_nn1, curve_train_nn2;
    vector<double> curve_test_nn1, curve_test_nn2;

    RegressionCNN* best_nn1 = nullptr;
    RegressionCNN* best_nn2 = nullptr;

    for(double frac=step;frac<=1.0;frac+=step) {
        double score_nn1, score_nn2;
        score_nn1 = score_nn2 = numeric_limits<double>::max();

        delete best_nn1;
        delete best_nn2;
        best_nn1 = best_nn2 = nullptr;

        for (int i = 1; i <= REPS; i++) {
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);

            {
                auto nn1 = buildRegressionNN_1(inputSize);
                nn1->train(train);
                double score = evaluate(*nn1, train);
                if (score < score_nn1) {
                    swap(nn1, best_nn1);
                    score_nn1 = score;
                }
                delete nn1;
            }

            {
                auto nn2 = buildRegressionNN_2(inputSize);
                nn2->train(train);
                double score = evaluate(*nn2, train);
                if (score < score_nn2) {
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

    filesystem::create_directories(path+"reg_nn1/");
    best_nn1->save(path+"reg_nn1/");
    saveScore(path+"reg_nn1/train_curve.data", curve_train_nn1);
    saveScore(path+"reg_nn1/test_curve.data", curve_test_nn1);

    filesystem::create_directories(path+"reg_nn2/");
    best_nn2->save(path+"reg_nn2/");
    saveScore(path+"reg_nn2/train_curve.data", curve_train_nn2);
    saveScore(path+"reg_nn2/test_curve.data", curve_test_nn2);

    delete best_nn1;
    delete best_nn2;
}



void test_cnn(const string& path, const string& data_path) {
    cerr << "### TEST CNN ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto d_train = fromParsedFile(data_path+"train.data");
    auto test = fromParsedFile(data_path+"test.data");

    int inputSize = d_train[0].x.size();

    vector<double> curve_train_cnn1, curve_train_cnn2;
    vector<double> curve_test_cnn1, curve_test_cnn2;

    RegressionCNN* best_cnn1 = nullptr;
    RegressionCNN* best_cnn2 = nullptr;

    for(double frac=step;frac<=1.0;frac+=step) {
        double score_cnn1, score_cnn2;
        score_cnn1 = score_cnn2 = numeric_limits<double>::max();

        delete best_cnn1;
        delete best_cnn2;
        best_cnn1 = best_cnn2 = nullptr;

        for (int i = 1; i <= REPS; i++) {
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);

            {
                auto cnn1 = buildRegressionCNN_1();
                cnn1->train(train);
                double score = evaluate(*cnn1, train);
                if (score < score_cnn1) {
                    swap(cnn1, best_cnn1);
                    score_cnn1 = score;
                }
                delete cnn1;
            }

            {
                auto cnn2 = buildRegressionCNN_2();
                cnn2->train(train);
                double score = evaluate(*cnn2, train);
                if (score < score_cnn2) {
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

    filesystem::create_directories(path+"reg_cnn1/");
    best_cnn1->save(path+"reg_cnn1/");
    saveScore(path+"reg_cnn1/train_curve.data", curve_train_cnn1);
    saveScore(path+"reg_cnn1/test_curve.data", curve_test_cnn1);

    filesystem::create_directories(path+"reg_cnn2/");
    best_cnn2->save(path+"reg_cnn2/");
    saveScore(path+"reg_cnn2/train_curve.data", curve_train_cnn2);
    saveScore(path+"reg_cnn2/test_curve.data", curve_test_cnn2);

    delete best_cnn1;
    delete best_cnn2;
}



int main() {
    cerr << fixed << setprecision(3);

    const string path = "../../models/regression_nn/";
    filesystem::create_directories(path);

    test_nn(path+"pca/", "../../data/pca_data/");
    test_nn(path+"kpca_500/", "../../data/kernel_pca_data_500/");
    test_nn(path+"kpca_1000/", "../../data/kernel_pca_data_1000/");
    test_nn(path+"kpca_2000/", "../../data/kernel_pca_data_2000/");

    return 0;
}