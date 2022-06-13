#include"RidgeRegression.hpp"
#include "KernelRidgeRegression.hpp"
#include"Utils.hpp"

#include<filesystem>
#include <fstream>
#include<iostream>
#include <map>

using namespace std;

const int REPS = 5;
const double step = 0.1;

double gauss(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    if (&a == &b)
        return 1;
    return exp(-(a - b).squaredNorm() * 10);
}

void saveScore(const string& path, const vector<double>& score) {
    ofstream file(path);
    assert(file.is_open());
    for(auto i : score) file << i << " ";
    file << "\n";
}

void stdize(std::vector<Data> &train, std::vector<Data> &val, std::vector<Data> &test) {
    Eigen::VectorXd mi = mean(train), sigma = deviation(train);

    standardize(train, mi, sigma);
    standardize(val, mi, sigma);
    standardize(test, mi, sigma);
}

void addOne(std::vector<Data> &data) {
    for (auto &i : data) {
        i.x.conservativeResize(i.x.size() + 1);
        i.x[i.x.size() - 1] = 1;
//        std::cerr << i << std::endl;
    }
}

void test(const string& path, const string& data_path, const float &trf = 0.6) {
    cerr << "### TEST RIDGE REGRESSION ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto data = fromParsedFile(data_path+"train.data");
    auto data2 = fromParsedFile(data_path+"test.data");
    data.insert(data.end(), data2.begin(), data2.end());


    std::vector<double> curve_train, curve_val, curve_test;
    RidgeRegression *best = nullptr;
    double best_score = numeric_limits<double>::max();

    for(double frac=step;frac<=1.0;frac+=step) {
        double score_ridge = numeric_limits<double>::max(), score_train = 0, score_val = 0, score_test = 0;

        RidgeRegression *tmpBest = nullptr;

        for (int i = 1; i <= REPS; i++) {
            shuffle(data);
            auto [d_train, rest] = split(data, trf);
            auto [val, test] = split(rest, 0.5);
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);
            stdize(train, val, test);
            addOne(train);
            addOne(val);
            addOne(test);


            {
                auto model = new RidgeRegression(.25, 65, 2);
                model->train(train, val);
                double score = model->error(test);
                score_test += score;
                score_train += model->error(train);
                score_val += model->error(val);
                if (score < score_ridge) {
                    score_ridge = score;
                    swap(tmpBest, model);
                    if (score < best_score) {
                        swap(best, tmpBest);
                        best_score = score;
                    }
                }
                delete model;
            }
        }
        delete tmpBest;

        cerr << "Tested frac = " << 100.0 * frac << "%" << endl;
        cerr << "Score Ridge = " << score_ridge << endl;
        curve_train.push_back(score_train / REPS);
        curve_val.push_back(score_val / REPS);
        curve_test.push_back(score_test / REPS);
    }

    filesystem::create_directories(path+"lin/");
    best->save(path+"lin/params.out");
    saveScore(path+"lin/train_curve.data", curve_train);
    saveScore(path+"lin/test_curve.data", curve_test);
    saveScore(path+"lin/val_curve.data", curve_val);

    delete best;
}


void test_k(const string& path, const string& data_path, const float &trf = 0.4) {
    cerr << "### TEST KERNEL RIDGE REGRESSION ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto data = fromParsedFile(data_path+"train.data");
    auto data2 = fromParsedFile(data_path+"test.data");
    data.insert(data.end(), data2.begin(), data2.end());

    std::vector<double> curve_train, curve_val, curve_test;
    KernelRidgeRegression *best = nullptr;
    double best_score = numeric_limits<double>::max();

    for(double frac=step;frac<=0.31;frac+=step) {
        double score_ridge = numeric_limits<double>::max(), score_train = 0, score_val = 0, score_test = 0;

        KernelRidgeRegression *tmpBest = nullptr;

        for (int i = 1; i <= REPS; i++) {
            shuffle(data);
            auto [d_train, rest] = split(data, trf);
            auto [val, test] = split(rest, 0.5);
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);
            stdize(train, val, test);

            {
                auto model = new KernelRidgeRegression(gauss, .1, .5, 2);
                model->train(train, val);
                double score = model->error(test);
                score_test += score;
                score_train += model->error(train);
                score_val += model->error(val);
                if (score < score_ridge) {
                    score_ridge = score;
                    swap(tmpBest, model);
                    if (score < best_score) {
                        swap(best, tmpBest);
                        best_score = score;
                    }
                }
                delete model;
            }
        }
        delete tmpBest;

        cerr << "Tested frac = " << 100.0 * frac << "%" << endl;
        cerr << "Score Ridge = " << score_ridge << endl;
        curve_train.push_back(score_train / REPS);
        curve_val.push_back(score_val / REPS);
        curve_test.push_back(score_test / REPS);
    }

    filesystem::create_directories(path+"kernel/");
    best->save(path+"kernel/params.out");
    saveScore(path+"kernel/train_curve.data", curve_train);
    saveScore(path+"kernel/test_curve.data", curve_test);
    saveScore(path+"kernel/val_curve.data", curve_val);

    delete best;
}

int main() {
    cerr << fixed << setprecision(3);

    const string path = "../../models/ridge/";
    filesystem::create_directories(path);

    test_k(path+"pca/", "../../data/pca_data/");
//    test(path+"kpca_500/", "../../data/kernel_pca_data_500/");
//    test(path+"kpca_1000/", "../../data/kernel_pca_data_1000/");
//    test_k(path+"kpca_2000/", "../../data/kernel_pca_data_2000/");

    return 0;
}