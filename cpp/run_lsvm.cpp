#include"MultiClassLinearSVM.hpp"
#include"Utils.hpp"

#include<filesystem>
#include<iostream>
#include <omp.h>

using namespace std;

const int REPS = 5;
const double step = 0.1;

void saveScore(const string& path, const vector<double>& score) {
    ofstream file(path);
    assert(file.is_open());
    for(auto i : score) file << i << " ";
    file << "\n";
}

void test(const string& path, const string& data_path) {
    cerr << "### TEST LINEAR SVM ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto data = fromParsedFile(data_path+"train.data");
    auto data2 = fromParsedFile(data_path+"test.data");
    data.insert(data.end(), data2.begin(), data2.end());

    int inputSize = data[0].x.size();

    std::vector<double> curve_train, curve_val, curve_test;
    MultiClassLinearSVM *best = nullptr;

    for(double frac=step;frac<=1.0;frac+=step) {
        double score_svm = numeric_limits<double>::max(), score_train, score_val;

        for (int i = 1; i <= REPS; i++) {
            shuffle(data);
            auto [d_train, rest] = split(data, 0.6);
            auto [val, test] = split(rest, 0.5);
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);

            {
                auto svm = new MultiClassLinearSVM(116, inputSize, 1, 129, 2);
                svm->train(train, val);
                double score = 1 - svm->error(test);
                if (score < score_svm) {
                    score_svm = score;
                    swap(best, svm);
                    score_train = 1 - best->error(train);
                    score_val = 1 - best->error(val);
                }
                delete svm;
            }
        }

        cerr << "Tested frac = " << 100.0 * frac << "%" << endl;
        cerr << "Score SVM = " << score_svm << endl;
        curve_train.push_back(score_train);
        curve_val.push_back(score_val);
        curve_test.push_back(score_svm);
    }

    filesystem::create_directories(path+"l_svm/");
    best->save(path+"l_svm/params.out");
    saveScore(path+"l_svm/train_curve.data", curve_train);
    saveScore(path+"l_svm/test_curve.data", curve_test);
    saveScore(path+"l_svm/val_curve.data", curve_val);

    delete best;
}



int main() {
    cerr << fixed << setprecision(3);

    const string path = "../../models/svm/";
    filesystem::create_directories(path);

    test(path+"pca/", "../../data/pca_data/");
    test(path+"kpca_500/", "../../data/kernel_pca_data_500/");
    test(path+"kpca_1000/", "../../data/kernel_pca_data_1000/");
    test(path+"kpca_2000/", "../../data/kernel_pca_data_2000/");

    return 0;
}