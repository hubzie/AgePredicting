#include"MultiClassKernelSVM.hpp"
#include"Utils.hpp"

#include<filesystem>
#include<iostream>
#include <omp.h>

using namespace std;

const int REPS = 5;
const double step = 0.1;

double gauss01(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    if (&a == &b)
        return 1;
    return exp(-(a - b).squaredNorm() * .1);
}

double gauss3(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    if (&a == &b)
        return 1;
    return exp(-(a - b).squaredNorm() * 3);
}

void saveScore(const string& path, const vector<double>& score) {
    ofstream file(path);
    assert(file.is_open());
    for(auto i : score) file << i << " ";
    file << "\n";
}

void test(const string& path, const string& data_path) {
    cerr << "### TEST Kernel SVM ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto data = fromParsedFile(data_path+"train.data");
    auto data2 = fromParsedFile(data_path+"test.data");
    data.insert(data.end(), data2.begin(), data2.end());

    std::vector<double> curve_train1, curve_val1, curve_test1, curve_train2, curve_val2, curve_test2;
    MultiClassKernelSVM *best1 = nullptr, *best2 = nullptr;

    for(double frac=step;frac<=1.0;frac+=step) {
        double score_svm1 = numeric_limits<double>::max(), score_train1, score_val1;
        double score_svm2 = numeric_limits<double>::max(), score_train2, score_val2;

        for (int i = 1; i <= REPS; i++) {
            shuffle(data);
            auto [d_train, rest] = split(data, 0.6);
            auto [val, test] = split(rest, 0.5);
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);

            {
                auto svm = new MultiClassKernelSVM(gauss3, 116, 1, 65, 2);
                svm->train(train, val);
                double score = 1 - svm->error(test);
                if (score < score_svm1) {
                    score_svm1 = score;
                    swap(best1, svm);
                    score_train1 = 1 - best1->error(train);
                    score_val1 = 1 - best1->error(val);
                }
                delete svm;
            }
            {
                auto svm = new MultiClassKernelSVM(gauss01, 116, 1, 65, 2);
                svm->train(train, val);
                double score = 1 - svm->error(test);
                if (score < score_svm2) {
                    score_svm2 = score;
                    swap(best2, svm);
                    score_train2 = 1 - best2->error(train);
                    score_val2 = 1 - best2->error(val);
                }
                delete svm;
            }
        }

        cerr << "Tested frac = " << 100.0 * frac << "%" << endl;
        cerr << "SVM Score for tau = 3: " << score_svm1 << ", tau = 0.1: " << score_svm2 << endl;
        curve_train1.push_back(score_train1);
        curve_val1.push_back(score_val1);
        curve_test1.push_back(score_svm1);
        curve_train2.push_back(score_train2);
        curve_val2.push_back(score_val2);
        curve_test2.push_back(score_svm2);
    }

    filesystem::create_directories(path+"k_svm/");
    best1->save(path+"k_svm/params3.out");
    saveScore(path+"k_svm/train_curve3.data", curve_train1);
    saveScore(path+"k_svm/test_curve3.data", curve_test1);
    saveScore(path+"k_svm/val_curve3.data", curve_val1);
    best2->save(path+"k_svm/params0.1.out");
    saveScore(path+"k_svm/train_curve0.1.data", curve_train2);
    saveScore(path+"k_svm/test_curve0.1.data", curve_test2);
    saveScore(path+"k_svm/val_curve0.1.data", curve_val2);

    delete best1;
    delete best2;
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