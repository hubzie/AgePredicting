#include"MultiClassKernelSVM.hpp"
#include"Utils.hpp"

#include<filesystem>
#include<iostream>
#include <map>

using namespace std;

const int REPS = 3;
const double step = 0.2;

void filter(vector<Data>& data, function<int(int)> func) {
    for(auto& d : data)
        d.y = func(d.y);

    data.erase(remove_if(
            data.begin(),
            data.end(),
            [](auto d){ return d.y == -1; }
    ), data.end());
}

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

static map<int,int> RENUM = {
        { 1, 0 }, { 2, 1 }, { 3, 2 }, { 4, 3 }, { 5, 4 }, { 6, 5 }, { 7, 6 }, { 8, 7 }, { 9, 8 }, { 10, 9 }, { 11, 10 }, { 12, 11 }, { 13, 12 }, { 14, 13 }, { 15, 14 }, { 16, 15 }, { 17, 16 }, { 18, 17 }, { 19, 18 }, { 20, 19 }, { 21, 20 }, { 22, 21 }, { 23, 22 }, { 24, 23 }, { 25, 24 }, { 26, 25 }, { 27, 26 }, { 28, 27 }, { 29, 28 }, { 30, 29 }, { 31, 30 }, { 32, 31 }, { 33, 32 }, { 34, 33 }, { 35, 34 }, { 36, 35 }, { 37, 36 }, { 38, 37 }, { 39, 38 }, { 40, 39 }, { 41, 40 }, { 42, 41 }, { 43, 42 }, { 44, 43 }, { 45, 44 }, { 46, 45 }, { 47, 46 }, { 48, 47 }, { 49, 48 }, { 50, 49 }, { 51, 50 }, { 52, 51 }, { 53, 52 }, { 54, 53 }, { 55, 54 }, { 56, 55 }, { 57, 56 }, { 58, 57 }, { 59, 58 }, { 60, 59 }, { 61, 60 }, { 62, 61 }, { 63, 62 }, { 64, 63 }, { 65, 64 }, { 66, 65 }, { 67, 66 }, { 68, 67 }, { 69, 68 }, { 70, 69 }, { 71, 70 }, { 72, 71 }, { 73, 72 }, { 74, 73 }, { 75, 74 }, { 76, 75 }, { 77, 76 }, { 78, 77 }, { 79, 78 }, { 80, 79 }, { 81, 80 }, { 82, 81 }, { 83, 82 }, { 84, 83 }, { 85, 84 }, { 86, 85 }, { 87, 86 }, { 88, 87 }, { 89, 88 }, { 90, 89 }, { 91, 90 }, { 92, 91 }, { 93, 92 }, { 95, 93 }, { 96, 94 }, { 99, 95 }, { 100, 96 }, { 101, 97 }, { 105, 98 }, { 110, 99 }, { 115, 100 }, { 116, 101 }
};

int filter_all(int x) {
    return RENUM[x];
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

void stdize(std::vector<Data> &train, std::vector<Data> &val, std::vector<Data> &test) {
    Eigen::VectorXd mi = mean(train), sigma = deviation(train);

    standardize(train, mi, sigma);
    standardize(val, mi, sigma);
    standardize(test, mi, sigma);
}

void test(const string& path, const string& data_path, const function<int(int)> &func, const int &classes, const float &trf = 0.6) {
    cerr << "### TEST Kernel SVM ###" << endl;
    cerr << "Dataset " << data_path << endl;

    cerr << "Loading data..." << endl;
    auto data = fromParsedFile(data_path+"train.data");
    auto data2 = fromParsedFile(data_path+"test.data");
    data.insert(data.end(), data2.begin(), data2.end());
    filter(data, func);

    std::vector<double> curve_train1, curve_val1, curve_test1, curve_train2, curve_val2, curve_test2;
    MultiClassKernelSVM *best1 = nullptr, *best2 = nullptr;

    for(double frac=step;frac<=1.0;frac+=step) {
        double score_svm1 = numeric_limits<double>::min(), score_train1 = 0, score_val1 = 0, score_test1 = 0;
        double score_svm2 = numeric_limits<double>::min(), score_train2 = 0, score_val2 = 0, score_test2 = 0;

        MultiClassKernelSVM *tmpBest1 = nullptr, *tmpBest2 = nullptr;
        double best_score1 = numeric_limits<double>::min(), best_score2 = numeric_limits<double>::min();

        for (int i = 1; i <= REPS; i++) {
            shuffle(data);
            auto [d_train, rest] = split(data, trf);
            auto [rest2, junk] = split(rest, 0.3);
            auto [val, test] = split(rest2, 0.5);
            cerr << "Iteration #" << i << " - Testing frac = " << 100.0 * frac << "%" << endl;
            vector<Data> train(d_train.begin(), d_train.begin() + d_train.size() * frac);
            stdize(train, val, test);

            {
                auto svm = new MultiClassKernelSVM(gauss3, classes, .25, 65, 2);
                svm->train(train, val);
                double score = 1 - svm->error(test);
                score_test1 += score;
                score_train1 += 1 - svm->error(train);
                score_val1 += 1 - svm->error(val);
                if (score > score_svm1) {
                    score_svm1 = score;
                    swap(tmpBest1, svm);
                    if (score > best_score1) {
                        swap(best1, tmpBest1);
                        best_score1 = score;
                    }
                }
                delete svm;
            }
            {
                auto svm = new MultiClassKernelSVM(gauss01, classes, .25, 65, 2);
                svm->train(train, val);
                double score = 1 - svm->error(test);
                score_test2 += score;
                score_train2 += 1 - svm->error(train);
                score_val2 += 1 - svm->error(val);
                if (score > score_svm2) {
                    score_svm2 = score;
                    swap(tmpBest2, svm);
                    if (score > best_score2) {
                        swap(best2, tmpBest2);
                        best_score2 = score;
                    }
                }
                delete svm;
            }
        }
        delete tmpBest1;
        delete tmpBest2;

        cerr << "Tested frac = " << 100.0 * frac << "%" << endl;
        cerr << "SVM Score for gamma = 3: " << score_svm1 << ", gamma = 0.1: " << score_svm2 << endl;
        curve_train1.push_back(score_train1 / REPS);
        curve_val1.push_back(score_val1 / REPS);
        curve_test1.push_back(score_test1 / REPS);
        curve_train2.push_back(score_train2 / REPS);
        curve_val2.push_back(score_val2 / REPS);
        curve_test2.push_back(score_test2 / REPS);
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

//    test(path+"pca/2_50/", "../../data/pca_data/", filter_2_50, 2);
//    test(path+"kpca_500/2_50/", "../../data/kernel_pca_data_500/", filter_2_50, 2);
//    test(path+"kpca_1000/2_50/", "../../data/kernel_pca_data_1000/", filter_2_50, 2);
//    test(path+"kpca_2000/2_50/", "../../data/kernel_pca_data_2000/", filter_2_50, 2);
//
//    test(path+"pca/23_34/", "../../data/pca_data/", filter_23_34, 2);
//    test(path+"kpca_500/23_34/", "../../data/kernel_pca_data_500/", filter_23_34, 2);
//    test(path+"kpca_1000/23_34/", "../../data/kernel_pca_data_1000/", filter_23_34, 2);
//    test(path+"kpca_2000/23_34/", "../../data/kernel_pca_data_2000/", filter_23_34, 2);
//
//    test(path+"pca/bucket/", "../../data/pca_data/", filter_bucket, 8, 0.1);
//    test(path+"kpca_500/bucket/", "../../data/kernel_pca_data_500/", filter_bucket, 8);
//    test(path+"kpca_1000/bucket/", "../../data/kernel_pca_data_1000/", filter_bucket, 8);
//    test(path+"kpca_2000/bucket/", "../../data/kernel_pca_data_2000/", filter_bucket, 8, 0.1);

//    test(path+"pca/all/", "../../data/pca_data/", filter_all, 102, 0.1);
//    test(path+"kpca_500/all/", "../../data/kernel_pca_data_500/", filter_all, 102);
//    test(path+"kpca_1000/all/", "../../data/kernel_pca_data_1000/", filter_all, 102);
    test(path+"kpca_2000/all/", "../../data/kernel_pca_data_2000/", filter_all, 102, 0.1);

    return 0;
}