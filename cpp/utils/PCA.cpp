#include"PCA.hpp"

#include<Eigen/Eigenvalues>
#include<fstream>
#include<iostream>
#include<vector>

using namespace Eigen;
using namespace std;

const std::string PCA::NAME = "PCA_SAVE_FILE";

void PCA::prepare(const vector<Data>& data, double compression) {
    assert(status == NOT_INITIALIZED);
    assert(!data.empty());

    int n = data.size(), m = data[0].x.size();

    MatrixXd X(n,m);
    for(int i=0;i<n;i++)
        X.row(i) = data[i].x;

    mean = VectorXd::Zero(m);
    for(int i=0;i<n;i++) mean += X.row(i);
    mean /= n;
    for(int i=0;i<n;i++) X.row(i) -= mean;

    cerr << "PCA: Dataset size = " << n << endl;
    cerr << "PCA: Compression rate = " << compression << endl;
    cerr << "PCA: Computing SVD decomposition..." << endl;
    BDCSVD<MatrixXd> svd(X, ComputeFullV);

    double frac = 0.0;
    auto s = svd.singularValues();
    for(int i=0;i<s.size();i++) frac += s(i)*s(i);
    frac *= compression;

    int l = 0;
    double sum = 0.0;
    for(int i=0;i<s.size();i++,l++) {
        if (sum > frac) break;
        sum += s(i)*s(i);
    }

    cerr << "PCA: Compressed from " << m << " to " << l << " features" << endl;

    V = svd.matrixV();
    V.conservativeResize(m, l);

    status = INITIALIZED;
}

void PCA::fromFile(const string& path) {
    assert(status == NOT_INITIALIZED);

    cerr << "PCA: Loading from file..." << endl;

    ifstream file(path);
    assert(file.is_open());

    string name;
    assert(file >> name);

    assert(name == NAME);

    int n, m;
    assert(file >> n >> m);

    mean = VectorXd(n);
    for(int i=0;i<n;i++)
        assert(file >> mean(i));

    V = MatrixXd(n,m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            assert(file >> V(i,j));

    status = INITIALIZED;
}

VectorXd PCA::transform(const VectorXd& x) const {
    assert(status == INITIALIZED);
    return V.transpose() * (x - mean);
}

void PCA::save(const string& path) {
    assert(status == INITIALIZED);

    cerr << "PCA: Saving..." << endl;

    ofstream file(path);
    assert(file.is_open());

    cerr << "PCA: Size = "<< V.rows() << "x" << V.cols() << endl;

    file << NAME << "\n";
    file << V.rows() << " " << V.cols() << "\n";
    file << mean.transpose() << "\n";
    file << V << "\n";
}