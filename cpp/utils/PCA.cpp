#include"PCA.hpp"

#include<Eigen/Eigenvalues>
#include<vector>

#include<iostream>

using namespace Eigen;
using namespace std;

PCA::PCA(const std::vector<Data>& data, double compression) {
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
    BDCSVD svd(X, ComputeFullV);

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
}

VectorXd PCA::transform(const VectorXd& x) const { return V.transpose() * (x - mean); }
MatrixXd PCA::getV() const { return V; }
