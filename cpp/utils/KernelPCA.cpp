#include"KernelPCA.hpp"

#include<Eigen/Eigenvalues>

#include<fstream>
#include<iostream>

using namespace Eigen;
using namespace std;

static MatrixXd meanRow(MatrixXd X) {
    for(int i=0;i<X.rows();i++) {
        double val = X.row(i).mean();
        for(int j=0;j<X.cols();j++)
            X(i,j) = val;
    }
    return X;
}

static MatrixXd meanCol(MatrixXd X) {
    for(int j=0;j<X.cols();j++) {
        double val = X.col(j).mean();
        for(int i=0;i<X.rows();i++)
            X(i,j) = val;
    }
    return X;
}

VectorXd KernelPCA::apply(const VectorXd& x) const {
    VectorXd result(base.cols());
    for(int i=0;i<base.cols();i++)
        result(i) = kernel(x, base.col(i));
    return result;
}

void KernelPCA::prepare(const vector<Data>& data, KernelFunc func, double compression) {
    assert(status == NOT_INITIALIZED);
    assert(!data.empty());

    kernel = func;
    int n = data.size(), m = data[0].x.size();

    cerr << "KernelPCA: Dataset size = " << n << endl;
    cerr << "KernelPCA: Compression rate = " << compression << endl;
    cerr << "KernelPCA: Transforming data..." << endl;

    MatrixXd K(n,n);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            K(i,j) = kernel(data[i].x, data[j].x);

    K = K - meanRow(K) - meanCol(K) + meanRow(meanCol(K));

    cerr << "KernelPCA: Computing SVD decomposition..." << endl;
    BDCSVD<MatrixXd> svd(K, ComputeFullV);

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

    cerr << "KernelPCA: Compressed from " << m << " to " << l << " features" << endl;

    V = svd.matrixV();
    V.conservativeResize(n, l);
    for(int i=0;i<l;i++)
        V.col(i) /= s(i);

    base = MatrixXd(m, n);
    for(int i=0;i<n;i++)
        base.col(i) = data[i].x;

    status = INITIALIZED;
}

void KernelPCA::fromFile(const string& path, KernelFunc func) {
    assert(status == NOT_INITIALIZED);

    cerr << "KernelPCA: Loading from file..." << endl;

    kernel = func;

    ifstream file(path);
    assert(file.is_open());

    string name;
    assert(file >> name);

    assert(name == NAME);

    int n, m, k;
    assert(file >> n >> m >> k);

    base = MatrixXd(n,k);
    for(int i=0;i<n;i++)
        for(int j=0;j<k;j++)
            assert(file >> base(i,j));

    V = MatrixXd(n,m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            assert(file >> V(i,j));

    status = INITIALIZED;
}

VectorXd KernelPCA::transform(const VectorXd& x) const {
    assert(status == INITIALIZED);
    return V.transpose() * apply(x);
}

void KernelPCA::save(const string& path) {
    assert(status == INITIALIZED);

    cerr << "KernelPCA: Saving..." << endl;

    ofstream file(path);
    assert(file.is_open());

    cerr << "KernelPCA: Size = "<< V.rows() << "x" << V.cols() << endl;

    file << NAME << "\n";
    file << V.rows() << " " << V.cols() << " " << base.cols() << "\n";
    file << base << "\n";
    file << V << "\n";
}