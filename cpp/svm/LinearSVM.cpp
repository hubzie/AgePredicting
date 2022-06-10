#include <fstream>
#include <random>
#include <iostream>

#include "../utils/Utils.hpp"
#include "LinearSVM.hpp"

short LinearSVM::_call(const Data &input) const {
    return place(input) > 0 ? 1 : -1;
}

inline double LinearSVM::place(const Data &input) const {
    return w.dot(input.x) - b;
}

const double eps = 1e-3;

bool LinearSVM::update(const int &i1, const int &i2, const Data &d1, const Data &d2) {
    const double &y1 = d1.y, &y2 = d2.y;
    const auto &x1 = d1.x, &x2 = d2.x;

    double E1 = place(d1) - y1;
    double E2 = place(d2) - y2;
    double s = y1 == y2 ? 1 : -1;

    double L = std::max(0.0, y1 == y2 ? a[i1] + a[i2] - C : a[i2] - a[i1]);
    double H = std::min(C, y1 == y2 ? a[i1] + a[i2] : C + a[i2] - a[i1]);
    if (L + eps > H)
        return false;

    double K11 = x1.squaredNorm(), K12 = x1.dot(x2), K22 = x2.squaredNorm();
    double eta = K11 + K22 - 2.0 * K12;

    double a2;
    if (eta > 0) {
        a2 = std::min(std::max(a[i2] + y2 * (E1 - E2) / eta, L), H);
    } else {
        double f1 = y1 * (E1 + b) - a[i1] * K11 - s * a[i2] * K12;
        double f2 = y2 * (E2 + b) - a[i2] * K22 - s * a[i1] * K12;
        double L1 = a[i1] + s * (a[i2] - L);
        double H1 = a[i1] + s * (a[i2] - H);
        double Lobj = L1 * f1 + L * f2 + .5 * (L1 * L1 * K11 + L * L * K22) + s * L * L1 * K12;
        double Hobj = H1 * f1 + H * f2 + .5 * (H1 * H1 * K11 + H * H * K22) + s * H * H1 * K12;

        if (Lobj < Hobj - eps)
            a2 = L;
        else if (Hobj < Lobj - eps)
            a2 = H;
        else return false;
    }
    double a1 = a[i1] + s * (a[i2] - a2);
    if (std::abs(a2 - a[i2]) < eps * (a2 + a[i2] + eps))
        return false;

    w += (a1 - a[i1]) * y1 * x1 + (a2 - a[i2]) * y2 * x2;
    if (!onBound(a1))
        b += E1 + y1 * (a1 - a[i1]) * K11 + y2 * (a2 - a[i2]) * K12;
    else if (!onBound(a2))
        b += E2 + y1 * (a1 - a[i1]) * K12 + y2 * (a2 - a[i2]) * K22;
    else b += .5 * (E1 + E2 + y1 * (a1 - a[i1]) * (K11 + K12) + y2 * (a2 - a[i2]) * (K22 + K12));

    if (!onBound(a1) && onBound(a[i1]))
        unbound.push_back(i1);
    if (!onBound(a2) && onBound(a[i2]))
        unbound.push_back(i2);
    a[i1] = a1;
    a[i2] = a2;

    return true;
}

inline bool LinearSVM::onBound(const double &x) const {
    return x < eps || x > C - eps;
}

const double tol = 1e-3;

bool LinearSVM::examineExample(const int &i2, const std::vector<Data> &training) {
    const double &y2 = training[i2].y;
    double E2 = place(training[i2]) - y2;
    double r2 = E2 * y2;
    if ((r2 < -tol && a[i2] < C - eps) || (r2 > tol && a[i2] > eps)) {
        int i1 = -1;
        double bestDiff = 0, diff;
        for (auto it = unbound.begin(); it != unbound.end(); ++it) {
            if (onBound(a[*it])) {
                it = unbound.erase(it);
                continue;
            }
            double E1 = place(training[*it]) - training[*it].y;
            if (E1 * E2 > 0)
                continue;
            diff = std::abs(E2 - E1);
            if (diff > bestDiff + eps) {
                bestDiff = diff;
                i1 = *it;
            }
        }

        if (i1 >= 0 && update(i1, i2, training[i1], training[i2]))
            return true;

        std::uniform_int_distribution<size_t> dist(0, training.size() - 1);
        auto pos = unbound.begin();
        std::advance(pos, dist(gen));
        for (auto it = unbound.begin(); it != unbound.end(); ++it, ++pos) {
            if (pos == unbound.end())
                pos = unbound.begin();
            if (update(*it, i2, training[*it], training[i2]))
                return true;
        }

        i1 = 0;
        for (size_t p = dist(gen); i1 < (int)training.size(); ++i1, ++p) {
            if (p == training.size())
                p = 0;
            if (onBound(a[p]) && update((int)p, i2, training[p], training[i2]))
                return true;
        }
    }
    return false;
}

void LinearSVM::_train(const std::vector<Data> &training, const std::vector<Data> &validation) {
    a.assign(training.size(), 0.0);
    w.setZero();

    bool change = false, examineAll = true;
    while (change || examineAll) {
        change = false;
        if (examineAll)
            for (int i = 0; i < (int)training.size(); ++i)
                change |= examineExample(i, training);
        else {
            for (auto i = unbound.begin(); i != unbound.end(); ++i) {
                if (onBound(a[*i])) {
                    i = unbound.erase(i);
                    continue;
                }
                change |= examineExample(*i, training);
            }
        }
        examineAll = !examineAll && !change;
    }
}

void LinearSVM::_save(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file.is_open())
        throw FileNotFound();

    file << w.transpose() << " " << b << '\n';
    file.close();
}

LinearSVM::LinearSVM(const int &width, const double &C): C(C), b(0.0) {
    w = Eigen::VectorXd::Zero(width);
    gen = std::mt19937(rd());
}
