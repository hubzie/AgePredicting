#include <fstream>
#include <random>
#include <iostream>
#include <chrono>
#include <utility>

#include "Utils.hpp"
#include "KernelSVM.hpp"

int KernelSVM::_call(const Data &input) const {
    return place(input) > 0 ? 1 : -1;
}

const int MAX_ZEROES = 10;

const double eps = 1e-3;

inline double KernelSVM::place(const Data &input) const {
    double res = -b;
    for (const int &it : non_zero)
        res += a[it] * X[it].y * K(X[it].x, input.x);
    return res;
}

inline void KernelSVM::clean() {
    for (auto it = non_zero.begin(); it != non_zero.end(); ++it) {
        if (a[*it] < eps) {
            it = non_zero.erase(it);
            continue;
        }
    }
    zeros = 0;
}

bool KernelSVM::update(const int &i1, const int &i2, const double &E1, const double &E2, const std::vector<Data> &training) {
    const double &y1 = training[i1].y, &y2 = training[i2].y;
    const auto &x1 = training[i1].x, &x2 = training[i2].x;

    const double s = y1 == y2 ? 1 : -1;

    const double L = std::max(0.0, y1 == y2 ? a[i1] + a[i2] - C : a[i2] - a[i1]);
    const double H = std::min(C, y1 == y2 ? a[i1] + a[i2] : C + a[i2] - a[i1]);
    if (L + eps > H)
        return false;

    const double K11 = K(x1, x1), K12 = K(x1,x2), K22 = K(x2, x2);
    const double eta = K11 + K22 - 2.0 * K12;

    double a2;
    if (eta > 0) {
        a2 = std::min(std::max(a[i2] + y2 * (E1 - E2) / eta, L), H);
    } else {
        const double f1 = y1 * (E1 + b) - a[i1] * K11 - s * a[i2] * K12;
        const double f2 = y2 * (E2 + b) - a[i2] * K22 - s * a[i1] * K12;
        const double L1 = a[i1] + s * (a[i2] - L);
        const double H1 = a[i1] + s * (a[i2] - H);
        const double Lobj = L1 * f1 + L * f2 + .5 * (L1 * L1 * K11 + L * L * K22) + s * L * L1 * K12;
        const double Hobj = H1 * f1 + H * f2 + .5 * (H1 * H1 * K11 + H * H * K22) + s * H * H1 * K12;

        if (Lobj < Hobj - eps)
            a2 = L;
        else if (Hobj < Lobj - eps)
            a2 = H;
        else return false;
    }
    const double a1 = a[i1] + s * (a[i2] - a2);
    if (std::abs(a2 - a[i2]) < eps * (a2 + a[i2] + eps))
        return false;

    if (!onBound(a1))
        b += E1 + y1 * (a1 - a[i1]) * K11 + y2 * (a2 - a[i2]) * K12;
    else if (!onBound(a2))
        b += E2 + y1 * (a1 - a[i1]) * K12 + y2 * (a2 - a[i2]) * K22;
    else b += .5 * (E1 + E2 + y1 * (a1 - a[i1]) * (K11 + K12) + y2 * (a2 - a[i2]) * (K22 + K12));

    if (!onBound(a1) && onBound(a[i1]))
        unbound.emplace_back(i1, E1);
    if (!onBound(a2) && onBound(a[i2]))
        unbound.emplace_back(i2, E2);

    if (a[i1] < eps && a1 > eps)
        non_zero.push_back(i1);
    if (a[i2] < eps && a2 > eps)
        non_zero.push_back(i2);

    zeros += (a1 < eps && a[i1] > eps);
    zeros += (a2 < eps && a[i2] > eps);
    if (zeros > MAX_ZEROES)
        clean();

    a[i1] = a1;
    a[i2] = a2;

    refreshCache(training);
    return true;
}

void KernelSVM::refreshCache(const std::vector<Data> &training) {
    std::for_each(unbound.begin(), unbound.end(), [&](std::pair<int, double> &el) -> void {
        el.second = place(training[el.first]) - training[el.first].y;
    });
}

inline bool KernelSVM::onBound(const double &x) const {
    return x < eps || x > C - eps;
}

const double tol = 1e-3;

bool KernelSVM::examineExample(const int &i2, const double &E2, const std::vector<Data> &training) {
    const double &y2 = training[i2].y;
    const double r2 = E2 * y2;
    double E1;
    if ((r2 < -tol && a[i2] < C - eps) || (r2 > tol && a[i2] > eps)) {
        int i1 = -1;
        double bestDiff = 0, diff;
        for (auto it = unbound.begin(); it != unbound.end(); ++it) {
            if (onBound(a[it->first])) {
                it = unbound.erase(it);
                continue;
            }
            diff = std::abs(E2 - it->second);
            if (diff > bestDiff + eps) {
                bestDiff = diff;
                i1 = it->first;
                E1 = it->second;
            }
        }

        if (i1 >= 0 && update(i1, i2, E1, E2, training))
            return true;

        std::uniform_int_distribution<size_t> dist(0, training.size() - 1);
        for (auto it = unbound.begin(), pos = std::next(unbound.begin(), dist(gen)); it != unbound.end(); ++it, ++pos) {
            if (pos == unbound.end())
                pos = unbound.begin();
            if (update(pos->first, i2, pos->second, E2, training))
                return true;
        }

        i1 = 0;
        for (size_t p = dist(gen); i1 < (int)training.size(); ++i1, ++p) {
            if (p == training.size())
                p = 0;
            if (onBound(a[p]) && update((int)p, i2, place(training[p]) - training[p].y, E2, training))
                return true;
        }
    }
    return false;
}

void KernelSVM::_train(const std::vector<Data> &training, const std::vector<Data> &validation) {
    a.assign(training.size(), 0);
    X = training;
    non_zero.clear();
    zeros = 0;

    bool change = false, examineAll = true;
    int iter = 0;
    while (change || examineAll) {
        if (++iter % 250 == 0)
            std::cout << "SVM iteration #" << iter << '\n';
        change = false;
        if (examineAll)
            for (int i = 0; i < (int)training.size(); ++i)
                change |= examineExample(i, place(training[i]) - training[i].y, training);
        else {
            for (auto i = unbound.begin(); i != unbound.end(); ++i) {
                if (onBound(a[i->first])) {
                    i = unbound.erase(i);
                    continue;
                }
                change |= examineExample(i->first, i->second, training);
            }
        }
        examineAll = !examineAll && !change;
    }
}

void KernelSVM::_save(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file.is_open())
        throw FileNotFound();

    for (const double &c : a)
        file << c << ' ';
    file << b << '\n';
    file.close();
}

KernelSVM::KernelSVM(Kernel K, const double &C):
    K(std::move(K)),
    C(C), b(0.0),
    gen(std::chrono::system_clock::now().time_since_epoch().count()) {

}
