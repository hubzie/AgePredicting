#pragma once

#include<Eigen/Dense>

#include<string>
#include<vector>
#include <random>

struct FileNotFound : std::exception {};

struct Data {
    Eigen::Matrix<double, 48*48, 1> x;
    short y;
};

std::vector<Data> fromFile(const std::string&);
