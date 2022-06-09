#pragma once

#include<Eigen/Dense>

#include<string>
#include<vector>

struct FileNotFound : std::exception {};
const int DEF_SIZE = 48*48;

struct Data {
    Eigen::VectorXd x;
    short y;
};

std::vector<Data> fromFile(const std::string&);
