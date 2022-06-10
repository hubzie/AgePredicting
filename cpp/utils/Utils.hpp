#pragma once

#include<Eigen/Dense>

#include<iostream>

#include<string>
#include<vector>

struct FileNotFound : std::exception {};

struct Data {
    Eigen::VectorXd x;
    short y;
};

std::ostream& operator<< (std::ostream&, const Data&);

std::vector<Data> fromFile(const std::string& path);
std::vector<Data> fromParsedFile(const std::string& path);

Eigen::VectorXd mean(const std::vector<Data>&);
Eigen::VectorXd deviation(const std::vector<Data>&);

void standardize(std::vector<Data>&, const Eigen::VectorXd& mean, const Eigen::VectorXd& dev);
void shuffle(std::vector<Data>&);
std::pair<std::vector<Data>, std::vector<Data>> split(const std::vector<Data>&, float frac);