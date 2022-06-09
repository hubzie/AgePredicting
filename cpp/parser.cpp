#include<iostream>
using namespace std;

#include"Utils.hpp"

int main() {
    auto data = fromFile("../../data/age_gender.csv");

    for(auto d : data)
        cout << d.x.transpose() << " -> " << d.y << "\n";

    return 0;
}