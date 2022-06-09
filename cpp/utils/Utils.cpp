#include"Utils.hpp"

#include<fstream>
#include<iostream>

using namespace std;

vector<Data> fromFile(const string& path) {
    ifstream file(path);

    if (!file.is_open())
        throw FileNotFound();

    // Header
    string line;
    getline(file, line);

    vector<Data> result;
    while(getline(file, line)) {
        for(char& c : line)
            c = (c == ',' ? ' ' : c);

        stringstream parser(line);

        Data d;
        parser >> d.y;
        string junk;
        // Ignore ethnicity, gender and img_name
        parser >> junk >> junk >> junk;

        double value;
        int it = 0;

        d.x.resize(DEF_SIZE);
        while(parser >> value)
            d.x(it++) = value;

        result.push_back(d);
    }

    return result;
}