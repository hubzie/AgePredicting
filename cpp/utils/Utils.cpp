#include"Utils.hpp"

#include<Eigen/Dense>

#include<fstream>
#include<map>
#include<random>

using namespace Eigen;
using namespace std;



ostream& operator<< (ostream& out, const Data& d) {
    return out << d.y << " " << d.x.transpose();
}



static vector<Data> readFromFile(const string& path, bool isParsed) {
    ifstream file(path);

    if (!file.is_open())
        throw FileNotFound();

    // Header
    string line;
    getline(file, line);

    vector<Data> result;
    vector<double> buff;
    while(getline(file, line)) {
        for(char& c : line)
            c = (c == ',' ? ' ' : c);

        stringstream parser(line);

        Data d;
        parser >> d.y;

        if (!isParsed) {
            string junk;
            // Ignore ethnicity, gender and img_name
            parser >> junk >> junk >> junk;
        }
        
        double value;
        buff.clear();
        while(parser >> value)
            buff.push_back(value);

        if(buff.size() == 0) continue;

        d.x = VectorXd(buff.size());
        for(int i=0;i<buff.size();i++)
            d.x(i) = buff[i];

        result.push_back(d);
    }

    return result;
}

vector<Data> fromFile(const string& path) { return readFromFile(path, false); }
vector<Data> fromParsedFile(const string& path) { return readFromFile(path, true); }

void save(const string& path, const vector<Data>& data) {
    ofstream file(path);
    assert(file.is_open());
    for(auto& d : data)
        file << d << "\n";
}

VectorXd mean(const vector<Data>& data) {
    VectorXd result = VectorXd::Zero(data[0].x.size());
    for(auto& d : data)
        result += d.x;
    result /= data.size();
    return result;
}

VectorXd deviation(const vector<Data>& data) {
    VectorXd result = VectorXd::Zero(data[0].x.size());
    auto m = mean(data);
    for(auto& d : data) {
        VectorXd tmp = d.x - m;
        for (int i=0;i<tmp.size();i++) tmp(i) *= tmp(i);
        result += tmp;
    }
    result /= data.size();
    return result;
}

void standardize(vector<Data>& data, const VectorXd& mean, const VectorXd& dev) {
    for(auto& r : data) {
        r.x = (r.x - mean);
        for (int i = 0; i < r.x.size(); i++)
            r.x(i) /= dev(i);
    }
}



void shuffle(vector<Data>& data) {
    static random_device rd;
    static mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);
}

pair<vector<Data>, vector<Data>> split(const vector<Data>& data, float frac) {
    vector<Data> a,b;
    map<int,int> cnt;

    for(auto& d : data) cnt[d.y]++;
    for(auto& [k,v] : cnt) v *= frac;

    for(auto& r : data)
        (--cnt[r.y] >= 0 ? a : b).push_back(r);

    return {a, b};
}

// returns vector of age group splits. e.g. {3, 5, 10} means groups are [1,3), [3, 5), [5, 10), [10,...)
std::vector<int> equalGroups(std::vector<Data> data, const int &numGroups) {
    std::sort(data.begin(), data.end(), [](const Data &a, const Data &b) -> bool {
        return a.y < b.y;
    });

    const int target = (int)data.size() / numGroups;
    int size = 0;

    std::vector<int> splits;
    for (int i = 1, j = 0; j < (int)data.size(); ++i) {
        int old_size = size;
        while (j < (int)data.size() && data[j].y <= i)
            ++size, ++j;
        if (size > target && old_size <= target) {
            splits.push_back(size - target < target - old_size ? i + 1 : i);
            size = 0;
        }
    }
    std::cerr << "Splits:";
    for (int &i : splits)
        std::cerr << ' ' << i;
    std::cerr << '\n';
    return splits;
}