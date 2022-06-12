#pragma once

#include <vector>
#include <string>
#include <fstream>

#include "../utils/Utils.hpp"

class Model {
    virtual void _train(const std::vector<Data> &training, const std::vector<Data> &validation) = 0;
    virtual void _save(const std::string &filename) const = 0;
    virtual void _load(std::ifstream &filename) = 0;
    [[nodiscard]] virtual int _call(const Data &input) const = 0;
    [[nodiscard]] virtual double _error(const std::vector<Data> &test) const;
    virtual double _distance(const std::vector<Data> &test) const;
public:
    inline void train(const std::vector<Data> &training, const std::vector<Data> &validation = std::vector<Data>()) {
        _train(training, validation);
    }
    inline void save(const std::string &filename) const {
        _save(filename);
    }
    [[nodiscard]] inline double error(const std::vector<Data> &test) const {
        return _error(test);
    }
    inline double distance(const std::vector<Data> &test) const {
        return _distance(test);
    }
    inline int operator()(const Data &input) const {
        return _call(input);
    }
    inline void load(const std::string &filename) {
        std::ifstream file(filename);
        _load(file);
    }
    inline void load(std::ifstream &file) {
        _load(file);
    }
    virtual ~Model() = default;
};
