#pragma once

#include <vector>
#include <string>

#include "../utils/Utils.hpp"

class Model {
    virtual void _train(const std::vector<Data> &training, const std::vector<Data> &validation) = 0;
    virtual void _save(const std::string &filename) const = 0;
    [[nodiscard]] virtual short _call(const Data &input) const = 0;
    [[nodiscard]] virtual double _error(const std::vector<Data> &test) const;
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
    inline short operator()(const Data &input) const {
        return _call(input);
    }
    virtual ~Model() = default;
};
