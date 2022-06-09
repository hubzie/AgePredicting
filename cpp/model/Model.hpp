#pragma once

#include <vector>
#include <string>

#include "utils/Utils.hpp"

class Model {
public:
  virtual void train(const std::vector<Data> &training, const std::vector<Data> &validation = std::vector<Data>()) = 0;
  virtual short operator()(const Data &input) const = 0;
  virtual void save(const std::string &filename) const = 0;
  
  virtual double error(const std::vector<Data> &test) const;

  virtual ~Model() {}
};
