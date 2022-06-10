#include <numeric>

#include "Model.hpp"

double Model::_error(const std::vector<Data> &test) const {
  return (double) std::transform_reduce(test.cbegin(), test.cend(), 0, std::plus<>(), [&](const Data &d) -> size_t {
      return d.y != (*this)(d);
  }) / (double) test.size();
}
