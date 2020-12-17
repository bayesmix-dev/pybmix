#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include "bayesmix/src/utils/distributions.hpp"
#include "bayesmix/src/utils/rng.hpp"

namespace py = pybind11;

int add(int i, int j) { return i + j; }

int subtract(int i, int j) { return i - j; }

int draw_uniform(int start, int end) {
  Eigen::VectorXd probas = Eigen::VectorXd::Ones(end - start);
  probas /= probas.sum();
  return bayesmix::categorical_rng(probas, bayesmix::Rng::Instance().get(),
                                   start);
}

PYBIND11_MODULE(pybmix, m) {
  m.def("add", &add);
  m.def("subtract", &subtract);
  m.def("draw_uniform", &draw_uniform);
}