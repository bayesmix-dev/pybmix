#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) { return i + j; }

int subtract(int i, int j) { return i - j; }

PYBIND11_MODULE(pybmix, m) {
  m.def("add", &add);
  m.def("subtract", &subtract);
}