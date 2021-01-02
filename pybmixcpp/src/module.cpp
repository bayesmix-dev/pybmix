#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include "algorithm_wrapper.hpp"
#include "serialized_collector.hpp"


namespace py = pybind11;


PYBIND11_MODULE(pybmixcpp, m) {
  add_algorithm_wrapper(m);
  add_serialized_collector(m);
}