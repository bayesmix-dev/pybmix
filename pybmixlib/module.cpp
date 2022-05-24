// #include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "algorithm_wrapper.hpp"
#include "bayesmix/src/utils/cluster_utils.h"
#include "serialized_collector.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pybmixlib, m) {
  py::add_ostream_redirect(m, "ostream_redirect");
  add_algorithm_wrapper(m);
  add_serialized_collector(m);
  m.def("_minbinder_cluster_estimate", &bayesmix::cluster_estimate);
}
