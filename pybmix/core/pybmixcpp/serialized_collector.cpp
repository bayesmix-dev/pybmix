#include "serialized_collector.hpp"

void add_serialized_collector(pybind11::module &m) {
    namespace py = pybind11;

    py::class_<SerializedCollector>(m, "SerializedCollector")
            .def(py::init<>())
            .def("get_serialized_state", &SerializedCollector::get_serialized_state)
            .def("get_serialized_chain", &SerializedCollector::get_serialized_chain);
}
