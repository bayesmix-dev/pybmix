#ifndef PYBMIX_AUXILIARY_FUNCTIONS_H
#define PYBMIX_AUXILIARY_FUNCTIONS_H

#include <random>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

//! Collection of auxiliary functions which simplify passing data and
//! the random engine between python and C++

namespace py = pybind11;
using namespace py::literals;

void synchronize_cpp_to_py_state(const std::mt19937 &cpp_gen,
                                 py::object &py_gen);

void synchronize_py_to_cpp_state(std::mt19937 &cpp_gen,
                                 const py::object &py_gen);

std::vector<double> list_to_vector(py::list &x);

#endif //PYBMIX_AUXILIARY_FUNCTIONS_H
