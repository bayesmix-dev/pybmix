#ifndef PYBMIX_PY_GLOBAL_
#define PYBMIX_PY_GLOBAL_

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;
using namespace py::literals;

namespace py_global{
    extern py::module_ numpy;
    extern py::module_ fun;
    extern py::module_ numpy_random;
    extern py::object py_engine;
    extern py::object py_gen;
    extern py::object posterior_hypers_evaluator;
    extern py::object like_lpdf_evaluator;
    extern py::object marg_lpdf_evaluator;
    extern py::object initialize_state_evaluator;
    extern py::object draw_evaluator;
    extern py::object update_summary_statistics_evaluator;
    extern py::object clear_summary_statistics_evaluator;
};

#endif