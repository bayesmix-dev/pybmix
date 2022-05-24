#include "py_global.h"

namespace py_global{
    py::scoped_interpreter guard{};
    py::module_ numpy = py::module_::import("numpy");
    py::module_ fun = py::module_::import("fun");
    py::module_ numpy_random = py::module_::import("numpy.random");
    py::object py_engine = numpy_random.attr("MT19937")();
    py::object py_gen = numpy_random.attr("Generator")(py_engine);
    py::object posterior_hypers_evaluator = fun.attr("compute_posterior_hypers");
    py::object like_lpdf_evaluator = fun.attr("like_lpdf");
    py::object marg_lpdf_evaluator = fun.attr("marg_lpdf");
    py::object initialize_state_evaluator = fun.attr("initialize_state");
    py::object draw_evaluator = fun.attr("draw");
    py::object update_summary_statistics_evaluator = fun.attr("update_summary_statistics");
    py::object clear_summary_statistics_evaluator = fun.attr("clear_summary_statistics");
};