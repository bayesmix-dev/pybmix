#include "py_global.h"

namespace py_global{
    //py::scoped_interpreter guard{}; Gives core dumped
    py::module_ numpy = py::module_::import("numpy");
    py::module_ fun = py::module_::import("fun"); //add fun to build folder
    py::module_ numpy_random = py::module_::import("numpy.random");
    py::object py_engine = numpy_random.attr("MT19937")();
    py::object py_gen = numpy_random.attr("Generator")(py_engine);
    py::object posterior_hypers_evaluator = fun.attr("compute_posterior_hypers");
    py::object like_lpdf_evaluator = fun.attr("like_lpdf");
    py::object marg_lpdf_evaluator = fun.attr("marg_lpdf");
    py::object initialize_state_evaluator = fun.attr("initialize_state");
    py::object initialize_hypers_evaluator = fun.attr("initialize_hypers");
    py::object draw_evaluator = fun.attr("draw");
    py::object update_summary_statistics_evaluator = fun.attr("update_summary_statistics");
    py::object clear_summary_statistics_evaluator = fun.attr("clear_summary_statistics");
    py::object sample_full_cond_evaluator = fun.attr("sample_full_cond");
    py::object propose_rwmh_evaluator = fun.attr("propose_rwmh");
    py::object eval_prior_lpdf_unconstrained_evaluator = fun.attr("eval_prior_lpdf_unconstrained");
    py::object eval_like_lpdf_unconstrained_evaluator = fun.attr("eval_like_lpdf_unconstrained");
};