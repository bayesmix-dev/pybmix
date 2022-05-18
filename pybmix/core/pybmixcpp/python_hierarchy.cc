#include "python_hierarchy.h"

#include <google/protobuf/stubs/casts.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <random>
#include <sstream>
#include <stan/math/prim/prob.hpp>
#include <string>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "pybmixcpp/bayesmix/src/utils/rng.h"
#include "py_global.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

void synchronize_cpp_to_py_state(const std::mt19937 &cpp_gen,
                                 py::object &py_gen);

void synchronize_py_to_cpp_state(std::mt19937 &cpp_gen,
                                 const py::object &py_gen);

std::vector<double> list_to_vector(py::list &x);

//! PYTHON
double PythonHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
    double result = py_global::like_lpdf_evaluator(datum, state.generic_state).cast<double>();
    return result;
}

//! PYTHON
double PythonHierarchy::marg_lpdf(const Python::Hyperparams &params,
                                  const Eigen::RowVectorXd &datum) const {
    double result = py_global::marg_lpdf_evaluator(datum, params.generic_hypers).cast<double>();
    return result;
}

//! PYTHON
void PythonHierarchy::initialize_state() {
    py::list state_py = py_global::initialize_state_evaluator(hypers->generic_hypers);
    state.generic_state = list_to_vector(state_py);
}

//! C++
void PythonHierarchy::initialize_hypers() {
    if (prior->has_values()) {
        // Set values
        hypers->generic_hypers.clear();
        int size = prior->values().size();
        for (int i = 0; i < size; ++i) {
            hypers->generic_hypers.push_back((prior->values().data())[i]);
        }
    }
}

//! PYTHON
//! TODO: put in python
void PythonHierarchy::update_hypers(
        const std::vector <bayesmix::AlgorithmState::ClusterState> &states) {
    auto &rng = bayesmix::Rng::Instance().get();
    if (prior->has_values()) return;
}

//! PYTHON
Python::State PythonHierarchy::draw(const Python::Hyperparams &params) {
  Python::State out;
  synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_global::py_gen);
  py::list draw_py = py_global::draw_evaluator(state.generic_state,params.generic_hypers,py_global::py_gen);
  out.generic_state = list_to_vector(draw_py);
  synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_global::py_gen);
  return out;
}


//! PYTHON
void PythonHierarchy::update_summary_statistics(
        const Eigen::RowVectorXd &datum, const bool add) {
    py::list sum_stats_py = py_global::update_summary_statistics_evaluator(datum,add,sum_stats);
//    data_sum = sum_stats_py[0].cast<double>();
//    data_sum_squares = sum_stats_py[1].cast<double>();
    sum_stats = list_to_vector(sum_stats_py);
}

//! PYTHON
void PythonHierarchy::clear_summary_statistics() {
    py::list sum_stats_py = py_global::clear_summary_statistics_evaluator(sum_stats);
//    data_sum = sum_stats_py[0].cast<double>();
//    data_sum_squares = sum_stats_py[1].cast<double>();
    sum_stats = list_to_vector(sum_stats_py);
}

//! PYTHON
Python::Hyperparams PythonHierarchy::compute_posterior_hypers() const {
    // Compute posterior hyperparameters
    Python::Hyperparams post_params;
    py::list post_params_py = py_global::posterior_hypers_evaluator(card,hypers->generic_hypers,sum_stats);
    post_params.generic_hypers = list_to_vector(post_params_py);
    return post_params;
    }


//! C++
void PythonHierarchy::set_state_from_proto(
        const google::protobuf::Message &state_) {
    auto &statecast = downcast_state(state_);
    int size = statecast.general_state().size();
    std::vector<double> aux_v{};
    for (int i = 0; i < size; ++i) {
        aux_v.push_back((statecast.general_state().data())[i]);
    }
    state.generic_state = aux_v;
    set_card(statecast.cardinality());
}

//! C++
std::shared_ptr <bayesmix::AlgorithmState::ClusterState>
PythonHierarchy::get_state_proto() const {
    bayesmix::Vector state_;
    state_.set_size(state.generic_state.size());
    *state_.mutable_data() = {
            state.generic_state.data(),
            state.generic_state.data() + state.generic_state.size()};
    auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
    out->mutable_general_state()->CopyFrom(state_);
    return out;
}

//! C++
void PythonHierarchy::set_hypers_from_proto(
        const google::protobuf::Message &hypers_) {
    auto &hyperscast = downcast_hypers(hypers_).python_state();
    int size = hyperscast.data().size();
    std::vector<double> aux_v{};
    for (int i = 0; i < size; ++i) {
        aux_v.push_back((hyperscast.data())[i]);
    }
    hypers->generic_hypers = aux_v;
}

//! C++
std::shared_ptr <bayesmix::AlgorithmState::HierarchyHypers>
PythonHierarchy::get_hypers_proto() const {
    bayesmix::Vector hypers_;
    hypers_.set_size(hypers->generic_hypers.size());
    *hypers_.mutable_data() = {
            hypers->generic_hypers.data(),
            hypers->generic_hypers.data() + hypers->generic_hypers.size()};
    auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
    out->mutable_python_state()->CopyFrom(hypers_);
    return out;
}

void synchronize_cpp_to_py_state(const std::mt19937 &cpp_gen,
                                 py::object &py_gen) {
    std::stringstream state{};
    state << cpp_gen;

    std::string aux_string{};
    py::list state_list{};

    for (unsigned int n = 0; n < 624; ++n) {
        state >> aux_string;
        state_list.append(std::stoul(aux_string));
    }
    state >> aux_string;
    unsigned int pos = std::stoul(aux_string);

    py::object array = py_global::numpy.attr("array")(state_list, "uint32");
    py::dict state_dict("key"_a = array, "pos"_a = pos);
    py::dict d("bit_generator"_a = "MT19937", "state"_a = state_dict);
    py_gen.attr("__setstate__")(d);
}

void synchronize_py_to_cpp_state(std::mt19937 &cpp_gen,
                                 const py::object &py_gen) {
    py::object py_state = py_gen.attr("__getstate__")();
    py::object state_ = py_state["state"]["key"].attr("tolist")();
    auto pos_ = py_state["state"]["pos"].cast<unsigned int>();

    std::stringstream ss_state_;
    for (auto elem: state_) {
        ss_state_ << elem.cast<unsigned int>() << " ";
    }
    ss_state_ << pos_;
    ss_state_ >> cpp_gen;
}


std::vector<double> list_to_vector(py::list &x) {
    unsigned int size = x.size();
    std::vector<double> v(size);
    for (unsigned int i = 0; i < size; ++i) {
        v[i] = x[i].cast<double>();
    }
    return v;
}