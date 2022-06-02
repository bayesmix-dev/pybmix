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
#include "bayesmix/src/utils/rng.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "auxiliary_functions.h"

//! PYTHON
double PythonHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
    double result = like_lpdf_evaluator(datum, state.generic_state).cast<double>();
    return result;
}

//! PYTHON
double PythonHierarchy::marg_lpdf(const Python::Hyperparams &params,
                                  const Eigen::RowVectorXd &datum) const {
    double result = marg_lpdf_evaluator(datum, params.generic_hypers).cast<double>();
    return result;
}

//! PYTHON
void PythonHierarchy::initialize_state() {
    py::list state_py = initialize_state_evaluator(hypers->generic_hypers);
    state.generic_state = list_to_vector(state_py);
}

//! C++
void PythonHierarchy::initialize_hypers() {
//    if (prior->has_values()) {
//        // Set values
//        hypers->generic_hypers.clear();
//        int size = prior->values().size();
//        for (int i = 0; i < size; ++i) {
//            hypers->generic_hypers.push_back((prior->values().data())[i]);
//        }
//    }
    py::list hypers_py = initialize_hypers_evaluator();
    hypers->generic_hypers = list_to_vector(hypers_py);
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
  synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_gen);
  py::list draw_py = draw_evaluator(state.generic_state,params.generic_hypers,py_gen);
  out.generic_state = list_to_vector(draw_py);
  synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_gen);
  return out;
}


//! PYTHON
void PythonHierarchy::update_summary_statistics(
        const Eigen::RowVectorXd &datum, const bool add) {
    py::list sum_stats_py = update_summary_statistics_evaluator(datum,add,sum_stats);
//    data_sum = sum_stats_py[0].cast<double>();
//    data_sum_squares = sum_stats_py[1].cast<double>();
    sum_stats = list_to_vector(sum_stats_py);
}

//! PYTHON
void PythonHierarchy::clear_summary_statistics() {
    py::list sum_stats_py = clear_summary_statistics_evaluator(sum_stats);
//    data_sum = sum_stats_py[0].cast<double>();
//    data_sum_squares = sum_stats_py[1].cast<double>();
    sum_stats = list_to_vector(sum_stats_py);
}

//! PYTHON
Python::Hyperparams PythonHierarchy::compute_posterior_hypers() const {
    // Compute posterior hyperparameters
    Python::Hyperparams post_params;
    py::list post_params_py = posterior_hypers_evaluator(card,hypers->generic_hypers,sum_stats);
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