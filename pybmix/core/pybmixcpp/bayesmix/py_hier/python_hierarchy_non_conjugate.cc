#include "python_hierarchy_non_conjugate.h"

#include <google/protobuf/stubs/casts.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <random>
#include <sstream>
#include <stan/math/prim/prob.hpp>
#include <string>
#include <vector>
#include <cmath>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "bayesmix/src/utils/rng.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "auxiliary_functions.h"

//! PYTHON
double PythonHierarchyNonConjugate::like_lpdf(const Eigen::RowVectorXd &datum) const {
    double result = like_lpdf_evaluator(datum, state.generic_state).cast<double>();
    return result;
}


//! PYTHON
void PythonHierarchyNonConjugate::initialize_state() {
    py::list state_py = initialize_state_evaluator(hypers->generic_hypers);
    state.generic_state = list_to_vector(state_py);
}

//! C++
void PythonHierarchyNonConjugate::initialize_hypers() {
    py::list hypers_py = initialize_hypers_evaluator();
    hypers->generic_hypers = list_to_vector(hypers_py);
}

//! PYTHON
void PythonHierarchyNonConjugate::update_hypers(
        const std::vector <bayesmix::AlgorithmState::ClusterState> &states) {
    auto &rng = bayesmix::Rng::Instance().get();
    if (prior->has_values()) return;
}

//! PYTHON
Python::State PythonHierarchyNonConjugate::draw(const Python::Hyperparams &params) {
    Python::State out;
    synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_gen);
    py::list draw_py = draw_evaluator(state.generic_state,params.generic_hypers,py_gen);
    out.generic_state = list_to_vector(draw_py);
    synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_gen);
    return out;
}


//! PYTHON
void PythonHierarchyNonConjugate::update_summary_statistics(
        const Eigen::RowVectorXd &datum, const bool add) {
    py::list results = update_summary_statistics_evaluator(datum,add,sum_stats, state.generic_state, cluster_data_values);
    py::list sum_stats_py = results[0];
    py::array cluster_data_values_py = results[1];
    sum_stats = list_to_vector(sum_stats_py);
    cluster_data_values = cluster_data_values_py.cast<Eigen::MatrixXd>();
}

//! PYTHON
void PythonHierarchyNonConjugate::clear_summary_statistics() {
    Eigen::MatrixXd empty;
    cluster_data_values = empty;
    py::list sum_stats_py = clear_summary_statistics_evaluator(sum_stats);
    sum_stats = list_to_vector(sum_stats_py);
}



//! PYTHON
void PythonHierarchyNonConjugate::sample_full_cond(const bool update_params /*= false*/) {
    if (this->card == 0) {
        // No posterior update possible
        this->sample_prior();
    } else {
        synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_gen);
        py::list result = sample_full_cond_evaluator(state.generic_state, sum_stats, py_gen, cluster_data_values, hypers->generic_hypers);
        synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_gen);
        py::list state_list = result[0];
        py::list sum_stats_list = result[1];
        state.generic_state = list_to_vector(state_list);
        sum_stats = list_to_vector(sum_stats_list);
    }
}


//! PYTHON
//Eigen::VectorXd PythonHierarchyNonConjugate::propose_rwmh(
//        const Eigen::VectorXd &curr_vals) {
//    synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_gen);
//    py::list proposal = propose_rwmh_evaluator(curr_vals, hypers->generic_hypers, py_gen);
//    synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_gen);
//    Eigen::VectorXd proposalcpp;
//    for(unsigned int i = 0; i < proposal.size(); ++i){
//        proposalcpp << proposal[i].cast<double>();
//    }
//    return proposalcpp;
//}
//
//
////! PYTHON
//double PythonHierarchyNonConjugate::eval_prior_lpdf_unconstrained(
//        const Eigen::VectorXd &unconstrained_parameters) {
//
//    double result = eval_prior_lpdf_unconstrained_evaluator(unconstrained_parameters, hypers->generic_hypers).cast<double>();
//    return result;
//}
//
//
////! PYTHON
//double PythonHierarchyNonConjugate::eval_like_lpdf_unconstrained(
//        const Eigen::VectorXd &unconstrained_parameters, const bool is_current) {
//    double result = eval_like_lpdf_unconstrained_evaluator(unconstrained_parameters, is_current, sum_stats, cluster_data_values).cast<double>();
//    return result;
//}


//! C++
void PythonHierarchyNonConjugate::set_state_from_proto(
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
PythonHierarchyNonConjugate::get_state_proto() const {
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
void PythonHierarchyNonConjugate::set_hypers_from_proto(
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
PythonHierarchyNonConjugate::get_hypers_proto() const {
    bayesmix::Vector hypers_;
    hypers_.set_size(hypers->generic_hypers.size());
    *hypers_.mutable_data() = {
            hypers->generic_hypers.data(),
            hypers->generic_hypers.data() + hypers->generic_hypers.size()};
    auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
    out->mutable_python_state()->CopyFrom(hypers_);
    return out;
}