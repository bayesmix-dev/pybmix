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
#include <cmath>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include "bayesmix/src/utils/rng.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "auxiliary_functions.h"


//template <class Derived, typename State, typename Hyperparams, typename Prior>
void PythonHierarchy::add_datum(
        const int id, const Eigen::RowVectorXd &datum,
        const bool update_params /*= false*/,
        const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) {
    assert(cluster_data_idx.find(id) == cluster_data_idx.end());
    card += 1;
    log_card = std::log(card);
    static_cast<PythonHierarchy *>(this)->update_ss(datum, covariate, true);
    cluster_data_idx.insert(id);
    if (update_params) {
        static_cast<PythonHierarchy *>(this)->save_posterior_hypers();
    }
}

//template <class Derived, typename State, typename Hyperparams, typename Prior>
void PythonHierarchy::remove_datum(
        const int id, const Eigen::RowVectorXd &datum,
        const bool update_params /*= false*/,
        const Eigen::RowVectorXd &covariate /* = Eigen::RowVectorXd(0)*/) {
    static_cast<PythonHierarchy *>(this)->update_ss(datum, covariate, false);
    set_card(card - 1);
    auto it = cluster_data_idx.find(id);
    assert(it != cluster_data_idx.end());
    cluster_data_idx.erase(it);
    if (update_params) {
        static_cast<PythonHierarchy *>(this)->save_posterior_hypers();
    }
}

//template <class Derived, typename State, typename Hyperparams, typename Prior>
void PythonHierarchy::write_state_to_proto(
        google::protobuf::Message *const out) const {
    std::shared_ptr<bayesmix::AlgorithmState::ClusterState> state_ =
            get_state_proto();
    auto *out_cast = downcast_state(out);
    out_cast->CopyFrom(*state_.get());
    out_cast->set_cardinality(card);
}

//template <class Derived, typename State, typename Hyperparams, typename Prior>
void PythonHierarchy::write_hypers_to_proto(
        google::protobuf::Message *const out) const {
    std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> hypers_ =
            get_hypers_proto();
    auto *out_cast = downcast_hypers(out);
    out_cast->CopyFrom(*hypers_.get());
}

//template <class Derived, typename State, typename Hyperparams, typename Prior>
Eigen::VectorXd
PythonHierarchy::like_lpdf_grid(
        const Eigen::MatrixXd &data,
        const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
    Eigen::VectorXd lpdf(data.rows());
    if (covariates.cols() == 0) {
        // Pass null value as covariate
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = static_cast<PythonHierarchy const *>(this)->get_like_lpdf(
                    data.row(i), Eigen::RowVectorXd(0));
        }
    } else if (covariates.rows() == 1) {
        // Use unique covariate
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = static_cast<PythonHierarchy const *>(this)->get_like_lpdf(
                    data.row(i), covariates.row(0));
        }
    } else {
        // Use different covariates
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = static_cast<PythonHierarchy const *>(this)->get_like_lpdf(
                    data.row(i), covariates.row(i));
        }
    }
    return lpdf;
}

// template <class Derived, typename State, typename Hyperparams, typename Prior>
void PythonHierarchy::sample_full_cond(
        const Eigen::MatrixXd &data,
        const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) {
    clear_data();
    clear_summary_statistics();
    if (covariates.cols() == 0) {
        // Pass null value as covariate
        for (int i = 0; i < data.rows(); i++) {
            static_cast<PythonHierarchy *>(this)->add_datum(i, data.row(i), false,
                                                            Eigen::RowVectorXd(0));
        }
    } else if (covariates.rows() == 1) {
        // Use unique covariate
        for (int i = 0; i < data.rows(); i++) {
            static_cast<PythonHierarchy *>(this)->add_datum(i, data.row(i), false,
                                                            covariates.row(0));
        }
    } else {
        // Use different covariates
        for (int i = 0; i < data.rows(); i++) {
            static_cast<PythonHierarchy *>(this)->add_datum(i, data.row(i), false,
                                                            covariates.row(i));
        }
    }
    static_cast<PythonHierarchy *>(this)->sample_full_cond(true);
}

//! PYTHON
double PythonHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
    double result = like_lpdf_evaluator(datum, state.generic_state).cast<double>();
    return result;
}


//! PYTHON
void PythonHierarchy::initialize_state() {
    py::list state_py = initialize_state_evaluator(hypers->generic_hypers);
    state.generic_state = list_to_vector(state_py);
}

//! C++
void PythonHierarchy::initialize_hypers() {
    py::list hypers_py = initialize_hypers_evaluator();
    hypers->generic_hypers = list_to_vector(hypers_py);
}

//! PYTHON
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
    py::list results = update_summary_statistics_evaluator(datum,add,sum_stats, state.generic_state, cluster_data_values);
    py::list sum_stats_py = results[0];
    py::array cluster_data_values_py = results[1];
    sum_stats = list_to_vector(sum_stats_py);
    cluster_data_values = cluster_data_values_py.cast<Eigen::MatrixXd>();
}

//! PYTHON
void PythonHierarchy::clear_summary_statistics() {
    Eigen::MatrixXd empty;
    cluster_data_values = empty;
    py::list sum_stats_py = clear_summary_statistics_evaluator(sum_stats);
    sum_stats = list_to_vector(sum_stats_py);
}



//! PYTHON
void PythonHierarchy::sample_full_cond(const bool update_params /*= false*/) {
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
    auto &hyperscast = downcast_hypers(hypers_).general_state();
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
    out->mutable_general_state()->CopyFrom(hypers_);
    return out;
}