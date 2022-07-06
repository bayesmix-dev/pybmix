#include "python_hierarchy.h"

#include <google/protobuf/stubs/casts.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <sstream>
#include <stan/math/prim/prob.hpp>
#include <string>
#include <vector>

#include "algorithm_state.pb.h"
#include "auxiliary_functions.h"
#include "bayesmix/src/utils/rng.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

std::shared_ptr <AbstractHierarchy> PythonHierarchy::clone() const {
    auto out = std::make_shared<PythonHierarchy>((*this));
    out->clear_data();
    out->clear_summary_statistics();
    return out;
}

std::shared_ptr <AbstractHierarchy> PythonHierarchy::deep_clone() const {
    auto out = std::make_shared<PythonHierarchy>((*this));

    out->clear_data();
    out->clear_summary_statistics();

    out->create_empty_prior();
    std::shared_ptr <google::protobuf::Message> new_prior(prior->New());
    new_prior->CopyFrom(*prior.get());
    out->get_mutable_prior()->CopyFrom(*new_prior.get());

    out->create_empty_hypers();
    auto curr_hypers_proto = get_hypers_proto();
    out->set_hypers_from_proto(*curr_hypers_proto.get());
    out->initialize();
    return out;
}

std::shared_ptr <bayesmix::AlgorithmState::HierarchyHypers>
PythonHierarchy::get_hypers_proto() const {
    bayesmix::Vector hypers_;
    hypers_.set_size(hypers->generic_hypers.size());
    *hypers_.mutable_data() = {hypers->generic_hypers.data(),
                               hypers->generic_hypers.data() +
                               hypers->generic_hypers.size()};
    auto out = std::make_shared<bayesmix::AlgorithmState::HierarchyHypers>();
    out->mutable_general_state()->CopyFrom(hypers_);
    return out;
}

double PythonHierarchy::get_marg_lpdf(
        const PyHier::Hyperparams &params, const Eigen::RowVectorXd &datum,
        const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) const {
    if (this->is_dependent()) {
        return marg_lpdf(params, datum, covariate);
    } else {
        return marg_lpdf(params, datum);
    }
}

google::protobuf::Message *PythonHierarchy::get_mutable_prior() {
    if (prior == nullptr) {
        create_empty_prior();
    }
    return prior.get();
}

std::shared_ptr <bayesmix::AlgorithmState::ClusterState>
PythonHierarchy::get_state_proto() const {
    bayesmix::Vector state_;
    state_.set_size(state.generic_state.size());
    *state_.mutable_data() = {state.generic_state.data(),
                              state.generic_state.data() +
                              state.generic_state.size()};
    auto out = std::make_shared<bayesmix::AlgorithmState::ClusterState>();
    out->mutable_general_state()->CopyFrom(state_);
    return out;
}

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

void PythonHierarchy::set_module(const std::string &module_name) {
    std::cout << "Using hierarchy implementation in " << module_name << ".py" << std::endl;

    hier_implementation = py::module_::import(module_name.c_str());

    draw_evaluator = hier_implementation.attr("draw");
    is_conjugate_evaluator = hier_implementation.attr("is_conjugate");
    initialize_state_evaluator = hier_implementation.attr("initialize_state");
    initialize_hypers_evaluator = hier_implementation.attr("initialize_hypers");
    like_lpdf_evaluator = hier_implementation.attr("like_lpdf");
    update_hypers_evaluator = hier_implementation.attr("update_hypers");
    update_summary_statistics_evaluator =
            hier_implementation.attr("update_summary_statistics");

    if (is_conjugate_evaluator().cast<bool>()) {
        posterior_hypers_evaluator =
                hier_implementation.attr("compute_posterior_hypers");
        marg_lpdf_evaluator = hier_implementation.attr("marg_lpdf");
    } else {
        sample_full_cond_evaluator = hier_implementation.attr("sample_full_cond");
    }
}

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

void PythonHierarchy::write_state_to_proto(
        google::protobuf::Message *const out) const {
    std::shared_ptr <bayesmix::AlgorithmState::ClusterState> state_ =
            get_state_proto();
    auto *out_cast = downcast_state(out);
    out_cast->CopyFrom(*state_.get());
    out_cast->set_cardinality(card);
}

void PythonHierarchy::write_hypers_to_proto(
        google::protobuf::Message *const out) const {
    std::shared_ptr <bayesmix::AlgorithmState::HierarchyHypers> hypers_ =
            get_hypers_proto();
    auto *out_cast = downcast_hypers(out);
    out_cast->CopyFrom(*hypers_.get());
}

Eigen::VectorXd PythonHierarchy::conditional_pred_lpdf_grid(
        const Eigen::MatrixXd &data,
        const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
    Eigen::VectorXd lpdf(data.rows());
    if (covariates.cols() == 0) {
        // Pass null value as covariate
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) =
                    (this)->conditional_pred_lpdf(data.row(i), Eigen::RowVectorXd(0));
        }
    } else if (covariates.rows() == 1) {
        // Use unique covariate
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = (this)->conditional_pred_lpdf(data.row(i), covariates.row(0));
        }
    } else {
        // Use different covariates
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = (this)->conditional_pred_lpdf(data.row(i), covariates.row(i));
        }
    }
    return lpdf;
}

Eigen::VectorXd PythonHierarchy::like_lpdf_grid(
        const Eigen::MatrixXd &data,
        const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
    Eigen::VectorXd lpdf(data.rows());
    if (covariates.cols() == 0) {
        // Pass null value as covariate
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = (this)->get_like_lpdf(data.row(i), Eigen::RowVectorXd(0));
        }
    } else if (covariates.rows() == 1) {
        // Use unique covariate
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = (this)->get_like_lpdf(data.row(i), covariates.row(0));
        }
    } else {
        // Use different covariates
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = (this)->get_like_lpdf(data.row(i), covariates.row(i));
        }
    }
    return lpdf;
}

Eigen::VectorXd PythonHierarchy::prior_pred_lpdf_grid(
        const Eigen::MatrixXd &data,
        const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) const {
    Eigen::VectorXd lpdf(data.rows());
    if (covariates.cols() == 0) {
        // Pass null value as covariate
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = (this)->prior_pred_lpdf(data.row(i), Eigen::RowVectorXd(0));
        }
    } else if (covariates.rows() == 1) {
        // Use unique covariate
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = (this)->prior_pred_lpdf(data.row(i), covariates.row(0));
        }
    } else {
        // Use different covariates
        for (int i = 0; i < data.rows(); i++) {
            lpdf(i) = (this)->prior_pred_lpdf(data.row(i), covariates.row(i));
        }
    }
    return lpdf;
}

void PythonHierarchy::initialize() {
    hypers = std::make_shared<PyHier::Hyperparams>();
    check_prior_is_set();
    initialize_hypers();
    initialize_state();
    posterior_hypers = *hypers;
    clear_data();
    clear_summary_statistics();
}

//! PYTHON
bool PythonHierarchy::is_conjugate() const {
    return is_conjugate_evaluator().cast<bool>();
}

//! PYTHON
void PythonHierarchy::clear_summary_statistics() {
    cluster_data_values.resize(0,0);
    sum_stats = std::vector<double>(0,sum_stats.size());
}

//! PYTHON
PyHier::Hyperparams PythonHierarchy::compute_posterior_hypers() const {
    PyHier::Hyperparams post_params;
    py::list post_params_py =
            posterior_hypers_evaluator(card, hypers->generic_hypers, sum_stats);
    post_params.generic_hypers = list_to_vector(post_params_py);
    return post_params;
}

//! PYTHON
PyHier::State PythonHierarchy::draw(const PyHier::Hyperparams &params) {
    PyHier::State out;
    synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_gen);
    py::list draw_py =
            draw_evaluator(state.generic_state, params.generic_hypers, py_gen);
    out.generic_state = list_to_vector(draw_py);
    synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_gen);
    return out;
}

//! PYTHON
void PythonHierarchy::sample_full_cond(const bool update_params /* = false */) {
    if (this->card == 0) {
        // No posterior update possible
        this->sample_prior();
    } else {
        if (this->is_conjugate()) {
            PyHier::Hyperparams params =
                    update_params ? this->compute_posterior_hypers() : posterior_hypers;
            state = this->draw(params);
        } else {
            synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_gen);
            py::list result = sample_full_cond_evaluator(
                    state.generic_state, sum_stats, py_gen, cluster_data_values,
                    hypers->generic_hypers);
            synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_gen);
            py::list state_list = result[0];
            py::list sum_stats_list = result[1];
            state.generic_state = list_to_vector(state_list);
            sum_stats = list_to_vector(sum_stats_list);
        }
    }
}

void PythonHierarchy::sample_full_cond(
        const Eigen::MatrixXd &data,
        const Eigen::MatrixXd &covariates /*= Eigen::MatrixXd(0, 0)*/) {
    clear_data();
    clear_summary_statistics();
    if (covariates.cols() == 0) {
        // Pass null value as covariate
        for (int i = 0; i < data.rows(); i++) {
            (this)->add_datum(i, data.row(i), false, Eigen::RowVectorXd(0));
        }
    } else if (covariates.rows() == 1) {
        // Use unique covariate
        for (int i = 0; i < data.rows(); i++) {
            (this)->add_datum(i, data.row(i), false, covariates.row(0));
        }
    } else {
        // Use different covariates
        for (int i = 0; i < data.rows(); i++) {
            (this)->add_datum(i, data.row(i), false, covariates.row(i));
        }
    }
    (this)->sample_full_cond(true);
}

void PythonHierarchy::sample_prior() { state = (this)->draw(*hypers); };

void PythonHierarchy::save_posterior_hypers() {
    if (this->is_conjugate()) {
        posterior_hypers = (this)->compute_posterior_hypers();
    } else {
        throw std::runtime_error("save_posterior_hypers() not implemented");
    }
};

//! PYTHON
void PythonHierarchy::update_hypers(
        const std::vector <bayesmix::AlgorithmState::ClusterState> &states) {
    // std::cout << "Starting update_hypers!" << std::endl;
    unsigned int size_states = states.size();
    std::vector <std::vector<double>> pass_states{};
    pass_states.reserve(size_states);
    unsigned int size;
    for (auto &st: states) {
        size = st.general_state().size();
        std::vector<double> aux_v{};
        aux_v.reserve(size);
        for (int i = 0; i < size; ++i) {
            aux_v.push_back((st.general_state().data())[i]);
        }
        pass_states.push_back(aux_v);
    }
    synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_gen);
    py::list new_hypers =
            update_hypers_evaluator(pass_states, hypers->generic_hypers, py_gen);
    synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_gen);
    hypers->generic_hypers = list_to_vector(new_hypers);
}

void PythonHierarchy::add_datum(
        const int id, const Eigen::RowVectorXd &datum,
        const bool update_params /*= false*/,
        const Eigen::RowVectorXd &covariate /*= Eigen::RowVectorXd(0)*/) {
    assert(cluster_data_idx.find(id) == cluster_data_idx.end());
    set_card(card + 1);
    log_card = std::log(card);
    (this)->update_ss(datum, covariate, true);
    cluster_data_idx.insert(id);
    if (update_params) {
        (this)->save_posterior_hypers();
    }
}

void PythonHierarchy::remove_datum(
        const int id, const Eigen::RowVectorXd &datum,
        const bool update_params /*= false*/,
        const Eigen::RowVectorXd &covariate /* = Eigen::RowVectorXd(0)*/) {
    (this)->update_ss(datum, covariate, false);
    set_card(card - 1);
    auto it = cluster_data_idx.find(id);
    assert(it != cluster_data_idx.end());
    cluster_data_idx.erase(it);
    if (update_params) {
        (this)->save_posterior_hypers();
    }
}

void PythonHierarchy::check_prior_is_set() const {
    if (prior == nullptr) {
        throw std::invalid_argument("Hierarchy prior was not provided");
    }
};

void PythonHierarchy::clear_data() {
    set_card(0);
    cluster_data_idx = std::set<int>();
};

void PythonHierarchy::create_empty_prior() {
    prior.reset(new bayesmix::PythonHierPrior);
};

void PythonHierarchy::create_empty_hypers() {
    hypers.reset(new PyHier::Hyperparams);
};

//! PYTHON
void PythonHierarchy::initialize_state() {
    py::list state_py = initialize_state_evaluator(hypers->generic_hypers);
    state.generic_state = list_to_vector(state_py);
}

//! PYTHON
void PythonHierarchy::initialize_hypers() {
    py::list hypers_py = initialize_hypers_evaluator();
    hypers->generic_hypers = list_to_vector(hypers_py);
}

void PythonHierarchy::set_card(const int card_) {
    card = card_;
    log_card = (card_ == 0) ? stan::math::NEGATIVE_INFTY : std::log(card_);
};

//! PYTHON
double PythonHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
    double result =
            like_lpdf_evaluator(datum, state.generic_state).cast<double>();
    return result;
}

//! PYTHON
double PythonHierarchy::marg_lpdf(const PyHier::Hyperparams &params,
                                  const Eigen::RowVectorXd &datum) const {
    double result =
            marg_lpdf_evaluator(datum, params.generic_hypers).cast<double>();
    return result;
}

double PythonHierarchy::marg_lpdf(const PyHier::Hyperparams &params,
                                  const Eigen::RowVectorXd &datum,
                                  const Eigen::RowVectorXd &covariate) const {
    if (!this->is_dependent()) {
        throw std::runtime_error(
                "Cannot call marg_lpdf() from a non-dependent hierarchy");
    } else {
        throw std::runtime_error("marg_lpdf() not implemented");
    }
};

//! PYTHON
void PythonHierarchy::update_summary_statistics(const Eigen::RowVectorXd &datum,
                                                const bool add) {
    py::list results = update_summary_statistics_evaluator(
            datum, add, sum_stats, state.generic_state, cluster_data_values);
    py::list sum_stats_py = results[0];
    py::array cluster_data_values_py = results[1];
    sum_stats = list_to_vector(sum_stats_py);
    cluster_data_values = cluster_data_values_py.cast<Eigen::MatrixXd>();
}
