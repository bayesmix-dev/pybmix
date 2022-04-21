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

namespace py = pybind11;
using namespace py::literals;

extern py::module_ numpy;
extern py::module_ fun;
extern py::module_ numpy_random;
extern py::object py_engine;
extern py::object py_gen;

void synchronize_cpp_to_py_state(const std::mt19937 &cpp_gen,
                                 py::object &py_gen);

void synchronize_py_to_cpp_state(std::mt19937 &cpp_gen,
                                 const py::object &py_gen);

py::list vector_to_list(const std::vector<double> &x);

std::vector<double> list_to_vector(py::list &x);

py::list eigen_to_list(const Eigen::RowVectorXd &x);

//! PYTHON
double PythonHierarchy::like_lpdf(const Eigen::RowVectorXd &datum) const {
    py::list state_py = vector_to_list(state.generic_state);
    py::list datum_py = eigen_to_list(datum);
    double result = fun.attr("like_lpdf")(datum_py, state_py).cast<double>();
    return result;
}

//! PYTHON
double PythonHierarchy::marg_lpdf(const Python::Hyperparams &params,
                                  const Eigen::RowVectorXd &datum) const {
    py::list params_py = vector_to_list(params.generic_hypers);
    py::list datum_py = eigen_to_list(datum);
    double result = fun.attr("marg_lpdf")(datum_py, params_py).cast<double>();
    return result;
}

//! PYTHON
void PythonHierarchy::initialize_state() {
    py::list hypers_py = vector_to_list(hypers->generic_hypers);
    py::list state_py = fun.attr("initialize_state")(hypers_py);
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
void PythonHierarchy::update_hypers(
        const std::vector <bayesmix::AlgorithmState::ClusterState> &states) {
    auto &rng = bayesmix::Rng::Instance().get();
    if (prior->has_values()) return;
}

//! PYTHON
Python::State PythonHierarchy::draw(const Python::Hyperparams &params) {
  Python::State out;
  synchronize_cpp_to_py_state(bayesmix::Rng::Instance().get(), py_gen);
  py::list params_py = vector_to_list(params.generic_hypers);
  py::list state_py = vector_to_list(state.generic_state);
  py::list draw_py = fun.attr("draw")(state_py,params_py,py_gen);
  out.generic_state = list_to_vector(draw_py);
  synchronize_py_to_cpp_state(bayesmix::Rng::Instance().get(), py_gen);
  return out;
}


//! PYTHON
void PythonHierarchy::update_summary_statistics(
        const Eigen::RowVectorXd &datum, const bool add) {
    if (add) {
        data_sum += datum(0);
        data_sum_squares += datum(0) * datum(0);
    } else {
        data_sum -= datum(0);
        data_sum_squares -= datum(0) * datum(0);
    }
}

//! PYTHON
void PythonHierarchy::clear_summary_statistics() {
    data_sum = 0;
    data_sum_squares = 0;
}

//! PYTHON
Python::Hyperparams PythonHierarchy::compute_posterior_hypers() const {
    // Initialize relevant variables
    if (card == 0) {  // no update possible
        return *hypers;
    }
    // Compute posterior hyperparameters
    Python::Hyperparams post_params;
    double y_bar = data_sum / (1.0 * card);  // sample mean
    double ss = data_sum_squares - card * y_bar * y_bar;
    post_params.generic_hypers.push_back(
            (hypers->generic_hypers[1] * hypers->generic_hypers[0] + data_sum) /
            (hypers->generic_hypers[1] + card));
    post_params.generic_hypers.push_back(hypers->generic_hypers[1] + card);
    post_params.generic_hypers.push_back(hypers->generic_hypers[2] + 0.5 * card);
    post_params.generic_hypers.push_back(
            hypers->generic_hypers[3] + 0.5 * ss +
            0.5 * hypers->generic_hypers[1] * card *
            (y_bar - hypers->generic_hypers[0]) *
            (y_bar - hypers->generic_hypers[0]) /
            (card + hypers->generic_hypers[1]));
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

    py::object array = numpy.attr("array")(state_list, "uint32");
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

py::list vector_to_list(const std::vector<double> &x) {
    py::list L;
    for (auto &elem: x) {
        L.append(elem);
    }
    return L;
}


py::list eigen_to_list(const Eigen::RowVectorXd &x) {
    py::list L;
    for (int i = 0; i < x.size(); ++i) {
        L.append(x[i]);
    }
    return L;
}

std::vector<double> list_to_vector(py::list &x) {
    std::vector<double> v;
    for (auto &elem: x) {
        v.push_back(elem.cast<double>());
    }
    return v;
}