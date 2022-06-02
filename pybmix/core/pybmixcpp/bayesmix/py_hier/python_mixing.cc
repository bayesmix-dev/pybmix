#include "python_mixing.h"

#include <google/protobuf/stubs/casts.h>

#include <memory>
#include <stan/math/prim/prob.hpp>
#include <vector>

#include "mixing_prior.pb.h"
#include "mixing_state.pb.h"
#include "bayesmix/src/hierarchies/abstract_hierarchy.h"
#include "bayesmix/src/utils/rng.h"

#include <Eigen/Dense>
#include <random>
#include <sstream>
#include <stan/math/prim/prob.hpp>
#include <string>
#include <vector>

#include "algorithm_state.pb.h"
#include "hierarchy_prior.pb.h"
#include "ls_state.pb.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "auxiliary_functions.h"

void PYTHONMixing::update_state(
        const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
        const std::vector<unsigned int> &allocations) {
//    auto &rng = bayesmix::Rng::Instance().get();
//    auto priorcast = cast_prior();
//    unsigned int n = allocations.size();

}

double PYTHONMixing::mass_existing_cluster(
        const unsigned int n, const unsigned int n_clust, const bool log,
        const bool propto, const std::shared_ptr<AbstractHierarchy> hier) const {
//    double out;
//    if (log) {
//        out = hier->get_log_card();
//        if (!propto) out -= std::log(n + state.generic_state[0]);
//    } else {
//        out = 1.0 * hier->get_card();
//        if (!propto) out /= (n + state.generic_state[0]);
//    }
//    return out;
    return 0;
}

double PYTHONMixing::mass_new_cluster(const unsigned int n,
                                         const unsigned int n_clust,
                                         const bool log,
                                         const bool propto) const {
//    double out;
//    if (log) {
//        out = std::log(state.generic_state[0]);
//        if (!propto) out -= std::log(n + state.generic_state[0]);
//    } else {
//        out = state.generic_state[0];
//        if (!propto) out /= (n + state.generic_state[0]);
//    }
//    return out;
    return 0;
}

//! C++
void PYTHONMixing::set_state_from_proto(
        const google::protobuf::Message &state_) {
//    auto &statecast = downcast_state(state_);
//    int size = statecast.general_state().size();
//    std::vector<double> aux_v{};
//    for (int i = 0; i < size; ++i) {
//        aux_v.push_back((statecast.general_state().data())[i]);
//    }
//    state.generic_state = aux_v;
}

//! C++
std::shared_ptr<bayesmix::MixingState> PYTHONMixing::get_state_proto()
const {
    bayesmix::Vector state_;
    state_.set_size(state.generic_state.size());
    *state_.mutable_data() = {
            state.generic_state.data(),
            state.generic_state.data() + state.generic_state.size()};
    auto out = std::make_shared<bayesmix::MixingState>();
    out->mutable_general_state()->CopyFrom(state_);
    return out;

}

//! PYTHON
void PYTHONMixing::initialize_state() {
//    py::list state_py = initialize_state_evaluator();
//    state.generic_state = list_to_vector(state_py);
}