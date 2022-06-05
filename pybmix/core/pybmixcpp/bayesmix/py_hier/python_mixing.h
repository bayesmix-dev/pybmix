#ifndef PYBMIX_PYTHON_MIXING_H
#define PYBMIX_PYTHON_MIXING_H

#include <google/protobuf/message.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "bayesmix/src/mixings/base_mixing.h"
#include "mixing_id.pb.h"
#include "mixing_prior.pb.h"
#include "bayesmix/src/hierarchies/abstract_hierarchy.h"
#include "auxiliary_functions.h"

namespace py = pybind11;
using namespace py::literals;

namespace PYTHON {
    struct State {
        std::vector<double> generic_state;
    };
};  // namespace Python

class PYTHONMixing
        : public BaseMixing<PYTHONMixing, PYTHON::State,bayesmix::PYTHONPrior> {
public:
    PYTHONMixing() = default;
    ~PYTHONMixing() = default;

    //! Performs conditional update of state, given allocations and unique values
    //! @param unique_values  A vector of (pointers to) Hierarchy objects
    //! @param allocations    A vector of allocations label
    void update_state(
            const std::vector<std::shared_ptr<AbstractHierarchy>> &unique_values,
            const std::vector<unsigned int> &allocations) override;

    //! Read and set state values from a given Protobuf message
    void set_state_from_proto(const google::protobuf::Message &state_) override;

    //! Writes current state to a Protobuf message and return a shared_ptr
    //! New hierarchies have to first modify the field 'oneof val' in the
    //! MixingState message by adding the appropriate type
    std::shared_ptr<bayesmix::MixingState> get_state_proto() const override;

    //! Returns the Protobuf ID associated to this class
    bayesmix::MixingId get_id() const override { return bayesmix::MixingId::PYTHON; }

    //! Returns whether the mixing is conditional or marginal
    bool is_conditional() const override { return false; }

protected:
    //! Returns probability mass for an old cluster (for marginal mixings only)
    //! @param n          Total dataset size
    //! @param log        Whether to return logarithm-scale values or not
    //! @param propto     Whether to include normalizing constants or not
    //! @param hier       `Hierarchy` object representing the cluster
    //! @return           Probability value
    double mass_existing_cluster(
            const unsigned int n, const unsigned int n_clust, const bool log,
            const bool propto,
            const std::shared_ptr<AbstractHierarchy> hier) const override;

    //! Returns probability mass for a new cluster (for marginal mixings only)
    //! @param n          Total dataset size
    //! @param log        Whether to return logarithm-scale values or not
    //! @param propto     Whether to include normalizing constants or not
    //! @param n_clust    Current number of clusters
    //! @return           Probability value
    double mass_new_cluster(const unsigned int n, const unsigned int n_clust,
                            const bool log, const bool propto) const override;

    //! Initializes state parameters to appropriate values
    void initialize_state() override;

    py::module_ numpy = py::module_::import("numpy");
    py::module_ mixing_fun = py::module_::import("mixing_fun"); //add fun to build folder
    py::module_ numpy_random = py::module_::import("numpy.random");
    py::object py_engine = numpy_random.attr("MT19937")();
    py::object py_gen = numpy_random.attr("Generator")(py_engine);
    py::object update_state_evaluator = mixing_fun.attr("update_state");
    py::object mass_existing_cluster_evaluator = mixing_fun.attr("mass_existing_cluster");
    py::object mass_new_cluster_evaluator = mixing_fun.attr("mass_new_cluster");

    py::object initialize_state_evaluator = mixing_fun.attr("initialize_state");
};



#endif //PYBMIX_PYTHON_MIXING_H
