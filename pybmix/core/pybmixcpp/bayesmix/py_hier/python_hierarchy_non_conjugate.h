#ifndef BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_NON_CONJUGATE_H_
#define BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_NON_CONJUGATE_H_

#include <google/protobuf/stubs/casts.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <set>

#include "algorithm_state.pb.h"
// #include "bayesmix/src/hierarchies/conjugate_hierarchy.h"
#include "bayesmix/src/hierarchies/base_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"

#include "python_hierarchy.h"

namespace py = pybind11;
using namespace py::literals;

//! Conjugate Normal Normal-InverseGamma hierarchy for univariate data.

//! This class represents a hierarchical model where data are distributed
//! according to a normal likelihood, the parameters of which have a
//! Normal-InverseGamma centering distribution. That is:
//! f(x_i|mu,sig) = N(mu,sig^2)
//!    (mu,sig^2) ~ N-IG(mu0, lambda0, alpha0, beta0)
//! The state is composed of mean and variance. The state hyperparameters,
//! contained in the Hypers object, are (mu_0, lambda0, alpha0, beta0), all
//! scalar values. Note that this hierarchy is conjugate, thus the marginal
//! distribution is available in closed form.  For more information, please
//! refer to parent classes: `AbstractHierarchy`, `BaseHierarchy`, and
//! `ConjugateHierarchy`.

// Already defined in python_hierarchy.h, do we want to keep it this way?
/*
namespace Python {
//! Custom container for State values
struct State {
  std::vector<double> generic_state;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  std::vector<double> generic_hypers;
};
};  // namespace Python
*/

using namespace Python;

class PythonHierarchyNonConjugate
        : public BaseHierarchy<PythonHierarchyNonConjugate, Python::State,
                Python::Hyperparams, bayesmix::PythonPrior> {
public:
    PythonHierarchyNonConjugate() = default;
    ~PythonHierarchyNonConjugate() = default;


    //! Returns the Protobuf ID associated to this class
    bayesmix::HierarchyId get_id() const override {
        return bayesmix::HierarchyId::PythonNonConjugate;
    }

    void save_posterior_hypers() {
        throw std::runtime_error("save_posterior_hypers() not implemented");
    }

    //! Generates new state values from the centering posterior distribution
    //! @param update_params  Save posterior hypers after the computation?
    void sample_full_cond(const bool update_params = false) override;
    //!


    //! Updates hyperparameter values given a vector of cluster states
    void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                       &states) override;

    //! Updates state values using the given (prior or posterior) hyperparameters
    Python::State draw(const Python::Hyperparams &params);

    //! Resets summary statistics for this cluster
    void clear_summary_statistics() override;

    //! Read and set state values from a given Protobuf message
    void set_state_from_proto(const google::protobuf::Message &state_) override;

    //! Read and set hyperparameter values from a given Protobuf message
    void set_hypers_from_proto(
            const google::protobuf::Message &hypers_) override;

    //! Writes current state to a Protobuf message and return a shared_ptr
    //! New hierarchies have to first modify the field 'oneof val' in the
    //! AlgoritmState::ClusterState message by adding the appropriate type
    std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
    const override;

    //! Writes current value of hyperparameters to a Protobuf message and
    //! return a shared_ptr.
    //! New hierarchies have to first modify the field 'oneof val' in the
    //! AlgoritmState::HierarchyHypers message by adding the appropriate type
    std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto()
    const override;

    //! Computes and return posterior hypers given data currently in this cluster
    // Python::Hyperparams compute_posterior_hypers() const;

    //! Returns whether the hierarchy models multivariate data or not
    bool is_multivariate() const override { return false; }

protected:
    //! Set of values of data points belonging to this cluster
    // std::list<Eigen::RowVectorXd> cluster_data_values;
    Eigen::MatrixXd cluster_data_values;


    //! Evaluates the log-likelihood of data in a single point
    //! @param datum      Point which is to be evaluated
    //! @return           The evaluation of the lpdf
    double like_lpdf(const Eigen::RowVectorXd &datum) const override;


    //! Updates cluster statistics when a datum is added or removed from it
    //! @param datum      Data point which is being added or removed
    //! @param add        Whether the datum is being added or removed
    void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                   const bool add) override;

    //! Initializes state parameters to appropriate values
    void initialize_state() override;

    //! Initializes hierarchy hyperparameters to appropriate values
    void initialize_hypers() override;

    //! Vector of summary statistics
    std::vector<double> sum_stats;


    //! methods to be used through pybind11
    py::module_ numpy = py::module_::import("numpy");
    py::module_ fun = py::module_::import("hierarchy_nc_implementation");
    py::module_ numpy_random = py::module_::import("numpy.random");
    py::object py_engine = numpy_random.attr("MT19937")();
    py::object py_gen = numpy_random.attr("Generator")(py_engine);
    py::object like_lpdf_evaluator = fun.attr("like_lpdf");
    py::object initialize_state_evaluator = fun.attr("initialize_state");
    py::object initialize_hypers_evaluator = fun.attr("initialize_hypers");
    py::object draw_evaluator = fun.attr("draw");
    py::object update_summary_statistics_evaluator = fun.attr("update_summary_statistics");
    py::object clear_summary_statistics_evaluator = fun.attr("clear_summary_statistics");
    py::object sample_full_cond_evaluator = fun.attr("sample_full_cond");
};

#endif  // BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_NON_CONJUGATE_H_
