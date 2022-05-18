#ifndef BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "algorithm_state.pb.h"
#include "pybmixcpp/bayesmix/src/hierarchies/conjugate_hierarchy.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"

namespace py = pybind11;
using namespace py::literals;

extern py::module_ fun;

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

class PythonHierarchy
    : public ConjugateHierarchy<PythonHierarchy, Python::State,
                                Python::Hyperparams, bayesmix::PythonPrior> {
 public:
  PythonHierarchy() = default;
  ~PythonHierarchy() = default;

//  py::object posterior_hypers_evaluator = fun.attr("compute_posterior_hypers");


  //! Updates hyperparameter values given a vector of cluster states
  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override;

  //! Updates state values using the given (prior or posterior) hyperparameters
  Python::State draw(const Python::Hyperparams &params);

  //! Resets summary statistics for this cluster
  void clear_summary_statistics() override;

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::Python;
  }

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
  Python::Hyperparams compute_posterior_hypers() const;

  //! Returns whether the hierarchy models multivariate data or not
  bool is_multivariate() const override { return false; }

 protected:
  //! Evaluates the log-likelihood of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  double like_lpdf(const Eigen::RowVectorXd &datum) const override;

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @return           The evaluation of the lpdf
  double marg_lpdf(const Python::Hyperparams &params,
                   const Eigen::RowVectorXd &datum) const override;

  //! Updates cluster statistics when a datum is added or removed from it
  //! @param datum      Data point which is being added or removed
  //! @param add        Whether the datum is being added or removed
  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const bool add) override;

  //! Initializes state parameters to appropriate values
  void initialize_state() override;

  //! Initializes hierarchy hyperparameters to appropriate values
  void initialize_hypers() override;

//  //! Sum of data points currently belonging to the cluster
//  double data_sum = 0;
//
//  //! Sum of squared data points currently belonging to the cluster
//  double data_sum_squares = 0;

    //! Vector of summary statistics
    std::vector<double> sum_stats;
};

#endif  // BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_H_
