#ifndef BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_H_

#include <google/protobuf/message.h>
#include <stan/math/rev.hpp>

#include <google/protobuf/stubs/casts.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <vector>
#include <set>
#include <stan/math/prim.hpp>

#include "bayesmix/src/hierarchies/abstract_hierarchy.h"
#include "algorithm_state.pb.h"
#include "hierarchy_id.pb.h"
#include "hierarchy_prior.pb.h"
#include "src/utils/rng.h"

namespace py = pybind11;
using namespace py::literals;

//! Base template class for a hierarchy object.

//! This class is a templatized version of, and derived from, the
//! `AbstractHierarchy` class, and the second stage of the curiously recurring
//! template pattern for `Hierarchy` objects (please see the docs of the parent
//! class for further information). It includes class members and some more
//! functions which could not be implemented in the non-templatized abstract
//! class.
//! See, for instance, `ConjugateHierarchy` and `NNIGHierarchy` to better
//! understand the CRTP patterns.

//! @tparam Derived      Name of the implemented derived class
//! @tparam State        Class name of the container for state values
//! @tparam Hyperparams  Class name of the container for hyperprior parameters
//! @tparam Prior        Class name of the container for prior parameters

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

class PythonHierarchy : public AbstractHierarchy {
 public:
  PythonHierarchy() = default;
  ~PythonHierarchy() = default;

    //! Set the update algorithm for the current hierarchy
    void set_updater(std::shared_ptr<AbstractUpdater> updater_) override{return;};

    //! Returns (a pointer to) the likelihood for the current hierarchy
    std::shared_ptr<AbstractLikelihood> get_likelihood() override{return nullptr;};

    //! Returns (a pointer to) the prior model for the current hierarchy
    std::shared_ptr<AbstractPriorModel> get_prior() override{return nullptr;};

    //! Returns whether the hierarchy depends on covariate values or not
    bool is_dependent() const override{return false;};

  //! Returns an independent, data-less copy of this object
  std::shared_ptr<AbstractHierarchy> clone() const override;

  //! Returns an independent, data-less copy of this object
  std::shared_ptr<AbstractHierarchy> deep_clone() const override;

  //! Evaluates the log-likelihood of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd like_lpdf_grid(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
                                                          0)) const override;

  //! Generates new state values from the centering prior distribution
  void sample_prior() override {
    state = static_cast<PythonHierarchy *>(this)->draw(*hypers);
  };

  //! Overloaded version of sample_full_cond(bool), mainly used for debugging
  virtual void sample_full_cond(
      const Eigen::MatrixXd &data,
      const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0, 0)) override;

  //! Returns the current cardinality of the cluster
  int get_card() const override { return card; };

  //! Returns the logarithm of the current cardinality of the cluster
  double get_log_card() const override { return log_card; };

  //! Returns the indexes of data points belonging to this cluster
  std::set<int> get_data_idx() const override { return cluster_data_idx;};

  //! Public wrapper for `marg_lpdf()` methods
  double get_marg_lpdf(
          const Python::Hyperparams &params, const Eigen::RowVectorXd &datum,
          const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const;

  //! Evaluates the log-prior predictive distribution of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double prior_pred_lpdf(const Eigen::RowVectorXd &datum,
                         const Eigen::RowVectorXd &covariate =
  Eigen::RowVectorXd(0)) const override {
      return get_marg_lpdf(*hypers, datum, covariate);
  };

  //! Evaluates the log-conditional predictive distr. of data in a single point
  //! @param datum      Point which is to be evaluated
  //! @param covariate  (Optional) covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  double conditional_pred_lpdf(const Eigen::RowVectorXd &datum,
                               const Eigen::RowVectorXd &covariate =
  Eigen::RowVectorXd(0)) const override {
      return get_marg_lpdf(posterior_hypers, datum, covariate);
  };

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd prior_pred_lpdf_grid(
          const Eigen::MatrixXd &data,
          const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
  0)) const override;

  //! Evaluates the log-prior predictive distr. of data in a grid of points
  //! @param data        Grid of points (by row) which are to be evaluated
  //! @param covariates  (Optional) covariate vectors associated to data
  //! @return            The evaluation of the lpdf
  virtual Eigen::VectorXd conditional_pred_lpdf_grid(
          const Eigen::MatrixXd &data,
          const Eigen::MatrixXd &covariates = Eigen::MatrixXd(0,
  0)) const override;

  //! Returns a pointer to the Protobuf message of the prior of this cluster
  google::protobuf::Message *get_mutable_prior() override;

  //! Writes current state to a Protobuf message by pointer
  void write_state_to_proto(
      google::protobuf::Message *const out) const override;

  //! Writes current values of the hyperparameters to a Protobuf message by
  //! pointer
  void write_hypers_to_proto(
      google::protobuf::Message *const out) const override;

  //! Returns the struct of the current state
  Python::State get_state() const { return state; };

  //! Returns the struct of the current prior hyperparameters
  Python::Hyperparams get_hypers() const { return *hypers; };

  //! Computes and return posterior hypers given data currently in this cluster
  Python::Hyperparams compute_posterior_hypers() const;

  //! Returns the struct of the current posterior hyperparameters
  Python::Hyperparams get_posterior_hypers() const { return posterior_hypers; };

  //! Adds a datum and its index to the hierarchy
  void add_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  //! Removes a datum and its index from the hierarchy
  void remove_datum(
      const int id, const Eigen::RowVectorXd &datum,
      const bool update_params = false,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) override;

  //! Main function that initializes members to appropriate values
  void initialize() override;

  //! Sets the (pointer to the) dataset matrix
  void set_dataset(const Eigen::MatrixXd *const dataset) override {
    dataset_ptr = dataset;
  };

  //! Returns the Protobuf ID associated to this class
  bayesmix::HierarchyId get_id() const override {
      return bayesmix::HierarchyId::PythonHier;
  };

  //! Saves posterior hyperparameters to the corresponding class member
  void save_posterior_hypers() {
      if(this->is_conjugate()){
          posterior_hypers =
                  static_cast<PythonHierarchy *>(this)->compute_posterior_hypers();
      }
      else{
          throw std::runtime_error("save_posterior_hypers() not implemented");
      }
  };

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
    void clear_summary_statistics();

    //! Read and set state values from a given Protobuf message
    void set_state_from_proto(const google::protobuf::Message &state_) override;

    //! Read and set hyperparameter values from a given Protobuf message
    void set_hypers_from_proto(
            const google::protobuf::Message &hypers_) override;

    //! Writes current state to a Protobuf message and return a shared_ptr
    //! New hierarchies have to first modify the field 'oneof val' in the
    //! AlgoritmState::ClusterState message by adding the appropriate type
    std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto() const override;

    //! Writes current value of hyperparameters to a Protobuf message and
    //! return a shared_ptr.
    //! New hierarchies have to first modify the field 'oneof val' in the
    //! AlgoritmState::HierarchyHypers message by adding the appropriate type
    std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto() const;

    //! Returns whether the hierarchy models multivariate data or not
    bool is_multivariate() const override { return false; };

    //! Returns whether the hierarchy is conjugate
    bool is_conjugate() const;

protected:
  //! Raises an error if the prior pointer is not initialized
  void check_prior_is_set() const {
    if (prior == nullptr) {
      throw std::invalid_argument("Hierarchy prior was not provided");
    }
  };

  //! Re-initializes the prior of the hierarchy to a newly created object
  void create_empty_prior() { prior.reset(new bayesmix::PythonHierPrior); };

  //! Re-initializes the hypers of the hierarchy to a newly created object
  void create_empty_hypers() { hypers.reset(new Python::Hyperparams); };

  //! Sets the cardinality of the cluster
  void set_card(const int card_) {
    card = card_;
    log_card = (card_ == 0) ? stan::math::NEGATIVE_INFTY : std::log(card_);
  };

  //! Resets cardinality and indexes of data in this cluster
  void clear_data() {
    set_card(0);
    cluster_data_idx = std::set<int>();
  };

  //! Down-casts the given generic proto message to a ClusterState proto
  bayesmix::AlgorithmState::ClusterState *downcast_state(
      google::protobuf::Message *const state_) const {
    return google::protobuf::internal::down_cast<
        bayesmix::AlgorithmState::ClusterState *>(state_);
  };

  //! Down-casts the given generic proto message to a ClusterState proto
  const bayesmix::AlgorithmState::ClusterState &downcast_state(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::ClusterState &>(state_);
  };

  //! Down-casts the given generic proto message to a HierarchyHypers proto
  bayesmix::AlgorithmState::HierarchyHypers *downcast_hypers(
      google::protobuf::Message *const state_) const {
    return google::protobuf::internal::down_cast<
        bayesmix::AlgorithmState::HierarchyHypers *>(state_);
  };

  //! Down-casts the given generic proto message to a HierarchyHypers proto
  const bayesmix::AlgorithmState::HierarchyHypers &downcast_hypers(
      const google::protobuf::Message &state_) const {
    return google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::HierarchyHypers &>(state_);
  };

  //! Container for state values
  Python::State state;

  //! Container for prior hyperparameters values
  std::shared_ptr<Python::Hyperparams> hypers;

  //! Container for posterior hyperparameters values
  Python::Hyperparams posterior_hypers;

  //! Pointer to a Protobuf prior object for this class
  std::shared_ptr<bayesmix::PythonHierPrior> prior;

  //! Set of indexes of data points belonging to this cluster
  std::set<int> cluster_data_idx;

  //! Current cardinality of this cluster
  int card = 0;

  //! Logarithm of current cardinality of this cluster
  double log_card = stan::math::NEGATIVE_INFTY;

  //! Pointer to the dataset matrix for the mixture model
  const Eigen::MatrixXd *dataset_ptr = nullptr;

    //! Set of values of data points belonging to this cluster
    // std::list<Eigen::RowVectorXd> cluster_data_values;
    Eigen::MatrixXd cluster_data_values;

    //! Evaluates the log-likelihood of data in a single point
    //! @param datum      Point which is to be evaluated
    //! @return           The evaluation of the lpdf
    double like_lpdf(const Eigen::RowVectorXd &datum) const override;

    //! Evaluates the log-marginal distribution of data in a single point
    //! @param params     Container of (prior or posterior) hyperparameter values
    //! @param datum      Point which is to be evaluated
    //! @return           The evaluation of the lpdf
    double marg_lpdf(const Python::Hyperparams &params,
                     const Eigen::RowVectorXd &datum) const;

  //! Evaluates the log-marginal distribution of data in a single point
  //! @param params     Container of (prior or posterior) hyperparameter values
  //! @param datum      Point which is to be evaluated
  //! @param covariate  Covariate vector associated to datum
  //! @return           The evaluation of the lpdf
  virtual double marg_lpdf(const Python::Hyperparams &params,
                           const Eigen::RowVectorXd &datum,
                           const Eigen::RowVectorXd &covariate) const {
      if (!this->is_dependent()) {
          throw std::runtime_error(
                  "Cannot call marg_lpdf() from a non-dependent hierarchy");
      } else {
          throw std::runtime_error("marg_lpdf() not implemented");
      }
  };

    //! Updates cluster statistics when a datum is added or removed from it
    //! @param datum      Data point which is being added or removed
    //! @param add        Whether the datum is being added or removed
    void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                   const bool add) override;

    //! Initializes state parameters to appropriate values
    void initialize_state();

    //! Initializes hierarchy hyperparameters to appropriate values
    void initialize_hypers();

    //! Vector of summary statistics
    std::vector<double> sum_stats;

    py::module_ numpy = py::module_::import("numpy");
    py::module_ fun = py::module_::import("hier_implementation");
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
    py::object posterior_hypers_evaluator = fun.attr("compute_posterior_hypers");
    py::object marg_lpdf_evaluator = fun.attr("marg_lpdf");
    // py::object update_hypers_evaluator = fun.attr("update_hypers");
    py::object is_conjugate_evaluator = fun.attr("is_conjugate");
};

#endif  // BAYESMIX_HIERARCHIES_PYTHON_HIERARCHY_H_
