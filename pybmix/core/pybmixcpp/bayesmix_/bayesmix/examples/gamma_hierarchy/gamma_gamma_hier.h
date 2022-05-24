#ifndef BAYESMIX_HIERARCHIES_GAMMAGAMMA_HIERARCHY_H_
#define BAYESMIX_HIERARCHIES_GAMMAGAMMA_HIERARCHY_H_

#include <google/protobuf/stubs/casts.h>
#include <src/hierarchies/base_hierarchy.h>
#include <src/hierarchies/conjugate_hierarchy.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "hierarchy_prior.pb.h"

namespace GammaGamma {
//! Custom container for State values
struct State {
  double rate;
};

//! Custom container for Hyperparameters values
struct Hyperparams {
  double shape, rate_alpha, rate_beta;
};
};  // namespace GammaGamma

class GammaGammaHierarchy
    : public ConjugateHierarchy<GammaGammaHierarchy, GammaGamma::State,
                                GammaGamma::Hyperparams,
                                bayesmix::EmptyPrior> {
 public:
  GammaGammaHierarchy(const double shape, const double rate_alpha,
                      const double rate_beta)
      : shape(shape), rate_alpha(rate_alpha), rate_beta(rate_beta) {
    create_empty_prior();
  }
  ~GammaGammaHierarchy() = default;

  double like_lpdf(const Eigen::RowVectorXd &datum) const override {
    return stan::math::gamma_lpdf(datum(0), hypers->shape, state.rate);
  }

  double marg_lpdf(
      const GammaGamma::Hyperparams &params, const Eigen::RowVectorXd &datum,
      const Eigen::RowVectorXd &covariate = Eigen::RowVectorXd(0)) const {
    throw std::runtime_error("marg_lpdf() not implemented");
    return 0;
  }

  GammaGamma::State draw(const GammaGamma::Hyperparams &params) {
    return GammaGamma::State{stan::math::gamma_rng(
        params.rate_alpha, params.rate_beta, bayesmix::Rng::Instance().get())};
  }

  void update_summary_statistics(const Eigen::RowVectorXd &datum,
                                 const bool add) {
    if (add) {
      data_sum += datum(0);
      ndata += 1;
    } else {
      data_sum -= datum(0);
      ndata -= 1;
    }
  }

  //! Computes and return posterior hypers given data currently in this cluster
  GammaGamma::Hyperparams compute_posterior_hypers() {
    GammaGamma::Hyperparams out;
    out.shape = hypers->shape;
    out.rate_alpha = hypers->rate_alpha + hypers->shape * ndata;
    out.rate_beta = hypers->rate_beta + data_sum;
    return out;
  }

  void initialize_state() override {
    state.rate = hypers->rate_alpha / hypers->rate_beta;
  }

  void initialize_hypers() {
    hypers->shape = shape;
    hypers->rate_alpha = rate_alpha;
    hypers->rate_beta = rate_beta;
  }

  //! Removes every data point from this cluster
  void clear_summary_statistics() {
    data_sum = 0;
    ndata = 0;
  }

  bool is_multivariate() const override { return false; }

  void set_state_from_proto(const google::protobuf::Message &state_) override {
    auto &statecast = google::protobuf::internal::down_cast<
        const bayesmix::AlgorithmState::ClusterState &>(state_);
    state.rate = statecast.general_state().data()[0];
    set_card(statecast.cardinality());
  }

  std::shared_ptr<bayesmix::AlgorithmState::ClusterState> get_state_proto()
      const override {
    bayesmix::Vector state_;
    state_.mutable_data()->Add(state.rate);

    auto out = std::make_unique<bayesmix::AlgorithmState::ClusterState>();
    out->mutable_general_state()->CopyFrom(state_);
    return out;
  }

  void update_hypers(const std::vector<bayesmix::AlgorithmState::ClusterState>
                         &states) override {
    return;
  }

  void write_hypers_to_proto(
      google::protobuf::Message *const out) const override {
    return;
  }

  void set_hypers_from_proto(
      const google::protobuf::Message &state_) override {
    return;
  }

  std::shared_ptr<bayesmix::AlgorithmState::HierarchyHypers> get_hypers_proto()
      const override {
    return nullptr;
  }

  bayesmix::HierarchyId get_id() const override {
    return bayesmix::HierarchyId::UNKNOWN_HIERARCHY;
  }

 protected:
  double data_sum = 0;
  int ndata = 0;

  double shape, rate_alpha, rate_beta;
};

#endif  // BAYESMIX_HIERARCHIES_GAMMAGAMMA_HIERARCHY_H_
