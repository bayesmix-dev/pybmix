#ifndef PYBMIX_ALGORITHM_WRAPPER_
#define PYBMIX_ALGORITHM_WRAPPER_

#include <pybind11/pybind11.h>

#include "bayesmix/src/includes.hpp"
#include "serialized_collector.hpp"

class AlgorithmWrapper {
 protected:
  SerializedCollector collector;
  Factory<BaseAlgorithm>& factory_algo = Factory<BaseAlgorithm>::Instance();
  Factory<BaseHierarchy>& factory_hier = Factory<BaseHierarchy>::Instance();
  Factory<BaseMixing>& factory_mixing = Factory<BaseMixing>::Instance();

  std::shared_ptr<BaseAlgorithm> algo;
  std::shared_ptr<BaseHierarchy> hier;
  std::shared_ptr<BaseMixing> mixing;

  std::shared_ptr<google::protobuf::Message> mix_prior;
  std::shared_ptr<google::protobuf::Message> hier_prior;

 public:
  AlgorithmWrapper() {}

  ~AlgorithmWrapper() {}

  AlgorithmWrapper(const std::string& algo_type, const std::string& hier_type,
                   const std::string& hier_prior_type,
                   const std::string& mix_type,
                   const std::string& mix_prior_type,
                   const std::string& serialized_hier_prior,
                   const std::string& serialized_mix_prior);

  void run(const Eigen::MatrixXd& data, int niter, int burnin,
           int rng_seed = -1);

  void say_hello();

  const SerializedCollector& get_collector() { return collector; }
};

#endif
