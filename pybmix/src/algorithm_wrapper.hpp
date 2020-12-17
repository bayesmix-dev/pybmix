#ifndef PYBMIX_ALGORITHM_WRAPPER_
#define PYBMIX_ALGORITHM_WRAPPER_

#include <pybind11/pybind11.h>

#include "bayesmix/src/includes.hpp"
#include "serialized_collector.hpp"

class AlgorithmWrapper {
 protected:
  SerializedCollector collector;
  Factory<BaseAlgorithm> &factory_algo = Factory<BaseAlgorithm>::Instance();
  Factory<BaseHierarchy> &factory_hier = Factory<BaseHierarchy>::Instance();
  Factory<BaseMixing> &factory_mixing = Factory<BaseMixing>::Instance();

  std::shared_ptr<BaseAlgorithm> algo;
  std::shared_ptr<BaseHierarchy> hier;
  std::shared_ptr<BaseMixing> mixing;

  google::protobuf::Message *mix_prior;
  google::protobuf::Message *hier_prior;

 public:
  AlgorithmWrapper() {}
  
  ~AlgorithmWrapper() {
    delete mix_prior;
    delete hier_prior;
  }

  AlgorithmWrapper(std::string algo_type, std::string hier_type,
                   std::string hier_prior_type, std::string mix_type,
                   std::string mix_prior_type,
                   std::string serialized_hier_prior,
                   std::string serialized_mix_prior);


  void run(Eigen::MatrixXd data, int niter, int burnin, double rng_seed = -1);

  void say_hello();
};

#endif