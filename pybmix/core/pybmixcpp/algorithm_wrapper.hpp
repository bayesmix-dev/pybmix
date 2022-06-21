#ifndef PYBMIX_ALGORITHM_WRAPPER_
#define PYBMIX_ALGORITHM_WRAPPER_

#include <stan/math/rev.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bayesmix/src/includes.h"
#include "py_hier/includes.h"
#include "serialized_collector.hpp"

class AlgorithmWrapper {
protected:
    SerializedCollector collector;
    AlgorithmFactory &factory_algo = AlgorithmFactory::Instance();
    HierarchyFactory &factory_hier = HierarchyFactory::Instance();
    MixingFactory &factory_mixing = MixingFactory::Instance();

    std::shared_ptr <BaseAlgorithm> algo;
    std::shared_ptr <AbstractHierarchy> hier;
    std::shared_ptr <AbstractMixing> mixing;

    std::shared_ptr <google::protobuf::Message> mix_prior;
    std::shared_ptr <google::protobuf::Message> hier_prior;
    bayesmix::AlgorithmParams algo_params;

public:
    AlgorithmWrapper() {}

    ~AlgorithmWrapper() {}

    AlgorithmWrapper(const std::string &algo_type, const std::string &hier_type,
                     const std::string &mix_type,
                     const std::string &serialized_hier_prior,
                     const std::string &serialized_mix_prior);

    void run(const Eigen::MatrixXd &data, int niter, int burnin,
             int rng_seed = -1);

    Eigen::MatrixXd eval_density(const Eigen::MatrixXd grid) {
        Eigen::MatrixXd out = algo->eval_lpdf(&collector, grid).array().exp();
        return out;
    }

    void say_hello();

    const SerializedCollector &get_collector() const { return collector; }

    void change_module(const std::string &module_name);

};

void add_algorithm_wrapper(pybind11::module &m);

#endif
