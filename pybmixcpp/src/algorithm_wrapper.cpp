#include "algorithm_wrapper.hpp"

AlgorithmWrapper::AlgorithmWrapper(std::string algo_type, std::string hier_type,
                                   std::string hier_prior_type,
                                   std::string mix_type,
                                   std::string mix_prior_type,
                                   std::string serialized_hier_prior,
                                   std::string serialized_mix_prior) {
  algo = factory_algo.create_object(algo_type);
  hier = factory_hier.create_object(hier_type);
  mixing = factory_mixing.create_object(mix_type);

  auto mix_prior_desc =
      google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(
          mix_prior_type);
  assert(mix_prior_desc != NULL);
  mix_prior = google::protobuf::MessageFactory::generated_factory()
                  ->GetPrototype(mix_prior_desc)
                  ->New();
  mix_prior->ParseFromString(serialized_mix_prior);

  auto hier_prior_desc =
      google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(
          hier_prior_type);
  assert(hier_prior_desc != NULL);
  hier_prior = google::protobuf::MessageFactory::generated_factory()
                   ->GetPrototype(hier_prior_desc)
                   ->New();
  hier_prior->ParseFromString(serialized_hier_prior);

  if (algo_type == "N8") {
    algo->set_n_aux(3);
  }
}

void AlgorithmWrapper::run(Eigen::MatrixXd data, int niter, int burnin,
                           double rng_seed) {
  mixing->set_prior(*mix_prior);
  hier->set_prior(*hier_prior);
  hier->initialize();
  if (rng_seed > 0) {
    auto &rng = bayesmix::Rng::Instance().get();
    rng.seed(rng_seed);
  }

  algo->set_maxiter(niter);
  algo->set_burnin(burnin);
  algo->set_mixing(mixing);
  algo->set_data(data);
  algo->set_initial_clusters(hier, 5);

  algo->run(&collector);
}

void AlgorithmWrapper::say_hello() {
  std::cout << "Hello from AlgorithmWrapper" << std::endl;
}