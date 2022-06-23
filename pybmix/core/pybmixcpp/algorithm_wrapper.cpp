#include "algorithm_wrapper.hpp"

#include "hierarchy_prior.pb.h"

AlgorithmWrapper::AlgorithmWrapper(const std::string &algo_type,
                                   const std::string &hier_type,
                                   const std::string &mix_type,
                                   const std::string &serialized_hier_prior,
                                   const std::string &serialized_mix_prior) {
    algo = factory_algo.create_object(algo_type);
    hier = factory_hier.create_object(hier_type);
    mixing = factory_mixing.create_object(mix_type);

    hier->get_mutable_prior()->ParseFromString(serialized_hier_prior);
    mixing->get_mutable_prior()->ParseFromString(serialized_mix_prior);

    if (algo_type == bayesmix::AlgorithmId_Name(bayesmix::AlgorithmId::Neal8)) {
        algo_params.set_neal8_n_aux(3);
    }
}

void AlgorithmWrapper::run(const Eigen::MatrixXd &data, int niter, int burnin,
                           int rng_seed) {
    hier->initialize();
    if (rng_seed > 0) {
        auto &rng = bayesmix::Rng::Instance().get();
        rng.seed(rng_seed);
    }

    algo_params.set_iterations(niter);
    algo_params.set_burnin(burnin);
    algo->read_params_from_proto(algo_params);

    algo->set_mixing(mixing);
    algo->set_data(data);
    algo->set_hierarchy(hier);

    algo->run(&collector);
}

void AlgorithmWrapper::say_hello() {
    std::cout << "Hello from AlgorithmWrapper" << std::endl;
}

void AlgorithmWrapper::change_hier(const std::string &module_name) {
    if (dynamic_cast<PythonHierarchy *>(hier.get()) != nullptr) {
        static_cast<PythonHierarchy *>(hier.get())->set_module(module_name.c_str());
    }
}

void AlgorithmWrapper::change_mix(const std::string &module_name) {
    if (dynamic_cast<PythonMixing *>(mixing.get()) != nullptr) {
        static_cast<PythonMixing *>(mixing.get())->set_module(module_name.c_str());
    }
}

void add_algorithm_wrapper(pybind11::module &m) {
    namespace py = pybind11;
    py::class_<AlgorithmWrapper>(m, "AlgorithmWrapper")
            .def(py::init<>())
            .def(py::init<const std::string &, const std::string &, const std::string &,
                    const std::string &, const std::string &>())
            .def("say_hello", &AlgorithmWrapper::say_hello)
            .def("run", &AlgorithmWrapper::run)
            .def("eval_density", &AlgorithmWrapper::eval_density)
            .def("get_collector", &AlgorithmWrapper::get_collector)
            .def("change_hier", &AlgorithmWrapper::change_hier)
            .def("change_mix", &AlgorithmWrapper::change_mix);
}
