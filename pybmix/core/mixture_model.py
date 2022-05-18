import logging
import numpy as np

import os
import sys
HERE = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(HERE, "../../build/")
sys.path.insert(0, os.path.realpath(BUILD_DIR))

import pybmix.core.mixing as mix
import pybmix.proto.algorithm_id_pb2 as algorithm_id
from pybmix.core.hierarchy import BaseHierarchy
from pybmix.core.chain import MCMCchain
from pybmix.proto.algorithm_state_pb2 import AlgorithmState
from pybmixcpp import AlgorithmWrapper, ostream_redirect

MARGINAL_ALGORITHMS = ["Neal2", "Neal3", "Neal8"] 
CONDITIONAL_ALGORITHMS = ["BlockedGibbs"]


class MixtureModel(object):
    def __init__(self, mixing, hierarchy):
        if not isinstance(mixing, mix.BaseMixing):
            raise ValueError("mixing parameter must be of type 'BaseMixing'")

        if not isinstance(hierarchy, BaseHierarchy):
            raise ValueError(
                "hierarchy parameter must be of type 'BaseHierarchy'")

        self.mixing = mixing
        self.hierarchy = hierarchy

    def run_mcmc(self, y, algorithm="Neal2", niter=1000, nburn=500, rng_seed=-1):
        if algorithm not in (MARGINAL_ALGORITHMS + CONDITIONAL_ALGORITHMS):
            raise ValueError(
                "'algorithm' parameter must be one of [{0}], found {1} instead".format(
                    ", ".join(MARGINAL_ALGORITHMS + CONDITIONAL_ALGORITHMS),
                    algorithm))

        self.algo_name = algorithm
        self.algo_id = algorithm_id.AlgorithmId.Value(self.algo_name)
        self._algo = AlgorithmWrapper(
            self.algo_name, self.hierarchy.NAME, self.mixing.NAME,
            self.hierarchy.prior_params.SerializeToString(),
            self.mixing.prior_proto.SerializeToString())
        
        with ostream_redirect(stdout=True, stderr=True):
            self._algo.run(y, niter, nburn, rng_seed)

    def get_chain(self, optimize_memory=False):
        deserialize = not optimize_memory
        
        return MCMCchain(
            self._algo.get_collector().get_serialized_chain(),
            AlgorithmState, deserialize)
