import logging
import numpy as np

import pybmix.core.mixing as mix
from pybmix.core.hierarchy import BaseHierarchy
from pybmix.core.chain import MCMCchain
from pybmix.proto.algorithm_state_pb2 import AlgorithmState
from pybmix.core.pybmixcpp import AlgorithmWrapper, ostream_redirect

# TODO: algo_type, hier_type, hier_prior_type, mix_type, mix_prior_tye -> Enums

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
        self._algo = AlgorithmWrapper(
            algorithm, self.hierarchy.NAME,
            self.hierarchy.prior_params.DESCRIPTOR.full_name,
            self.mixing.NAME, self.mixing.prior_proto.DESCRIPTOR.full_name,
            self.hierarchy.prior_params.SerializeToString(),
            self.mixing.prior_proto.SerializeToString())
        
        with ostream_redirect(stdout=True, stderr=True):
            self._algo.run(y, niter, nburn, rng_seed)

    def get_chain(self, optimize_memory=False):
        deserialize = not optimize_memory
        
        return MCMCchain(
            self._algo.get_collector().get_serialized_chain(),
            AlgorithmState, deserialize)
