import logging
import numpy as np

import pybmix.core.mixing as mix
from pybmix.core.hierarchy import BaseHierarchy
from pybmix.core.chain import MmcmChain
from pybmix.proto.marginal_state_pb2 import MarginalState
from pybmix.core.pybmixcpp import AlgorithmWrapper, ostream_redirect

# TODO: algo_type, hier_type, hier_prior_type, mix_type, mix_prior_tye -> Enums

MARGINAL_ALGORITHMS = ["N2", "N8"] 
CONDITIONAL_ALGORITHMS = []


class MixtureModel(object):
    def __init__(self, mixing, hierarchy):
        if not isinstance(mixing, mix.BaseMixing):
            raise ValueError("mixing parameter must be of type 'BaseMixing'")

        if not isinstance(hierarchy, BaseHierarchy):
            raise ValueError(
                "hierarchy parameter must be of type 'BaseHierarchy'")

        self.mixing = mixing
        self.hierarchy = hierarchy

    def run_mcmc(self, y, algorithm="N2", niter=1000, nburn=500, rng_seed=-1):
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
        if self.algo_name in MARGINAL_ALGORITHMS:
            objtype = MarginalState
        else:
            objtype = None
            logging.error("Algorithm is not marginal")
        
        return MmcmChain(
            self._algo.get_collector().get_serialized_chain(),
            objtype, deserialize)
