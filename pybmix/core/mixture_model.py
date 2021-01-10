import pybmix.core.mixing as mix
from pybmix.core.hierarchy import BaseHierarchy
from pybmix.core.pybmixcpp import AlgorithmWrapper

# TODO: algo_type, hier_type, hier_prior_type, mix_type, mix_prior_tye -> Enums


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
        self._algo = AlgorithmWrapper(
            algorithm, self.hierarchy.NAME,
            self.hierarchy.prior_params.DESCRIPTOR.full_name,
            self.mixing.NAME, self.mixing.prior_proto.DESCRIPTOR.full_name,
            self.hierarchy.prior_params.SerializeToString(),
            self.mixing.prior_proto.SerializeToString())

        self._algo.run(y, niter, nburn, rng_seed)
