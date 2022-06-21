import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(HERE, "../../build/")
sys.path.insert(0, os.path.realpath(BUILD_DIR))

from pybmix.core.mixture_model import MixtureModel
from pybmixcpp import _minbinder_cluster_estimate, ostream_redirect


class ClusterEstimator(object):
    """
    Computes cluster estimates from a MixtureModel.
    
    Parameters
    ----------
    mixture_model: an instance of MixtureModel
        the fitted mixture, assumes that 'run_mcmc' has called
    loss: string
        the loss function to use. Currently supports only the Binder loss 
        function with equal missclassification cost
    method: string
        the method to find the point estimates. Currently supports
        only the 'samples' method, that looks for the best partition among the
        ones visited by the MCMC sampler.
    """

    def __init__(self, mixture_model: MixtureModel, loss="binder_equal",
                 method="samples"):
        self.model = mixture_model
        self.chain = self.model.get_chain()
        self.loss = loss
        self.method = method

    def get_point_estimate(self):
        if self.method == "samples" and self.loss == "binder_equal":
            with ostream_redirect(stdout=True, stderr=True):
                return _minbinder_cluster_estimate(
                    self.chain.extract("cluster_allocs"))

        else:
            raise ValueError(
                "cluster point estimate only supports method='samples' and "
                "loss='binder_equal' for the moment")

    @staticmethod
    def group_by_cluster(partition):
        """Returns a list of indices, one for each cluster"""
        labels = np.unique(partition)
        out = []
        for l in labels:
            out.append(np.where(partition == l)[0])

        return out
