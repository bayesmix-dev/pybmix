import abc
import numpy as np

import pybmix.proto.hierarchy_prior_pb2 as hprior


class BaseHierarchy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def make_default_fixed_params(y):
        pass


class UnivariateNormal(BaseHierarchy):
    NAME = "NNIG"

    def __init__(self, prior_params=None):
        self.prior_params = hprior.NNIGPrior()
        if isinstance(prior_params, hprior.NNIGPrior.FixedValues):
            self.prior_params.fixed_values.CopyFrom(prior_params)
        elif isinstance(prior_params, hprior.NNIGPrior.NormalMeanPrior):
            self.prior_params.normal_mean_prior.CopyFrom(prior_params)
        elif isinstance(prior_params, hprior.NNIGPrior.NGGPrior):
            self.prior_params.ngg_prior.CopyFrom(prior_params)
        elif prior_params is not None:
            raise ValueError(
                "expected 'prior_params' to be of instance "
                "FixedValues, NormalMeanPrior or NGGPrior, "
                "found {0} instead".format(type(prior_params)))
         

    def make_default_fixed_params(self, y, exp_num_clusters=5):
        """ 
        Follow the approach in [1] to define a weakly informative prior

        Parameters
        ----------
        y : array_like of shape (n, ) 
            The observed data
        exp_num_clusters : int
            An "a priori" guess of th number of clusters in the data

        [1] Fraley, C., & Raftery, A. E. (2007). 
            Bayesian regularization for normal mixture estimation and 
            model-based clustering. Journal of classification, 24(2), 155-181.
        """
        self.prior_params.fixed_values.mean = np.mean(y)
        self.prior_params.fixed_values.shape = 3
        self.prior_params.fixed_values.scale = np.var(y) / exp_num_clusters
        self.prior_params.fixed_values.var_scaling = 0.01


class MultivariateNormal(BaseHierarchy):
    NAME = "NNW"
    pass


class LinearModel(BaseHierarchy):
    pass
