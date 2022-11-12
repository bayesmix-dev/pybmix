import abc
import logging
import numpy as np

import pybmix.proto.hierarchy_id_pb2 as hierarchy_id
import pybmix.proto.hierarchy_prior_pb2 as hprior
from pybmix.utils.proto_utils import get_oneof_types, set_oneof_field


class BaseHierarchy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def make_default_fixed_params(y):
        return logging.error("This function cannot be implemented!")


    def check_prior_params(self, self_prior_params, prior_params):
        if prior_params is None:
            return

        success = set_oneof_field("prior", self_prior_params, prior_params)
        if not success:
            raise ValueError(
                "expected 'prior_params' to be of instance [{0}]"
                "found {1} instead".format(
                    " ".join(get_oneof_types("prior", self_prior_params)),
                    type(prior_params)))


class UnivariateNormal(BaseHierarchy):
    """ This class represents a univariate normal hierarchy, i.e. the model 
        y_1, ... y_n | mu, s^2 ~ N(mu, s^2)
        (mu, s^2) ~ Normal-inverse gamma(mu0, lambda, a, b)
    Optionally, hyperpriors can be assumed on mu0, lambda, a  and b
    """
    ID = hierarchy_id.NNIG
    NAME = hierarchy_id.HierarchyId.Name(ID)

    def __init__(self, prior_params=None):
        self.prior_params = hprior.NNIGPrior()
        self.check_prior_params(self.prior_params, prior_params)

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
    ID = hierarchy_id.NNW
    NAME = hierarchy_id.HierarchyId.Name(ID)
    pass


class LinearModel(BaseHierarchy):
    ID = hierarchy_id.LinRegUni
    NAME = hierarchy_id.HierarchyId.Name(ID)
    pass


class PythonHierarchy(BaseHierarchy):
    """ This class represents a generic hierarchy to be implemented in a dedicated python file.
    The name of the file must be passed as a string when initializing this class.
    """
    ID = hierarchy_id.PythonHier
    NAME = hierarchy_id.HierarchyId.Name(ID)

    def __init__(self, hier_implementation):
        self.prior_params = hprior.PythonHierPrior()
        self.hier_implementation = hier_implementation

    def make_default_fixed_params(self, y):
        raise NotImplementedError("make_default_fixed_params not implemented for PythonHierarchy")
