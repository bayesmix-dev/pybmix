import abc
import logging
import numpy as np

from scipy.special import gamma

from pybmix.proto.distribution_pb2 import GammaDistribution
from pybmix.proto.mixing_prior_pb2 import DPPrior
from pybmix.utils.combinatorials import stirling


class BaseMixing(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def prior_cluster_distribution(self, grid, nsamples):
        """
        Evaluates the prior probability of the number of clusters on a
        grid

        Parameters
        ----------
        grid : array-like of shape (n,)
               Points where to evaluate the probability mass function
        nsamples : int
                   Number of samples
        """
        pass


class DirichletProcessMixing(BaseMixing):
    """ This class represents a Dirichlet Process used for mixing in a mixture
    model. The Drichlet process depends on a 'total_mass' parameter, which
    could also be random.

    Parameters
    ----------
    total_mass : float greater than 0 or None
                 total_mass (or concentration) parameter of the Dirichlet Process
                 if None, assumes that 'total_mass_prior' is passed
    total_mass_prior : pybmix.proto.distribution_pb2.GammaDistribution or None
                       parameters for the prior distribution of 'total_mass'
    """

    NAME = "DP"

    def __init__(self, total_mass=None, total_mass_prior=None):
        self._check_args(total_mass, total_mass_prior)
        self._build_prior_proto(total_mass, total_mass_prior)
        self.total_mass_fixed = total_mass
        self.total_mass_prior = total_mass_prior

    def prior_cluster_distribution(self, grid, nsamples):
        """
        Evaluates the prior probability of the number of clusters on a
        grid

        Parameters
        ----------
        grid : array-like of shape (n,)
               Points where to evaluate the probability mass function
        nsamples : int
                   Number of samples
        """
        if self.prior_proto.WhichOneof("totalmass") != "fixed_value":
            logging.warning(
                "'prior_cluster_distribution' is implemented only for "
                "fixed total_mass parameter. Substituting the prior expected "
                "value as fixed total_mass ")
            total_mass = self.total_mass_prior.shape / self.total_mass_prior.rate
        else:
            total_mass = self.total_mass_fixed

        out = np.zeros_like(grid, dtype=np.float)
        for i, g in enumerate(grid):
            out[i] = gamma(total_mass) / gamma(total_mass + nsamples) * \
                stirling(nsamples, g) * (total_mass) ** g

        return out

    def _build_prior_proto(self, total_mass, total_mass_prior):
        self.prior_proto = DPPrior()

        if total_mass is not None:
            self.prior_proto.fixed_value.totalmass = total_mass
        else:
            self.prior_proto.gamma_prior.totalmass_prior.CopyFrom(
                total_mass_prior)

    def _check_args(self, total_mass, total_mass_prior):
        error_msg = "Exactly one between total_mass and total_mass_prior " + \
                    "must not be none"

        if total_mass is None and total_mass_prior is None:
            raise ValueError(error_msg)

        if total_mass is not None and total_mass_prior is not None:
            raise ValueError(error_msg)

        if total_mass is not None and total_mass <= 0:
            raise ValueError(
                "Parameter 'total_mass' must be strictly greater than zero")

        if total_mass_prior is not None and not isinstance(
                total_mass_prior, GammaDistribution):
            error_msg = "parameter 'total_mass_prior' must be of instance " + \
                        "pybmix.proto.distribution_pb2.GammaDistribution"
            raise ValueError(error_msg)

        if total_mass_prior is not None and total_mass_prior.shape <= 0:
            error_msg = "parameter 'total_mass_prior.shape' must be strictly " + \
                        "greater than 0, found {0} instead".format(
                            total_mass_prior.shape)

        if total_mass_prior is not None and total_mass_prior.rate <= 0:
            error_msg = "parameter 'total_mass_prior.rate' must be strictly " + \
                        "greater than 0, found {0} instead".format(
                            total_mass_prior.rate)


class PitmanYorMixing(BaseMixing):
    NAME = "PY"
    pass