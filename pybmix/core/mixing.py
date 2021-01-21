import abc
import logging
import math
import numpy as np

from scipy.special import loggamma, gamma

from pybmix.proto.distribution_pb2 import GammaDistribution
from pybmix.proto.mixing_prior_pb2 import DPPrior, PYPrior
from pybmix.utils.combinatorials import stirling, generalized_factorial_memoizer


class BaseMixing(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def prior_cluster_distribution(self, grid, nsamples):
        """
        Evaluates the prior probability of the number of clusters on a
        grid
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
            raise ValueError(error_msg)

        if total_mass_prior is not None and total_mass_prior.rate <= 0:
            error_msg = "parameter 'total_mass_prior.rate' must be strictly " + \
                        "greater than 0, found {0} instead".format(
                            total_mass_prior.rate)
            raise ValueError(error_msg)




class PitmanYorMixing(BaseMixing):
    """ This class represents a Pitman-Yor process used for mixing in a mixture
    model. The Pitman-Yor process depends on a 'strength' parameter and on 
    a 'discout' parameter.
    We consider only the case of scrictly positive total_mass.

    Parameters
    ----------
    strength : float greater than 0
                strength (or concentration) parameter of the Pitman-Yor
                Process
    discount : float, in the range (0, 1). If discount == 0, PitmanYorMixing
               is the same of DirichletProcessMixing
    """

    NAME = "PY"

    def __init__(self, strength, discount):
        self._check_args(strength, discount)
        self._build_prior_proto(strength, discount)
        self.generalized_factorial = generalized_factorial_memoizer(discount)

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
        strength = self.prior_proto.fixed_values.strength
        discount = self.prior_proto.fixed_values.discount
        out = np.zeros_like(grid, dtype=np.float)
        
        vnk_den = loggamma(strength + nsamples) - loggamma(strength)
        for i, k in enumerate(grid):
            vnk_num = np.sum(
                [np.log(strength + l * discount) for l in range(k)])
            vnk = vnk_num - vnk_den
            out_logscale = vnk + np.log(self.generalized_factorial(nsamples, k)) - \
                k * np.log(discount)
            out[i] = np.exp(out_logscale)
        return out

    def _build_prior_proto(self, strength, discount):
        self.prior_proto = PYPrior()
        self.prior_proto.fixed_values.strength = strength
        self.prior_proto.fixed_values.discount = discount

    def _check_args(self, strength, discount):
        if strength <= 0:
            raise ValueError(
                "Parameter 'strength' must be strictly greater than zero")

        if discount >= 1 or discount <= 0:
            raise ValueError(
                "Parameter 'discount' must be in the range (0, 1")
