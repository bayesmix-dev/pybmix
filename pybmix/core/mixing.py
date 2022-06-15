import abc
import logging
import numpy as np

from joblib import Parallel, delayed
from scipy.special import loggamma, gamma

import pybmix.proto.mixing_id_pb2 as mixing_id
from pybmix.proto.distribution_pb2 import BetaDistribution, GammaDistribution
from pybmix.proto.mixing_prior_pb2 import DPPrior, PYPrior, TruncSBPrior, PythonMixPrior
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
    ID = mixing_id.DP
    NAME = mixing_id.MixingId.Name(ID)

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

    ID = mixing_id.PY
    NAME = mixing_id.MixingId.Name(ID)

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
                "Parameter 'discount' must be in the range (0, 1)")


class StickBreakMixing(BaseMixing):
    """ This class represents a Stick-Breaking process used for mixing in a mixture
    model. A Stick-Breaking process with 'H' components is defined as follows:
    
        w[0] = v[0],
        w[h] = v[h] x (1 - v[0])(1 - v[1])...(1 - v[h-1])   h=1, ... H-2
        w[H-1] = 1 - sum(w[:H-1])

    The stick proportions v are assumed to be independently distributed
    v[h] ~ Beta(a_h, b_h). Different choices of a_h and b_h define different
    processes. 
    For instance a_h = 1, b_h = alpha defines a truncation of the
    Dirichlet proces with total_mass = alpha. 
    a_h = 1 - discount, b_h = strength + (h+1) * discount corresponds to a 
    truncation of the Pitman-Yor process. 

    Parameters
    ----------
    n_comp : int greater than 1
        number of components in the process
    strength : float greater than 0 or None
        strength (or concentration) parameter of the Pitman-Yor Process
    discount : float, in the range (0, 1) or None. 
        If discount == 0, PitmanYorMixing is the same of DirichletProcessMixing
    beta_params : sequence of tuples or None
        The parameters of the Beta distributions of the stick proportions
    """

    ID = mixing_id.TruncSB
    NAME = mixing_id.MixingId.Name(ID)

    def __init__(self, n_comp, strength=None, discount=None, beta_params=None):
        self._check_args(n_comp, strength, discount, beta_params)
        self._build_prior_proto(n_comp, strength, discount, beta_params)

    def prior_cluster_distribution(self, grid, nsamples, mc_iter=2000):
        """
        Evaluates the prior probability of the number of clusters on a
        grid

        Parameters
        ----------
        grid : array-like of shape (n,)
               Points where to evaluate the probability mass function
        nsamples : int
                   Number of samples
        mc_iter : int
                 Number of Monte Carlo iterations used to approximate this
                 distribution.
        """

        niter = int(mc_iter)
        nus = self._simulate_weights(niter)
        weights = self._break_sticks(nus)
        # simulate from the categorical in batch using the inverse-cdf method
        # see https://stackoverflow.com/a/62875642
        cum_prob = np.cumsum(weights, axis=-1)
        r = np.random.uniform(size=(nsamples, niter, 1))
        clus_allocs = np.argmax(cum_prob > r, axis=-1)
        mc_samples = np.apply_along_axis(lambda x: len(np.unique(x)), axis=0,
                                         arr=clus_allocs)
        out = np.zeros_like(grid, dtype=np.float)
        for i, k in enumerate(grid):
            out[i] = np.sum(mc_samples == k)
        return out

    def _simulate_weights(self, n):
        return np.random.beta(
            np.vstack([self._a_coeffs] * n),
            np.vstack([self._b_coeffs] * n))

    def _break_sticks(self, nus):
        out = np.zeros((nus.shape[0], self.n_comp))
        nus = np.hstack([nus, np.ones(nus.shape[0]).reshape(-1, 1)])
        out[:, 0] = nus[:, 0]
        out[:, 1:] = nus[:, 1:] * np.cumprod(1 - nus[:, :-1], axis=1)
        return out

    def _build_prior_proto(self, n_comp, strength, discount, beta_params):
        self.prior_proto = TruncSBPrior()
        self.prior_proto.num_components = n_comp
        if self._type == "general":
            for bp in beta_params:
                param = BetaDistribution(shape_a=bp[0], shape_b=bp[1])
                self.prior_proto.beta_priors.append(param)
        elif self._type == "DP":
            self.prior_proto.dp_prior.totalmass = strength
        elif self._type == "PY":
            self.prior_proto.py_prior.strength = strength
            self.prior_proto.py_prior.discount = discount
        else:
            raise ValueError("Internal Error")

    def _check_args(self, n_comp, strength=None, discount=None, beta_params=None):
        self.n_comp = n_comp
        if beta_params is not None:
            if (strength, discount) != (None, None):
                raise ValueError(
                    "Only one between '(strength, discount)' and "
                    "'beta_params' must be specified")

            if len(beta_params) != n_comp - 1:
                raise ValueError(
                    "incompatible 'n_comp' and 'beta_params': "
                    "ncomp - 1={0} != 'len(beta_params)={1}'".format(
                        n_comp - 1, len(beta_params)))

            self._a_coeffs = np.array([x[0] for x in beta_params])
            self._b_coeffs = np.array([x[1] for x in beta_params])

            if not (np.all(self._a_coeffs > 0) and np.all(self._b_coeffs > 0)):
                raise ValueError("Found negative or zero values in 'beta_params'")

            self._type = "general"

        elif strength is not None:
            if strength <= 0:
                raise ValueError(
                    "Parameter 'strength' must be strictly greater than zero")

            if discount is not None:
                if discount >= 1 or discount <= 0:
                    raise ValueError(
                        "Parameter 'discount' must be in the range (0, 1)")
                self._type = "PY"
            else:
                discount = 0
                self._type = "DP"

            self._a_coeffs = np.ones(n_comp - 1) * (1 - discount)
            self._b_coeffs = np.array(
                [strength + (h + 1) * discount for h in range(n_comp - 1)])

        else:
            raise ValueError("Not enough parameters provided")
