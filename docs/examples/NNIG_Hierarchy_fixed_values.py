"""
Conjugate Normal Normal-InverseGamma hierarchy for univariate data.

This class represents a hierarchical model where data are distributed
according to a normal likelihood, the parameters of which have a
Normal-InverseGamma centering distribution. That is:
f(x_i|mu,sig) = N(mu,sig^2)
   (mu,sig^2) ~ N-IG(mu0, lambda0, alpha0, beta0)
This hierarchy is conjugate, therefore the following methods have to be
implemented: is_conjugate, like_lpdf, marg_lpdf, initialize_state,
initialize_hypers, update_hypers, draw, compute_posterior_hypers,
update_summary_statistics.
The state is composed of mean and variance. The state hyperparameters,
contained in the Hypers object, are (mu0, lambda0, alpha0, beta0), all
scalar values. In the following implementation
the hyperparameters are assumed as fixed values so update hypers does not
modify them.
"""

import numpy as np
import scipy.stats as ss


def is_conjugate():
    """

    Return
    ------
    bool
        True for conjugate, False for non-conjugate hierarchies
    """
    return True


def like_lpdf(x, state):
    """ Likelihood log-density

    Parameters
    ----------
    x: list
        point in which lpdf is evaluated
    state: list
        model parameters
    """
    mu = state[0]
    sig = state[1]
    return ss.norm.logpdf(x, mu, np.sqrt(sig))


def marg_lpdf(x, hypers):
    """ Marginal log density

    Parameters
    ----------
    x: list
        point in which lpdf is evaluated
    hypers: list
        model hyperparameters
    """
    mu0 = hypers[0]
    lambda0 = hypers[1]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    sig_n = np.sqrt(beta0 * (lambda0 + 1) / (alpha0 * lambda0))
    return ss.t.logpdf(x, 2 * alpha0, mu0, sig_n)


def initialize_state(hypers):
    """

    Parameters
    ----------
    hypers: list
        model hyperparameters

    Return
    ------
    list
        initial value of the state
    """
    mu0 = hypers[0]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    return [mu0, beta0 / (alpha0 + 1)]


def initialize_hypers():
    """ In this example no prior is assumed on the hyperparameters,
    i.e. fixed values are assumed for the hyperparameters

    Return
    ------
    list
        initial value of the hyperparameters
    """
    return [1, 1, 1, 1]


def update_hypers(states, hypers, rng):
    """Update hypers if a prior is assumed on the hyperparameters,
    otherwise if fixed values are assumed, return hypers

    Parameters
    ----------
    state: list
        model parameters
    list hypers: list
        model hyperparameters
    state: list
        model parameter

    Return
    ------
    list[list]
        updated hyperparameters
    """
    return hypers


def draw(state, hypers, rng):
    """ Samples values for the state parameters

    Parameters
    ----------
    state: list
        model parameters
    hypers: list
        model hyperparameters
    rng:
        random number generator to be used when sampling

    Return
    ------
        sampled state values
    """
    sig = state[1]
    mu0 = hypers[0]
    lambda0 = hypers[1]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    mu_ = ss.norm.rvs(mu0, np.sqrt(sig / lambda0), random_state=rng)
    sig_ = 1 / (ss.gamma.rvs(alpha0,
                             random_state=rng) / beta0)  # Inverse gamma of shape=(alpha0) and rate=(1/beta0)
    return [mu_, sig_]


def compute_posterior_hypers(card, hypers, sum_stats):
    """Computation of the posterior hyperparameters

    Parameters
    ----------
    card: int
        cardinality of the cluster
    hypers: list
        model hyperparameters
    sum_stats: list
        list of summary statistics used

    Return
    ------
    list:
        posterior hyperparameters
    """
    data_sum = sum_stats[0]
    data_sum_squares = sum_stats[1]
    mu0 = hypers[0]
    lambda0 = hypers[1]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    if card == 0:
        return hypers
    post_hypers = [0] * len(hypers)
    y_bar = data_sum / card
    sstat = data_sum_squares - card * (y_bar ** 2)
    post_hypers[0] = (lambda0 * mu0 + data_sum) / (lambda0 + card)  # mean
    post_hypers[1] = lambda0 + card  # var_scaling
    post_hypers[2] = alpha0 + 0.5 * card  # shape
    num = 0.5 * lambda0 * card * ((y_bar - mu0) ** 2)
    denom = card + lambda0
    post_hypers[3] = beta0 + 0.5 * sstat + num / denom  # scale
    return post_hypers


def update_summary_statistics(x, add, sum_stats, state, cluster_data_values):
    """ Updates cluster statistics when a datum is added or removed from it,
    the statistics appears in the sampling of the full conditionals of the
    model for non-conjugate hierarchies and in the computation of the posterior
    hyperparameters for conjugate hierarchies. In this model, the summary statistics
    are the current and proposed values of the sum of absolute differences.

    Parameters
    ----------
    x: list
        datum (univariate)
    add: bool
        if True, the datum has to be added to the cluster, if False, it has to be removed from the cluster
    sum_stats: list
        list of summary statistics used
    state: list
        model parameters
    cluster_data_values: list
        data in the current cluster

    Return
    ------
    list[list]
        updated summary statistics and cluster data values
    """
    if not len(sum_stats):
        sum_stats = [0, 0]
    data_sum = sum_stats[0]
    data_sum_squares = sum_stats[1]
    if add:
        data_sum += x[0]
        data_sum_squares += x[0] ** 2
    else:
        data_sum -= x[0]
        data_sum_squares -= x[0] ** 2
    sum_stats = [data_sum, data_sum_squares]
    return [sum_stats, cluster_data_values]
