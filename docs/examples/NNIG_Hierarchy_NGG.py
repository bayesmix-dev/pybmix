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
scalar values. In the following implementation we assume a prior on the
hyperparameters. In particular:
    mu0 ~ N(PRIOR_MEAN, PRIOR_VAR)
    lambda0 ~ Gamma(PRIOR_ALPHA, PRIOR_BETA)
    alpha0 = PRIOR_SHAPE (fixed)
    beta0 ~ Gamma(PRIOR_A, PRIOR_B)
"""

import numpy as np
import scipy.stats as ss

PRIOR_MEAN = 1.0
PRIOR_VAR = 2.25
PRIOR_ALPHA = 0.2
PRIOR_BETA = 0.6
PRIOR_SHAPE = 1.5
PRIOR_A = 4.0
PRIOR_B = 2.0


def is_conjugate():
    """

    Returns
    -------
    bool
        True for conjugate, False for non-conjugate hierarchies
    """
    return True


def like_lpdf(x, state):
    """ Likelihood log-density

    Parameters
    ----------
    x : :obj:`list` of :obj:`float`
        point in which lpdf is evaluated
    state : :obj:`list` of :obj:`float`
        model parameters
    """
    mu = state[0]
    sig = state[1]
    return ss.norm.logpdf(x, mu, np.sqrt(sig))


def marg_lpdf(x, hypers):
    """ Marginal log density

    Parameters
    ----------
    x : :obj:`list` of :obj:`float`
        point in which lpdf is evaluated
    hypers : :obj:`list` of :obj:`float`
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
    hypers : :obj:`list` of :obj:`float`
        model hyperparameters

    Returns
    -------
    :obj:`list` of :obj:`float`
        initial value of the state
    """
    mu0 = hypers[0]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    return [mu0, beta0 / (alpha0 + 1)]


def initialize_hypers():
    """ In this example NGG prior is assumed on the hyperparameters
    mu0 ~ N(PRIOR_MEAN, PRIOR_VAR)
    lambda0 ~ Gamma(PRIOR_ALPHA, PRIOR_BETA)
    alpha0 = PRIOR_SHAPE (fixed)
    beta0 ~ Gamma(PRIOR_A, PRIOR_B)

    Returns
    -------
    :obj:`list` of :obj:`float`
        initial value of the hyperparameters
    """

    mean = PRIOR_MEAN
    var_scaling = PRIOR_ALPHA / PRIOR_BETA
    shape = PRIOR_SHAPE
    scale = PRIOR_A / PRIOR_B
    return [mean, var_scaling, shape, scale]


def draw(state, hypers, rng):
    """ Samples values for the state parameters

    Parameters
    ----------
    state : :obj:`list` of :obj:`float`
        model parameters
    hypers : :obj:`list` of :obj:`float`
        model hyperparameters
    rng : numpy.random._generator.Generator
        random number generator to be used when sampling

    Returns
    -------
    :obj:`list` of :obj:`float`
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
    card : int
        cardinality of the cluster
    hypers : :obj:`list` of :obj:`float`
        model hyperparameters
    sum_stats : :obj:`list` of :obj:`float`
        list of summary statistics used

    Returns
    -------
    :obj:`list` of :obj:`float`
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
    x : :obj:`list` of :obj:`float`
        datum (univariate)
    add : bool
        if True, the datum has to be added to the cluster, if False, it has to be removed from the cluster
    sum_stats : :obj:`list` of :obj:`float`
        list of summary statistics used
    state : :obj:`list` of :obj:`float`
        model parameters
    cluster_data_values : :obj:`list` of :obj:`float`
        data in the current cluster

    Returns
    -------
    :obj:`list` of :obj:`list` of :obj:`float`
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


def update_hypers(states, hypers, rng):
    """Updates the hyperparameters according to the NGG prior assumption:
    mu0 ~ N(PRIOR_MEAN, PRIOR_VAR)
    lambda0 ~ Gamma(PRIOR_ALPHA, PRIOR_BETA)
    alpha0 = PRIOR_SHAPE (fixed)
    beta0 ~ Gamma(PRIOR_A, PRIOR_B)

    Parameters
    ----------
    states : :obj:`list` of :obj:`list` of :obj:`float`
        states of the clusters
    hypers : :obj:`list` of :obj:`float`
        model hyperparameters
    rng : numpy.random._generator.Generator
        random number generator to be used when sampling

    Returns
    -------
    :obj:`list` of :obj:`float`
        updated hyperparameters
    """
    states = np.array(states)
    num = np.sum(states[:, 0] / states[:, 1])
    b_n = np.sum(1.0 / states[:, 1])
    beta_n = np.sum(((hypers[0] - states[:, 0])**2)/states[:, 1])
    var = hypers[1] * b_n + 1 / PRIOR_VAR
    b_n += PRIOR_B
    num = hypers[1] * num + PRIOR_MEAN / PRIOR_VAR
    beta_n = PRIOR_BETA + 0.5 * beta_n
    sig_n = 1 / var
    mu_n = num / var
    alpha_n = PRIOR_ALPHA + 0.5 * len(states)
    a_n = PRIOR_A + len(states) * hypers[2]
    # Update hyperparameters with posterior random Gibbs sampling
    new_mean = ss.norm.rvs(mu_n, sig_n, random_state=rng)
    new_var_scaling = ss.gamma.rvs(a=alpha_n, loc=0, scale=1. / beta_n, random_state=rng)
    new_shape = hypers[2]
    new_scale = ss.gamma.rvs(a=a_n, loc=0, scale=1. / b_n, random_state=rng)
    return [new_mean, new_var_scaling, new_shape, new_scale]
