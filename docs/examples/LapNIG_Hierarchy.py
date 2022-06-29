"""
Laplace Normal-InverseGamma hierarchy for univariate data.

This file contains the Python implementation of a hierarchical model where
data are distributed according to a laplace likelihood, the parameters of
which have a Normal-InverseGamma centering distribution. That is:
    f(x_i|mu,lam) = Laplace(mu,lam)
       (mu,lam) ~ N-IG(mu0, sigma0, alpha0, beta0)
This hierarchy is non conjugate, therefore the following methods have to be
implemented: is_conjugate, like_lpdf, initialize_state, initialize_hypers,
update_hypers, draw, update_summary_statistics, sample_full_cond.
The state is a list passed from c++ as std::vector, composed of [mu, lam].
Similarly, the hypers is a list composed of [mu_0, sigma0, alpha0, beta0,
mean_var, log_scale_var], all scalar values.
The hyperprameters log_scale_var and mean_var are used to perform a step of
Random Walk Metropolis Hastings to sample from the full conditionals.

mean = hypers[0]
var = hypers[1]
shape = hypers[2]
scale = hypers[3]
mh_mean_var = hypers[4]
mh_log_scale_var = hypers[5]
"""

import numpy as np
import scipy.stats as ss


def is_conjugate():
    """

    Returns
    -------
    :bool:
        True for conjugate, False for non-conjugate hierarchies
    """
    return False


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
    lam = state[1]
    return ss.laplace.logpdf(x[0], mu, lam)


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
    """

    Returns
    -------
    :obj:`list` of :obj:`float`
        initial value of the hyperparameters
    """
    return [0, 10, 2, 1, 10, 1]


def update_hypers(state, hypers, rng):
    """ Update hypers if a prior is assumed on the hyperparameters,
    otherwise if fixed values are assumed, return hypers

    Parameters
    ----------
    state : :obj:`list` of :obj:`float`
        model parameters
    hypers : :obj:`list` of :obj:`float`
        model hyperparameters
    rng : numpy.random._generator.Generator'
        random number generator to be used when sampling

    Returns
    -------
    :obj:`list` of :obj:`float`
        updated hypers
    """
    return hypers


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
    mu0 = hypers[0]
    sigma0 = hypers[1]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    mu_ = ss.norm.rvs(loc=mu0, scale=np.sqrt(sigma0), random_state=rng)
    lam_ = ss.invgamma.rvs(a=alpha0, loc=0, scale=beta0, random_state=rng)
    return [mu_, lam_]


def update_summary_statistics(x, add, sum_stats, state, cluster_data_values):
    """ Updates cluster statistics when a datum is added or removed from it,
    the statistics appears in the sampling of the full conditionals of the model for non-conjugate hierarchies and
    in the computation of the posterior hyperparameters for conjugate hierarchies.
    In this model, the summary statistics are the current and proposed values of the sum of absolute differences,

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
    mu = state[0]
    if not len(sum_stats):
        sum_stats = [0, 0]  # initialize sum_stats to zeros
    if add:
        sum_stats[0] += abs(mu - x[0])
        cluster_data_values = np.append(cluster_data_values, x)
    else:
        sum_stats[0] -= abs(mu - x[0])
        ind = np.where(cluster_data_values == x)
        cluster_data_values = np.delete(cluster_data_values, ind[0][0], ind[1][0])
    return [sum_stats, cluster_data_values]


def sample_full_cond(state, sum_stats, rng, cluster_data_values, hypers):
    """ Sampling from the full conditional of the model

    Parameters
    ----------
    state : :obj:`list` of :obj:`float`
        model parameters
    sum_stats : :obj:`list` of :obj:`float`
        list of summary statistics used
    rng : numpy.random._generator.Generator
        random number generator to be used when sampling
    cluster_data_values : :obj:`list` of :obj:`float`
        data in the current cluster
    hypers : :obj:`list` of :obj:`float`
        model hyperparameters

    Returns
    -------
    :obj:`list` of :obj:`list` of :obj:`float`
        sampled state and updated summary statistics
    """
    # only the case when card != 0, when card == 0 draw is called from c++
    mu = state[0]
    lam = state[1]
    curr_unc_params = [mu, np.log(lam)]
    prop_unc_params = _propose_rwmh(curr_unc_params, hypers, rng)
    log_target_prop = _eval_prior_lpdf_unconstrained(prop_unc_params, hypers) + _eval_like_lpdf_unconstrained(
        prop_unc_params, False, sum_stats, cluster_data_values)
    log_target_curr = _eval_prior_lpdf_unconstrained(curr_unc_params, hypers) + _eval_like_lpdf_unconstrained(
        curr_unc_params, True, sum_stats, cluster_data_values)
    log_a_rate = log_target_prop - log_target_curr
    log_alpha = np.log(ss.uniform.rvs(loc=0, scale=1, size=1, random_state=rng))
    if log_alpha < log_a_rate:
        state[0] = prop_unc_params[0]
        state[1] = np.exp(prop_unc_params[1])
        sum_stats[0] = sum_stats[1]  # sum_abs_diff_curr = sum_abs_diff_prop

    return [state, sum_stats]


def _propose_rwmh(curr_unc_params, hypers, rng):
    """ Computes a proposal state through a Random Walk step for the Metropolis Hastings algorithm

    Parameters
    ----------
    curr_unc_params : :obj:`list` of :obj:`float`
        mu and log(lam)
    hypers : :obj:`list` of :obj:`float`
        model hyperparameters
    rng : numpy.random._generator.Generator
        random number generator to be used when sampling
    """
    mean_var = hypers[4]
    log_scale_var = hypers[5]
    candidate_mean = curr_unc_params[0] + ss.norm.rvs(loc=0, scale=np.sqrt(mean_var), size=1, random_state=rng)
    candidate_log_scale = curr_unc_params[1] + ss.norm.rvs(loc=0, scale=np.sqrt(log_scale_var), size=1,
                                                           random_state=rng)
    proposal = [candidate_mean, candidate_log_scale]
    return proposal


def _eval_prior_lpdf_unconstrained(unc_params, hypers):
    """

    Parameters
    ----------
    curr_unc_params : :obj:`list` of :obj:`float`
        mu and log(lam)
    hypers : :obj:`list` of :obj:`float`
        model hyperparameters
    """
    mu0 = hypers[0]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    mu_ = unc_params[0]
    log_scale_ = unc_params[1]
    scale_ = np.exp(log_scale_)
    return ss.norm.logpdf(mu_, loc=mu0, scale=np.sqrt(hypers[1])) + \
           ss.invgamma.logpdf(scale_, a=alpha0, loc=0, scale=1 / beta0) + \
           log_scale_


def _eval_like_lpdf_unconstrained(unc_params, is_current, sum_stats, cluster_data_values):
    """

    Parameters
    ----------
    curr_unc_params : :obj:`list` of :obj:`float`
        mu and log(lam)
    is_current : bool
        if True, sum_stats[0] is used, if False, sum_stats[1] is computed and used
    sum_stats : :obj:`list` of :obj:`float`
        list of summary statistics used
    cluster_data_values : :obj:`list` of :obj:`float`
        data in the current cluster
    """
    mu_ = unc_params[0]
    log_scale_ = unc_params[1]
    scale = np.exp(log_scale_)
    diff_sum = 0
    if is_current:
        diff_sum = sum_stats[0]
    else:
        for elem in cluster_data_values:
            diff_sum += abs(elem[0] - mu_)
        sum_stats[1] = diff_sum
    return np.log(0.5 / scale) + (-0.5 / scale * diff_sum)
