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
hyperparameters.
"""

import numpy as np
import scipy.stats as ss


def is_conjugate():
    """
    :return: True for conjugate, False for non-conjugate hierarchies
    """
    return True


def like_lpdf(x, state):
    """
    Likelihood log-density
    :param list x: point in which lpdf is evaluated
    :param list state: model parameters
    """
    mu = state[0]
    sig = state[1]
    return ss.norm.logpdf(x, mu, np.sqrt(sig))


def marg_lpdf(x, hypers):
    """
    Marginal log density
    :param list x: point in which lpdf is evaluated
    :param list hypers: model hyperparameters
    """
    mu0 = hypers[0]
    lambda0 = hypers[1]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    sig_n = np.sqrt(beta0 * (lambda0 + 1) / (alpha0 * lambda0))
    return ss.t.logpdf(x, 2 * alpha0, mu0, sig_n)


def initialize_state(hypers):
    """
    :param list hypers: model hyperparameters
    :return: initial value of the state
    """
    mu0 = hypers[0]
    alpha0 = hypers[2]
    beta0 = hypers[3]
    return [mu0, beta0 / (alpha0 + 1)]


def initialize_hypers():
    """
    In this example NGG prior is assumed on the hyperparameters
    mu0 ~ Normal(mu00, sigma00)
    lambda0 ~ Gamma(alpha00, beta00)
    beta0 ~ Gamma(a00, b00)
    alpha0 is fixed value
    :return: initial value of the hyperparameters
    """
    mu00 = 1
    sigma00 = 2.25
    alpha00 = 0.2
    beta00 = 0.6
    a00 = 4.0
    b00 = 2.0
    alpha0 = 1.5

    mean = mu00
    var_scaling = alpha00 / beta00
    shape = alpha0
    scale = a00 / b00
    return [mean, var_scaling, shape, scale]


def draw(state, hypers, rng):
    """
    Samples values for the state parameters
    :param list state: model parameters
    :param hypers: model hyperparameters
    :param rng: random number generator to be used when sampling
    :return: sampled state values
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
    """
    Computation of posterior hyperparameters
    :param int card: cardinality of the cluster
    :param hypers: model hyperparameters
    :param list sum_stats: list of summary statistics used
    :return: list: posterior hyperparameters
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
    """
    Updates cluster statistics when a datum is added or removed from it,
    the statistics appears in the sampling of the full conditionals of the model for non-conjugate hierarchies and
    in the computation of the posterior hyperparameters for conjugate hierarchies.
    In this model, the summary statistics are the current and proposed values of the sum of absolute differences,

    :param list x: datum (univariate)
    :param bool add: if True, the datum has to be added to the cluster, if False, it has to be removed from the cluster
    :param list sum_stats: list of summary statistics used
    :param list state: model parameters
    :param list cluster_data_values: data in the current cluster
    :return: list[list]: updated summary statistics and cluster data values
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


def update_hypers(state, hypers, rng):
    """
    Updates the hyperparameters according to the NGG prior assumption:
    mu0 ~ Normal(mu00, sigma00)
    lambda0 ~ Gamma(alpha00, beta00)
    beta0 ~ Gamma(a00, b00)
    alpha0 is a fixed value
    :param list state: model parameters
    :param list hypers: model hyperparameters
    :param list state: model parameter
    :return: updated hyperparameters
    """

    mu00 = 1
    sig200 = 2.25
    alpha00 = 0.2
    beta00 = 0.6
    a00 = 4.0
    b00 = 2.0

    b_n = 0.0
    num = 0.0
    beta_n = 0.0
    for st in state:
        mean = st[0]
        var = st[1]
        b_n += 1 / var
        num += mean / var
        beta_n += (hypers[0] - mean) * (hypers[0] - mean) / var
    var = hypers[1] * b_n + 1 / sig200
    b_n += b00
    num = hypers[1] * num + mu00 / sig200
    beta_n = beta00 + 0.5 * beta_n
    sig_n = 1 / var
    mu_n = num / var
    alpha_n = alpha00 + 0.5 * len(state)
    a_n = a00 + len(state) * hypers[2]
    # Update hyperparameters with posterior random Gibbs sampling
    new_mean = ss.norm.rvs(mu_n, sig_n, random_state=rng)
    new_var_scaling = ss.gamma.rvs(a=alpha_n, loc=0, scale=1. / beta_n, random_state=rng)
    new_shape = hypers[2]
    new_scale = ss.gamma.rvs(a=a_n, loc=0, scale=1. / b_n, random_state=rng)
    return [new_mean, new_var_scaling, new_shape, new_scale]
