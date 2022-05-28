import numpy as np
import scipy.stats as ss


def like_lpdf(x, state):
    mean = state[0]
    var = state[1]
    return ss.norm.logpdf(x, mean, var)


def marg_lpdf(x, hypers):
    mean = hypers[0]
    var_scaling = hypers[1]
    shape = hypers[2]
    scale = hypers[3]
    sig_n = np.sqrt(scale * (var_scaling + 1) / (shape * var_scaling))
    return ss.t.logpdf(x, 2 * shape, mean, sig_n)


def initialize_state(hypers):
    mean = hypers[0]
    #    var_scaling = hypers[1]
    shape = hypers[2]
    scale = hypers[3]
    return [mean, scale / (shape + 1)]


def initialize_hypers():
    return [1, 1, 1, 1]


def draw(state, hypers, rng):
    #    s_mean = state[0]
    s_var = state[1]
    h_mean = hypers[0]
    h_var_scaling = hypers[1]
    h_shape = hypers[2]
    h_scale = hypers[3]
    r1 = ss.norm.rvs(h_mean, np.sqrt(s_var / h_var_scaling), random_state=rng)
    r2 = 1 / (ss.gamma.rvs(h_shape,
                           random_state=rng) / h_scale)  # Inverse gamma of shape=(h_shape) and rate=(1/h_scale)
    return [r1, r2]


def compute_posterior_hypers(card, hypers, sum_stats):
    data_sum = sum_stats[0]
    data_sum_squares = sum_stats[1]
    h_mean = hypers[0]
    h_var_scaling = hypers[1]
    h_shape = hypers[2]
    h_scale = hypers[3]
    if card == 0:
        return hypers
    post_hypers = [0] * len(hypers)
    y_bar = data_sum / card
    sstat = data_sum_squares - card * (y_bar ** 2)
    post_hypers[0] = (h_var_scaling * h_mean + data_sum) / (h_var_scaling + card)  # mean
    post_hypers[1] = h_var_scaling + card  # var_scaling
    post_hypers[2] = h_shape + 0.5 * card  # shape
    num = 0.5 * h_var_scaling * card * ((y_bar - h_mean) ** 2)
    denom = card + h_var_scaling
    post_hypers[3] = h_scale + 0.5 * sstat + num / denom  # scale
    return post_hypers


def update_summary_statistics(x, add, sum_stats):
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
    return [data_sum, data_sum_squares]


def clear_summary_statistics(sum_stats):
    data_sum = 0
    data_sum_squares = 0
    return [data_sum, data_sum_squares]


# NON-CONJUGATE
def sample_full_cond(iter_, accepted_, state, sum_stats, rng, curr_vals, hypers):
    # only the case when card != 0
    iter_ += 1
    curr_unc_params = [state[0], np.log(state[1])]
    prop_unc_params = propose_rwmh(curr_unc_params, hypers, rng)
    log_target_prop = eval_prior_lpdf_unconstrained(prop_unc_params, hypers) + eval_like_lpdf_unconstrained(
        prop_unc_params,
        False)
    log_target_curr = eval_prior_lpdf_unconstrained(curr_unc_params, hypers) + eval_like_lpdf_unconstrained(
        curr_unc_params,
        True)
    log_a_rate = log_target_prop - log_target_curr
    alpha = ss.uniform.rvs(0, 1, size=1, random_state=rng)
    if alpha < log_a_rate:
        accepted_ += 1
        state[0] = prop_unc_params[0]
        state[1] = np.exp(prop_unc_params[1])
        sum_stats[0] = sum_stats[1]  # sum_abs_diff_curr = sum_abs_diff_prop

    return [iter_, accepted_, state, sum_stats]


def propose_rwmh(curr_vals, hypers, rng):
    candidate_mean = curr_vals[0] + ss.norm.rvs(0, np.sqrt(hypers[4]), size=1, random_state=rng)
    candidate_log_scale = curr_vals[1] + ss.norm.rvs(0, np.sqrt(hypers[5]), size=1, random_state=rng)
    proposal = [candidate_mean, candidate_log_scale]
    return proposal


def eval_prior_lpdf_unconstrained(unconstrained_parameters, hypers):
    mu = unconstrained_parameters[0]
    log_scale = unconstrained_parameters[1]
    scale = np.exp(log_scale)
    return ss.norm.pdf(mu, hypers[0], np.sqrt(hypers[1])) + ss.invgamma(scale, a=hypers[2], loc=0,
                                                                        scale=hypers[3]) + log_scale


def eval_like_lpdf_unconstrained(unconstrained_parameters, is_current, sum_stats, cluster_data_values):
    mean = unconstrained_parameters[0]
    log_scale = unconstrained_parameters[1]
    scale = np.exp(log_scale)
    diff_sum = 0
    if is_current:
        diff_sum = sum_stats[0]
    else:
        for elem in cluster_data_values:
            diff_sum += abs(elem[0, 0] - mean)
        sum_stats[1] = diff_sum
    return np.log(0.5 / scale) + (-0.5 / scale * diff_sum)
