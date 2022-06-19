import numpy as np
import scipy.stats as ss


def like_lpdf(x, state):
    mean = state[0]
    scale = state[1]
    return ss.laplace.logpdf(x, mean, scale)


def initialize_state(hypers):
    # print("----------- This is LapNIG_Hierarchy_2 -----------------")
    mean = hypers[0]
    # var = hypers[1]
    shape = hypers[2]
    scale = hypers[3]
    # mh_mean_var = hypers[4]
    # mh_log_scale_var = hypers[5]
    return [mean, scale / (shape + 1)]


def initialize_hypers():
    return [0, 10, 2, 1, 10, 1]


def draw(state, hypers, rng):
    mean = hypers[0]
    var = hypers[1]
    shape = hypers[2]
    scale = hypers[3]
    # mh_mean_var = hypers[4]
    # mh_log_scale_var = hypers[5]
    r1 = ss.norm.rvs(mean, np.sqrt(var), random_state=rng)
    r2 = ss.invgamma.rvs(a=shape, loc=0, scale=scale, random_state=rng)

    return [r1, r2]  # mean, scale


def update_summary_statistics(x, add, sum_stats, state, cluster_data_values):
    if not len(sum_stats):
        sum_stats = [0, 0]
    if add:
        sum_stats[0] += abs(state[0] - x[0])
        cluster_data_values = np.append(cluster_data_values, x)
    else:
        sum_stats[0] -= abs(state[0] - x[0])
        ind = np.where(cluster_data_values == x)
        cluster_data_values = np.delete(cluster_data_values,ind[0][0],ind[1][0])
    return [sum_stats, cluster_data_values]


def clear_summary_statistics(sum_stats):
    return [0, 0]


# NON-CONJUGATE
def sample_full_cond(state, sum_stats, rng, curr_vals, hypers):
    # only the case when card != 0
    # mean = hypers[0]
    # var = hypers[1]
    # shape = hypers[2]
    # scale = hypers[3]
    # mh_mean_var = hypers[4]
    # mh_log_scale_var = hypers[5]
    curr_unc_params = [state[0], np.log(state[1])]
    prop_unc_params = propose_rwmh(curr_unc_params, hypers, rng)
    log_target_prop = eval_prior_lpdf_unconstrained(prop_unc_params, hypers) + eval_like_lpdf_unconstrained(
        prop_unc_params, False, sum_stats, curr_vals)
    log_target_curr = eval_prior_lpdf_unconstrained(curr_unc_params, hypers) + eval_like_lpdf_unconstrained(
        curr_unc_params, True, sum_stats, curr_vals)
    log_a_rate = log_target_prop - log_target_curr
    log_alpha = np.log(ss.uniform.rvs(0, 1, size=1, random_state=rng))
    if log_alpha < log_a_rate:
        state[0] = prop_unc_params[0]
        state[1] = np.exp(prop_unc_params[1])
        sum_stats[0] = sum_stats[1]  # sum_abs_diff_curr = sum_abs_diff_prop

    return [state, sum_stats]


def propose_rwmh(curr_vals, hypers, rng):
    # mean = hypers[0]
    # var = hypers[1]
    # shape = hypers[2]
    # scale = hypers[3]
    mh_mean_var = hypers[4]
    mh_log_scale_var = hypers[5]
    candidate_mean = curr_vals[0] + ss.norm.rvs(0, np.sqrt(mh_mean_var), size=1, random_state=rng)
    candidate_log_scale = curr_vals[1] + ss.norm.rvs(0, np.sqrt(mh_log_scale_var), size=1, random_state=rng)
    proposal = [candidate_mean, candidate_log_scale]
    return proposal


def eval_prior_lpdf_unconstrained(unconstrained_parameters, hypers):
    mean = hypers[0]
    # var = hypers[1]
    shape = hypers[2]
    scale = hypers[3]
    # mh_mean_var = hypers[4]
    # mh_log_scale_var = hypers[5]
    mu = unconstrained_parameters[0]
    log_scale = unconstrained_parameters[1]
    scale_ = np.exp(log_scale)
    return ss.norm.logpdf(mu, mean, np.sqrt(hypers[1])) + \
           ss.invgamma.logpdf(scale_, a=shape, loc=0, scale=1 / scale) + \
           log_scale


def eval_like_lpdf_unconstrained(unconstrained_parameters, is_current, sum_stats, cluster_data_values):
    mean = unconstrained_parameters[0]
    log_scale = unconstrained_parameters[1]
    scale = np.exp(log_scale)
    diff_sum = 0
    if is_current:
        diff_sum = sum_stats[0]
    else:
        for elem in cluster_data_values:
            diff_sum += abs(elem[0] - mean)
        sum_stats[1] = diff_sum
    return np.log(0.5 / scale) + (-0.5 / scale * diff_sum)


def marg_lpdf(x, hypers):
    pass


def compute_posterior_hypers(card, hypers, sum_stats):
    pass


def is_conjugate():
    return False


def update_hypers(states, hypers, rng):
    return [hypers[0], hypers[1], hypers[2], hypers[3], hypers[4], hypers[5]]
