import numpy as np
import scipy.stats as ss


def like_lpdf(x, state):
    print("*************** fun1 - like_lpdf *************")
    mean = state[0]
    scale = state[1]
    return ss.norm.laplace.pdf(x, mean, scale)


def initialize_state(hypers):
    print("*************** fun1 - initialize_state *************")
    mean = hypers[0]
    shape = hypers[2]
    scale = hypers[3]
    return [mean, scale / (shape + 1)]


def initialize_hypers():
    print("*************** fun1 - initialize_hypers *************")
    return [1, 1, 1, 1]


def draw(state, hypers, rng):
    print("*************** fun1 - draw *************")
    r1 = ss.norm.rvs(hypers[0], np.sqrt(hypers[3]), random_state=rng)
    r2 = ss.invgamma.rvs(hypers[2], 1 / hypers[3], random_state=rng)

    return [r1, r2]


def update_summary_statistics(x, add, sum_stats, state, cluster_data_values):
    print("*************** fun1 - update_summary_statistics*************")
    if not len(sum_stats):
        sum_stats = [0, 0]
    if add:
        sum_stats[0] += abs(state[0] - x[0])
        cluster_data_values.append(x)
    else:
        sum_stats[0] -= abs(state[0] - x[0])
        ind = np.where(cluster_data_values == x)
        np.delete(cluster_data_values, ind)
    return [sum_stats, cluster_data_values]


def clear_summary_statistics(sum_stats):
    print("*************** fun1 - clear_summary_statistics *************")
    return [0, 0]


# NON-CONJUGATE
def sample_full_cond(iter_, accepted_, state, sum_stats, rng, curr_vals, hypers):
    print("*************** fun1 - sample_full_cond *************")
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
    print("*************** fun1 - propose_rwmh *************")
    candidate_mean = curr_vals[0] + ss.norm.rvs(0, np.sqrt(hypers[4]), size=1, random_state=rng)
    candidate_log_scale = curr_vals[1] + ss.norm.rvs(0, np.sqrt(hypers[5]), size=1, random_state=rng)
    proposal = [candidate_mean, candidate_log_scale]
    return proposal


def eval_prior_lpdf_unconstrained(unconstrained_parameters, hypers):
    print("*************** fun1 - eval_prior_lpdf_uncostrained *************")
    mu = unconstrained_parameters[0]
    log_scale = unconstrained_parameters[1]
    scale = np.exp(log_scale)
    return ss.norm.pdf(mu, hypers[0], np.sqrt(hypers[1])) + ss.invgamma(scale, a=hypers[2], loc=0,
                                                                        scale=hypers[3]) + log_scale


def eval_like_lpdf_unconstrained(unconstrained_parameters, is_current, sum_stats, cluster_data_values):
    print("*************** fun1 - eval_like_lpdf_uncostrained *************")
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
