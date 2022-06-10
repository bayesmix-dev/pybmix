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
    print("---------------This is hierarchy_c_implementation_1-----------------")
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
