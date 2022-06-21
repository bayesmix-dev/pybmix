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
    # print("--------------- This is NNIG_Hierarchy_2 -----------------")
    mean = hypers[0]
    #    var_scaling = hypers[1]
    shape = hypers[2]
    scale = hypers[3]
    return [mean, scale / (shape + 1)]


def initialize_hypers():
    # Get hyperparameters:
    # For mu0
    mu00 = 5.5
    sigma00 = 2.25
    # For lambda0
    alpha00 = 0.2
    beta00 = 0.6
    # For beta0
    a00 = 4.0
    b00 = 2.0
    # For alpha0
    alpha0 = 1.5
    # Set initial values
    mean = mu00
    var_scaling = alpha00 / beta00
    shape = alpha0
    scale = a00 / b00
    return [mean, var_scaling, shape, scale]


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


def update_summary_statistics(x, add, sum_stats, state, cluster_data_values):
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


def clear_summary_statistics(sum_stats):
    data_sum = 0
    data_sum_squares = 0
    return [data_sum, data_sum_squares]


def sample_full_cond(state, sum_stats, rng, curr_vals, hypers):
    pass


def is_conjugate():
    return True


def update_hypers(states, hypers, rng):
    # print("Old Hypers: ", hypers)
    # print("States: ", states)
    # Set the hyperparameters(here set like in bayesmix/resources/tutorial/nnig_ngg.asciipb)
    # For mu0
    mu00 = 5.5
    sig200 = 2.25
    # For lambda0
    alpha00 = 0.2
    beta00 = 0.6
    # For tau0
    a00 = 4.0
    b00 = 2.0
    # Compute posterior hyperparameters
    b_n = 0.0
    num = 0.0
    beta_n = 0.0
    for st in states:
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
    alpha_n = alpha00 + 0.5 * len(states)
    a_n = a00 + len(states) * hypers[2]
    # Update hyperparameters with posterior random Gibbs sampling
    new_mean = ss.norm.rvs(mu_n, sig_n, random_state=rng)
    new_var_scaling = ss.gamma.rvs(a=alpha_n, loc=0, scale=1. / beta_n, random_state=rng)
    new_shape = hypers[2]
    new_scale = ss.gamma.rvs(a=a_n, loc=0, scale=1. / b_n, random_state=rng)
    # print("New Hypers: ", [new_mean, new_var_scaling, new_shape, new_scale])
    return [new_mean, new_var_scaling, new_shape, new_scale]
