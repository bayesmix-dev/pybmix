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


def draw(state, hypers, rng):
    #    s_mean = state[0]
    s_var = state[1]
    h_mean = hypers[0]
    h_var_scaling = hypers[1]
    h_shape = hypers[2]
    h_scale = hypers[3]
    r1 = ss.norm.rvs(h_mean, np.sqrt(s_var / h_var_scaling), random_state=rng)
    r2 = 1 / (ss.gamma.rvs(h_shape, random_state=rng) / h_scale) # Inverse gamma of shape=(h_shape) and rate=(1/h_scale)
    return [r1, r2]
