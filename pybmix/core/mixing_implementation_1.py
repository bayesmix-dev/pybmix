import numpy as np


def initialize_state():
    total_mass = 5
    return [total_mass]


def update_state(prior_values, allocation_size, unique_values):
    return [5]


def mass_existing_cluster(n, n_clust, log, propto, hier_card, state):
    total_mass = state[0]
    if log:
        out = np.log(hier_card)
        if not propto:
            out -= np.log(n + total_mass)
    else:
        out = hier_card
        if not propto:
            out /= (n + total_mass)
    return out


def mass_new_cluster(n, n_clust, log, propto, state):
    total_mass = state[0]
    if log:
        out = np.log(total_mass)
        if not propto:
            out -= np.log(n + total_mass)
    else:
        out = total_mass
        if not propto:
            out /= (n + total_mass)
    return out


def is_conditional():
    return False


def mixing_weights(log, propto, state):
    pass