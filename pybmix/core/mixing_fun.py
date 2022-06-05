import os

import numpy as np
import scipy.stats as ss
import importlib


def initialize_state(hypers):
    total_mass = 2
    return [total_mass]


def update_state(prior_values, allocation_size, unique_values):
    return


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