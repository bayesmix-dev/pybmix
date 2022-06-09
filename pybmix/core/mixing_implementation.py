import os

import numpy as np
import scipy.stats as ss
import importlib


def initialize_state():
    true_name = os.getenv('MIX_NAME')
    module = importlib.import_module(true_name)
    return module.initialize_state()


def update_state(prior_values, allocation_size, unique_values):
    true_name = os.getenv('MIX_NAME')
    module = importlib.import_module(true_name)
    return module.update_state(prior_values, allocation_size, unique_values)


def mass_existing_cluster(n, n_clust, log, propto, hier_card, state):
    true_name = os.getenv('MIX_NAME')
    module = importlib.import_module(true_name)
    return module.mass_existing_cluster(n, n_clust, log, propto, hier_card, state)


def mass_new_cluster(n, n_clust, log, propto, state):
    true_name = os.getenv('MIX_NAME')
    module = importlib.import_module(true_name)
    return module.mass_new_cluster(n, n_clust, log, propto, state)


def is_conditional():
    true_name = os.getenv('MIX_NAME')
    module = importlib.import_module(true_name)
    return module.is_conditional()


def mixing_weights(log, propto, state):
    true_name = os.getenv('MIX_NAME')
    module = importlib.import_module(true_name)
    return module.mixing_weights(log, propto, state)
