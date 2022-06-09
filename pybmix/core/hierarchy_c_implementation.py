import os

import numpy as np
import scipy.stats as ss
import importlib


def like_lpdf(x, state):
    true_name = os.getenv('HIER_C_NAME')
    module = importlib.import_module(true_name)
    return module.like_lpdf(x, state)


def marg_lpdf(x, hypers):
    true_name = os.getenv('HIER_C_NAME')
    module = importlib.import_module(true_name)
    return module.marg_lpdf(x, hypers)


def initialize_state(hypers):
    true_name = os.getenv('HIER_C_NAME')
    module = importlib.import_module(true_name)
    return module.initialize_state(hypers)


def initialize_hypers():
    true_name = os.getenv('HIER_C_NAME')
    module = importlib.import_module(true_name)
    return module.initialize_hypers()


def draw(state, hypers, rng):
    true_name = os.getenv('HIER_C_NAME')
    module = importlib.import_module(true_name)
    return module.draw(state, hypers, rng)


def compute_posterior_hypers(card, hypers, sum_stats):
    true_name = os.getenv('HIER_C_NAME')
    module = importlib.import_module(true_name)
    return module.compute_posterior_hypers(card, hypers, sum_stats)


def update_summary_statistics(x, add, sum_stats):
    true_name = os.getenv('HIER_C_NAME')
    module = importlib.import_module(true_name)
    return module.update_summary_statistics(x, add, sum_stats)


def clear_summary_statistics(sum_stats):
    true_name = os.getenv('HIER_C_NAME')
    module = importlib.import_module(true_name)
    return module.clear_summary_statistics(sum_stats)
