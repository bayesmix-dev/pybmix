import importlib
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
CURR_DIR = os.path.join(HERE, "../../")
EX_DIR = os.path.join(CURR_DIR, "docs/examples/")
sys.path.insert(0, os.path.realpath(EX_DIR))


def like_lpdf(x, state):
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.like_lpdf(x, state)


def initialize_state(hypers):
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.initialize_state(hypers)


def initialize_hypers():
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.initialize_hypers()


def draw(state, hypers, rng):
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.draw(state, hypers, rng)


def update_summary_statistics(x, add, sum_stats, state, cluster_data_values):
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.update_summary_statistics(x, add, sum_stats, state, cluster_data_values)


def clear_summary_statistics(sum_stats):
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.clear_summary_statistics(sum_stats)


# NON-CONJUGATE
def sample_full_cond(state, sum_stats, rng, curr_vals, hypers):
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.sample_full_cond(state, sum_stats, rng, curr_vals, hypers)


def marg_lpdf(x, hypers):
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.marg_lpdf(x, hypers)


def compute_posterior_hypers(card, hypers, sum_stats):
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.compute_posterior_hypers(card, hypers, sum_stats)


def is_conjugate():
    true_name = os.getenv('HIER_NAME')
    module = importlib.import_module(true_name)
    return module.is_conjugate()
