import os
import importlib

def like_lpdf(x, state):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.like_lpdf(x, state)


def initialize_state(hypers):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.initialize_state(hypers)


def initialize_hypers():
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.initialize_hypers()


def draw(state, hypers, rng):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.draw(state, hypers, rng)


def update_summary_statistics(x, add, sum_stats, state, cluster_data_values):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.update_summary_statistics(x, add, sum_stats, state, cluster_data_values)


def clear_summary_statistics(sum_stats):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.clear_summary_statistics(sum_stats)


# NON-CONJUGATE
def sample_full_cond(state, sum_stats, rng, curr_vals, hypers):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.sample_full_cond(state, sum_stats, rng, curr_vals, hypers)


def propose_rwmh(curr_vals, hypers, rng):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.propose_rwmh(curr_vals, hypers, rng)


def eval_prior_lpdf_unconstrained(unconstrained_parameters, hypers):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.eval_prior_lpdf_unconstrained(unconstrained_parameters, hypers)


def eval_like_lpdf_unconstrained(unconstrained_parameters, is_current, sum_stats, cluster_data_values):
    true_name = os.getenv('HIER_NC_NAME')
    module = importlib.import_module(true_name)
    return module.eval_like_lpdf_unconstrained(unconstrained_parameters, is_current, sum_stats, cluster_data_values)