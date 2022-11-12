"""
This file contains the implementation of the EPPF induced by the Dirithclet process (DP)
introduced in Ferguson (1973), see also Sethuraman (1994).
The EPPF induced by the DP depends on a `totalmass` parameter M.
Given a clustering of n elements into k clusters, each with cardinality
n_j (j = 1, ..., k) the EPPF of the DP gives the following
probabilities for the cluster membership of the (n+1)-th observation:

   p(j-th cluster | ...) = n_j / (n + M)
   p(new cluster | ...) = M / (n + M)

The state is solely composed of M, but we also store log(M) for efficiency
reasons. For more information about the class, please refer instead to base
classes, `AbstractMixing` and `BaseMixing`.
"""

import numpy as np


def initialize_state():
    """ Initializing the state (total mass parameter M)

    Returns
    -------
    float
        initial value of the total mass parameter M
    """
    total_mass = 5
    return [total_mass]


def update_state(state, prior_values, allocation_size, unique_values):
    """ Updating the state (total mass parameter M)

    Returns
    -------
    float
        updated value of the total mass parameter M
    """
    return state


def mass_existing_cluster(n, n_clust, log, propto, hier_card, state):
    """ Returns probability mass for an old cluster (for marginal mixings only)

    Parameters
    ----------
    n : int
        Total dataset size
    n_clust : int
        Number of clusters
    log : bool
        Whether to return logarithm-scale values or not
    propto : bool
        Whether to include normalizing constants or not
    hier_card : int
        Number of data points assigned to the `Hierarchy` object representing the cluster
    state : :obj:`list` of :obj:`float`
        State (total mass parameter M) values for each cluster

    Returns
    -------
    float
        probability mass
    """
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
    """ Returns probability mass for a new cluster (for marginal mixings only)

    Parameters
    ----------
    n : int
        Total dataset size
    n_clust : int
        Number of clusters
    log : bool
        Whether to return logarithm-scale values or not
    propto : bool
        Whether to include normalizing constants or not
    state : :obj:`list` of :obj:`float`
        State (total mass parameter M) values for each cluster

    Returns
    -------
    float
        probability mass
    """
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
    """

    Returns
    -------
    :bool:
        True for conditional, False for marginal mixings
    """
    return False
