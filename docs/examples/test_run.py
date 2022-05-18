import os
import sys
HERE = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(HERE, "../../")
sys.path.insert(0, os.path.realpath(BUILD_DIR))

import numpy as np
import matplotlib.pyplot as plt
from pybmix.core.mixing import DirichletProcessMixing
from pybmix.core.hierarchy import PythonHierarchy
from pybmix.core.mixture_model import MixtureModel
np.random.seed(2021)


def sample_from_mixture(weights, means, sds, n_data):
    n_comp = len(weights)
    clus_alloc = np.random.choice(np.arange(n_comp), p=[0.5, 0.5], size=n_data)
    return np.random.normal(loc=means[clus_alloc], scale=sds[clus_alloc])


y = sample_from_mixture(np.array([0.5, 0.5]), np.array([-3, 3]), np.array([1, 1]), 200)


plt.hist(y)
plt.show()

mixing = DirichletProcessMixing(total_mass=5)
hierarchy = PythonHierarchy()
hierarchy.make_default_fixed_params(y, 2)
mixture = MixtureModel(mixing, hierarchy)

mixture.run_mcmc(y, algorithm="Neal2", niter=2000, nburn=1000)

from pybmix.estimators.density_estimator import DensityEstimator

grid = np.linspace(-6, 6, 500)
dens_est = DensityEstimator(mixture)
densities = dens_est.estimate_density(grid)

plt.hist(y, density=True)
plt.plot(grid, np.mean(densities, axis=0), lw=3, label="predictive density")
idxs = [5, 100, 300]

for idx in idxs:
    plt.plot(grid, densities[idx, :], "--", label="iteration: {0}".format(idx))
    plt.legend()
    plt.show()

