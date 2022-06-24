import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(HERE, "../../")
CORE_DIR = os.path.join(BUILD_DIR, "pybmix/core/")
sys.path.insert(0, os.path.realpath(BUILD_DIR))
sys.path.insert(0, os.path.realpath(CORE_DIR))

import numpy as np
import matplotlib.pyplot as plt
from pybmix.core.mixing import DirichletProcessMixing, PythonMixing
from pybmix.core.hierarchy import PythonHierarchy
from pybmix.core.mixture_model import MixtureModel
from pybmix.estimators.density_estimator import DensityEstimator


np.random.seed(2021)


def sample_from_mixture(weights, means, sds, n_data):
    n_comp = len(weights)
    clus_alloc = np.random.choice(np.arange(n_comp), p=[0.5, 0.5], size=n_data)
    return np.random.normal(loc=means[clus_alloc], scale=sds[clus_alloc])


y = sample_from_mixture(np.array([0.5, 0.5]), np.array([-3, 3]), np.array([1, 1]), 200)

plt.hist(y, bins=20)
plt.show()

mixing = PythonMixing(mix_implementation="DP_mixing",state=[5], prior=[0])  # Python implementation of DP mixing
# mixing = DirichletProcessMixing(total_mass=5)  # DP mixing

hierarchy = PythonHierarchy("NNIG_Hierarchy_NGG")

# Checking that other classes work too
# hierarchy = UnivariateNormal()
# hierarchy.make_default_fixed_params(y,2)

mixture = MixtureModel(mixing, hierarchy)

niter = 110
nburn = 10
mixture.run_mcmc(y, algorithm="Neal2", niter=niter, nburn=nburn)


grid = np.linspace(-6, 6, 500)
dens_est = DensityEstimator(mixture)
densities = dens_est.estimate_density(grid)

plt.hist(y, density=True, bins=20)
plt.plot(grid, np.mean(densities, axis=0), lw=3, label="predictive density")
plt.legend()
plt.show()
idxs = [int((niter - nburn) * 0.1), int((niter - nburn) * 0.2), int((niter - nburn) * 0.9)]

for idx in idxs:
    plt.plot(grid, densities[idx, :], "--", label="iteration: {0}".format(idx))
    plt.legend()
    plt.show()
