import numpy as np

from pybmix.core.mixture_model import MixtureModel


class DensityEstimator(object):
    def __init__(self, mixture_model: MixtureModel):
        self.model = mixture_model

    @staticmethod
    def make_grid(intervals, npoints=100):
        """
        Returns an n-dimensional equispaced grid, with bounds specified by the
        intervals

        Parameters
        ----------
        intervals: np.array of shape (dimension, 2)
            the extremes of the intervals
        npoints: int
            the number of points of each dimension of the grid
        """
        if intervals.ndim == 1:
            return np.linspace(intervals[0], intervals[1], npoints)

        marginal_grids = []
        for i in range(intervals.shape[0]):
            marginal_grids.append(
                np.linspace(intervals[i, 0], intervals[i, 1], npoints))

        return np.meshgrid(*marginal_grids)

    def estimate_density(self, grid, mean=False):
        """Estimate the mixture density over a fixed grid.
        If mean is False, return a matrix of shape (num_mcmc_iter, len(grid))
        else returns a vector of length (len(grid)).
        
        Parameters
        ----------
        grid: np.array of shape (num_points, num_dimensions)
            a grid of points where to evaluate the mixture density
        mean: bool
            if True, returns only the mean of the densities
        """
        dens = self.model._algo.eval_density((grid))
        if mean:
            dens = np.mean(dens, axis=0)

        return dens
