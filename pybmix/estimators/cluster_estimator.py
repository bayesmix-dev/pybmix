from pybmix.core.mixture_model import MixtureModel
from pybmix.core.pybmixcpp import _minbinder_cluster_estimate

class ClusterEstimator(object):
    """
    Computes cluster estimates from a MixtureModel.
    
    Parameters
    ----------
    mixture_model: an instance of MixtureModel
        the fitted mixture, assumes that 'run_mcmc' has called
    loss: string
        the loss function to use. Currently supports only the Binder loss 
        function with equal missclassification cost
    method: string
        the method to find the point estimates. Currently supports
        only the 'samples' method, that looks for the best partition among the
        ones visited by the MCMC sampler.
    """
    def __init__(self, mixture_model: MixtureModel, loss="binder_equal",
                 method="samples"):
        self.model = mixture_model
        self.chain = self.model.get_chain()
        self.loss = loss
        self.method = method

    def get_point_estimate(self):
        if self.method == "samples" and self.loss == "binder_equal":
            return _minbinder_cluster_estimate(
                self.chain.extract("cluster_allocs"))
        
        else:
            raise ValueError(
                "cluster point estimate only supports method='samples' and "
                "loss='binder_equal' for the moment")
    
