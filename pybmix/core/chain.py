import logging
import numpy as np
from google.protobuf.pyext._message import RepeatedScalarContainer

from pybmix.utils.proto_utils import get_field
import pybmix.proto.matrix_pb2 as matrix_pb2


class MmcmChain(object):
    """This class represents an MCMC chain obtained by running the algorithm
    for a MixtureModel. It saves the visited states during the MCMC iterations
    and has a useful 'extract' method to get the chain of one (possibly
    vector or matrix-valued) parameter in a numpy array.

    Parameters
    ----------
    serialized_chain: list of bytes
        serialized protobuf messages representing the states of the MCMC
    objtype: 
    """
    def __init__(self, serialized_chain, objtype, deserialize=True):
        self.objtype = objtype
        self.serialized_chain = serialized_chain
        self.chain = None
        if deserialize:
            self.chain = np.array(
                [self._deserialize(x) for x in serialized_chain])

    def extract(self, param_name, to_arviz=False):
        """Extracts the chain relative to 'param_name' in a numpy format
        If 'param_name' is in a nested field, use the dot syntax to
        join fields.

        Parameters
        ----------
        param_name: string
            the name of the parameter to extract
        to_arviz: bool (default False)
            if True, converts the chain to an instance of
            arviz.data.inference_data.InferenceData

        Example
        -------
            Suppose that the chain is a list of MarginalState messages and 
            we want to get the cardinality of the first cluster
        >>> chain = mixture_model.get_chain()
        >>> card_chain = chain.extract("cluster_states[0].cardinality")
        """
        chain = self.chain if self.chain is not None else self.serialized_chain
        extractor = self._get_extractor(param_name)
        out = None

        # we need to perform these checks because we're checking also
        # for base classes
        if isinstance(extractor(chain[0]), RepeatedScalarContainer):
            out = self._extract_repeated(chain, extractor)
        elif isinstance(extractor(chain[0]), matrix_pb2.Vector):
            out = self._extract_vector(chain, extractor)
        elif isinstance(extractor(chain[0]), matrix_pb2.Matrix):
            out = self._extract_matrix(chain, extractor)
        else:
            try:
                out = np.array([extractor(x) for x in chain])
            except Exception as e:
                logging.error(e)

        if out is not None and to_arviz:
            out = self.to_arviz(param_name, chain)

        return out
    
    def _extract_repeated(self, chain, extractor):
        first = extractor(chain[0])
        out = np.empty((len(chain), len(first)), dtype=type(first[0]))
        out[0, :] = first
        for i in range(1, len(chain)):
            out[i, :] = extractor(chain[i])
        return out

    def _extract_vector(self, chain, extractor):
        def to_numpy(msg):
           return np.array(msg.data) 

        first = extractor(chain[0])
        out = np.empty((len(chain), first.size))
        for i in range(len(chain)):
            out[i, :] = to_numpy(extractor(chain[i]))
        return out

    def _extract_matrix(self, chain, extractor):
        def to_numpy(msg):
            order = "T" if msg.rowmajor else "F"
            return np.array(msg.data).reshape(msg.rows, msg.cols, order=order)
        
        first = extractor(chain[0])
        out = np.empty((len(chain), first.rows, first.cols))
        for i in range(len(chain)):
            out[i, :, :] = to_numpy(extractor(chain[i]))
        return out


    def _get_extractor(self, param_name):
        if self.chain is None:
            def extractor(x): return get_field(self._deserialize(x), param_name)
        else:
            def extractor(x): return get_field(x, param_name)

        return extractor
    
    @staticmethod
    def to_arviz(name, chain):
        import arviz as az

        if len(chain.shape) == 1:
            chain = chain.reshape(-1, 1)

        return az.convert_to_inference_data({name: chain[np.newaxis, :, :]})

    def _deserialize(self, bytes):
        out = self.objtype()
        out.ParseFromString(bytes)
        return out
