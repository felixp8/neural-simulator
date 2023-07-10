import numpy as np
import sklearn.decomposition as skdecomp
import sklearn.manifold as skman
import functools
import inspect

from .manifold import fn as manifold_functions
from .neural_sampling import fn as neural_functions


def get_dim_reduction_function(
    reduced_dim,
    dim_reduction_method,
    dim_reduction_kwargs,
):
    if dim_reduction_method.lower() == "pca":
        estimator = skdecomp.PCA(n_components=reduced_dim, **dim_reduction_kwargs)
        dim_reduction_fn = estimator.fit_transform
    elif hasattr(skdecomp, dim_reduction_method) or hasattr(skman, dim_reduction_method):
        module = skdecomp if hasattr(skdecomp, dim_reduction_method) else skman
        estimator_class = getattr(module, dim_reduction_method)
        estimator_init_kwargs = {
            key: val for key, val in dim_reduction_kwargs.items()
            if key in inspect.signature(estimator_class.__init__).parameters
        }
        estimator = estimator_class(n_components=reduced_dim, **estimator_init_kwargs)
        estimator_call_kwargs = {
            key: val for key, val in dim_reduction_kwargs.items()
            if key not in estimator_init_kwargs
        }
        dim_reduction_fn = functools.partial(estimator.fit_transform, **estimator_call_kwargs)
    else: # TODO: support callable `dim_reduction_method`
        raise AssertionError
    return dim_reduction_fn


def get_manifold_embedding_function(
    manifold_embedding,
    manifold_kwargs,
):
    if hasattr(manifold_functions, manifold_embedding):
        manifold_embedding_fn = functools.partial(
            getattr(manifold_functions, manifold_embedding),
            **manifold_kwargs,
        )
    else:
        raise AssertionError
    return manifold_embedding_fn


def get_neural_sampling_function(
    neural_sampling,
    neural_sampling_kwargs,
):
    if hasattr(neural_functions, neural_sampling):
        neural_sampling_fn = functools.partial(
            getattr(neural_functions, neural_sampling),
            **neural_sampling_kwargs,
        )
    else:
        raise AssertionError
    return neural_sampling_fn


def export_to_nwb(*args, **kwargs):
    raise NotImplementedError
