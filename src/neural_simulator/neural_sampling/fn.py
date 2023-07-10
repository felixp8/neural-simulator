import numpy as np
from typing import Literal, Optional, Union


def exp(
    arr: np.ndarray, # (..., D)
    vscale: Union[float, np.ndarray] = 1., # float or (D,)
    hscale: Union[float, np.ndarray] = 1., # float or (D,)
    offset: Union[float, np.ndarray] = 0., # float or (D,)
):
    # TODO: implicit broadcasting may run into issues if any other arr dims == D
    return vscale * np.exp(arr * hscale + offset)

def relu(
    arr: np.ndarray, 
    scale: Union[float, np.ndarray] = 1., # float or (D,)
    offset: Union[float, np.ndarray] = 0., # float or (D,)
):
    return scale * np.clip(arr + offset, amin=0.)

def softplus(
    arr: np.ndarray,
    scale: Union[float, np.ndarray] = 1., # float or (D,)
    beta: Union[float, np.ndarray] = 1., # float or (D,)
    offset: Union[float, np.ndarray] = 0., # float or (D,)
    # threshold: float = 20., # used in pytorch
):
    return scale / beta * np.log(1 + np.exp(beta * arr + offset))

def sigmoid(
    arr: np.ndarray
    vscale: Union[float, np.ndarray] = 1., # float or (D,)
    hscale: Union[float, np.ndarray] = 1., # float or (D,)
    offset: Union[float, np.ndarray] = 0., # float or (D,)
):
    return vscale / (1. + np.exp(-hscale * arr + offset))


def lin_nonlin(
    points: np.ndarray,
    projection_dim: int,
    proj_weight_dist: Literal["normal", "uniform"] = "uniform",
    proj_weight_params: dict = {},
    nonlinearity: Literal["relu", "exp", "softplus", "sigmoid"] = "exp",
    nonlinearity_params: dict = {},
    noise_dist: Literal["poisson", "normal"],
    noise_params: dict = {},
    sort_dims: bool = False,
    standardize: bool = False,
    seed: Optional[int] = None,
):
    # init rng
    rng = np.random.default_rng(seed)
    # sample linear projection weights
    in_dim = points.shape[-1]
    proj_weights = getattr(rng, proj_weight_dist)(size=(in_dim, projection_dim), **proj_weight_params)
    if sort_dims:
        proj_weights = proj_weights[:, np.argsort(proj_weights[0])]
    # perform linear projection
    activity = points @ proj_weights
    if standardize:
        orig_mean = np.mean(activity, axis=0, keepdims=True)
        orig_std = np.std(activity, axis=0, keepdims=True)
        activity = (activity - orig_mean) / orig_std
    # get nonlinearity func and apply
    try:
        nonlin_fn = eval(nonlinearity) # TODO: not a good idea, fix
    except:
        raise AssertionError
    activity = nonlin_fn(activity, **nonlinearity_params)
    # sample from noise distribution
    # TODO: support noise distributions with more than one data-dependent parameter
    noise_fn = getattr(rng, noise_dist)
    data = noise_fn(activity, **noise_params).astype(float)
    # compile into dictionary for now
    # TODO: decide how to package data
    data_dict = {
        'mean': activity,
        'observation': data
    }
    return data_dict


def depasquale(*args, **kwargs):
    raise NotImplementedError

# other things to think about: 
# - calcium imaging (needs spike times, then convolution + noise)
# - LFP ?
# - unsorted spiking ?
    
