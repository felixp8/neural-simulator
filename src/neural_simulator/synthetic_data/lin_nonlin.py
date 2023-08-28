import numpy as np
from typing import Optional, Union, Literal

from .base import DataSampler


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
    arr: np.ndarray,
    vscale: Union[float, np.ndarray] = 1., # float or (D,)
    hscale: Union[float, np.ndarray] = 1., # float or (D,)
    offset: Union[float, np.ndarray] = 0., # float or (D,)
):
    return vscale / (1. + np.exp(-hscale * arr + offset))

NONLINEARITIES = {
    'exp': exp,
    'relu': relu,
    'softplus': softplus,
    'sigmoid': sigmoid
}


class LinearNonlinear(DataSampler):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proj_weight_dist: Literal["normal", "uniform"] = "uniform",
        proj_weight_params: dict = {},
        nonlinearity: Literal["relu", "exp", "softplus", "sigmoid"] = "exp",
        nonlinearity_params: dict = {},
        noise_dist: Literal["poisson", "normal"] = "poisson",
        noise_params: dict = {},
        sort_dims: bool = False,
        standardize: bool = False,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        proj_weights = getattr(self.rng, proj_weight_dist)(size=(input_dim, output_dim), **proj_weight_params)
        if sort_dims:
            proj_weights = proj_weights[:, np.argsort(proj_weights[0])]
        self.proj_weights = proj_weights
        self.nonlinearity = nonlinearity
        self.nonlinearity_params = nonlinearity_params
        self.noise_dist = noise_dist
        self.noise_params = noise_params
        self.sort_dims = sort_dims
        self.standardize = standardize
    
    def sample(self, trajectories: np.ndarray):
        assert trajectories.shape[-1] == self.proj_weights.shape[0]
        activity = trajectories @ self.proj_weights
        if self.standardize:
            axes = tuple(range(trajectories.ndim - 1))
            orig_mean = np.mean(activity, axis=axes, keepdims=True)
            orig_std = np.std(activity, axis=axes, keepdims=True)
            activity = (activity - orig_mean) / orig_std
        nonlinearity_fn = NONLINEARITIES.get(self.nonlinearity)
        activity = nonlinearity_fn(activity, **self.nonlinearity_params)
        noise_fn = getattr(self.rng, self.noise_dist)
        data = noise_fn(activity, **self.noise_params).astype(float)
        data_dict = {
            'means': activity,
            'observations': data,
        }
        return data_dict


class LinearNonlinearPoisson(LinearNonlinear):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proj_weight_dist: Literal["normal", "uniform"] = "uniform",
        proj_weight_params: dict = {},
        nonlinearity: Literal["relu", "exp", "softplus", "sigmoid"] = "exp",
        nonlinearity_params: dict = {},
        sort_dims: bool = False,
        standardize: bool = False,
        clip_val: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            proj_weight_dist=proj_weight_dist,
            proj_weight_params=proj_weight_params,
            nonlinearity=nonlinearity,
            nonlinearity_params=nonlinearity_params,
            noise_dist='poisson',
            sort_dims=sort_dims,
            standardize=standardize,
            seed=seed,
        )
        self.clip_val = clip_val
    
    def sample(self, trajectories: np.ndarray):
        data_dict = super().sample(trajectories=trajectories)
        data_dict['rates'] = data_dict.pop('means')
        data_dict['spikes'] = np.clip(data_dict.pop('observations'), a_min=None, a_max=self.clip_val)
        return data_dict