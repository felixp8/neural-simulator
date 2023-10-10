import numpy as np
from typing import Optional, Union, Literal

from .base import DataSampler


def exp(
    arr: np.ndarray,  # (..., D)
    vscale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    hscale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    offset: Union[float, np.ndarray] = 0.0,  # float or (D,)
):
    # TODO: implicit broadcasting may run into issues if any other arr dims == D
    return vscale * np.exp(arr * hscale + offset)


def relu(
    arr: np.ndarray,
    scale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    offset: Union[float, np.ndarray] = 0.0,  # float or (D,)
):
    return scale * np.clip(arr + offset, amin=0.0)


def softplus(
    arr: np.ndarray,
    scale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    beta: Union[float, np.ndarray] = 1.0,  # float or (D,)
    offset: Union[float, np.ndarray] = 0.0,  # float or (D,)
    # threshold: float = 20., # used in pytorch
):
    return scale / beta * np.log(1 + np.exp(beta * arr + offset))


def sigmoid(
    arr: np.ndarray,
    vscale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    hscale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    offset: Union[float, np.ndarray] = 0.0,  # float or (D,)
):
    return vscale / (1.0 + np.exp(-hscale * arr + offset))


NONLINEARITIES = {"exp": exp, "relu": relu, "softplus": softplus, "sigmoid": sigmoid}


class LinearNonlinear(DataSampler):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proj_weight_dist: Literal["normal", "uniform", "eye"] = "uniform",
        proj_weight_params: dict = {},
        normalize_proj: bool = False,
        nonlinearity: Literal["relu", "exp", "softplus", "sigmoid"] = "exp",
        nonlinearity_params: dict = {},
        noise_dist: Literal["poisson", "normal"] = "poisson",
        noise_params: dict = {},
        sort_dims: bool = False,
        mean_center: Optional[Literal["neuron", "all"]] = None,
        rescale_variance: Optional[Literal["neuron", "all"]] = None,
        target_variance: float = 1.0,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        if proj_weight_dist == "eye":
            proj_weights = np.eye(input_dim, output_dim)
            if proj_weight_params.get("shuffle", False):
                self.rng.shuffle(proj_weights, axis=0)
        else:
            proj_weights = getattr(self.rng, proj_weight_dist)(
                size=(input_dim, output_dim), **proj_weight_params
            )
        if normalize_proj:
            proj_weights /= np.linalg.norm(proj_weights, axis=0, keepdims=True) + 1e-5
        if sort_dims:
            proj_weights = proj_weights[:, np.argsort(proj_weights[0])]
        self.proj_weights = proj_weights
        self.nonlinearity = nonlinearity
        self.nonlinearity_params = nonlinearity_params
        self.noise_dist = noise_dist
        self.noise_params = noise_params
        self.sort_dims = sort_dims
        assert mean_center in [None, "neuron", "all"]
        assert rescale_variance in [None, "neuron", "all"]
        self.mean_center = mean_center
        self.rescale_variance = rescale_variance
        self.target_variance = target_variance
        self.orig_mean = None
        self.orig_std = None
        self.std_mean = None

    def sample(self, trajectories: np.ndarray):
        assert trajectories.shape[-1] == self.proj_weights.shape[0]
        activity = trajectories @ self.proj_weights
        if self.mean_center is not None:
            if self.orig_mean is None:
                axes = tuple(
                    range(trajectories.ndim - int(self.mean_center == "neuron"))
                )
                self.orig_mean = np.mean(activity, axis=axes, keepdims=True)
            activity = activity - self.orig_mean
        if self.rescale_variance is not None:
            if self.orig_std is None:
                axes = tuple(
                    range(trajectories.ndim - int(self.rescale_variance == "neuron"))
                )
                self.std_mean = np.mean(activity, axis=axes, keepdims=True)
                self.orig_std = np.std(activity, axis=axes, keepdims=True)
            activity = self.std_mean + (activity - self.std_mean) / (
                self.orig_std / self.target_variance + 1e-5
            )
        nonlinearity_fn = NONLINEARITIES.get(self.nonlinearity)
        activity = nonlinearity_fn(activity, **self.nonlinearity_params)
        noise_fn = getattr(self.rng, self.noise_dist)
        data = noise_fn(activity, **self.noise_params).astype(float)
        data_dict = {
            "means": activity,
            "observations": data,
        }
        return data_dict

    def get_params(self):
        params = super().get_params()
        params.update(
            dict(
                proj_weights=self.proj_weights,
                nonlinearity=self.nonlinearity,
                noise_dist=self.noise_dist,
            )
        )
        if self.orig_mean is not None:
            params.update(dict(orig_mean=self.orig_mean))
        if self.orig_std is not None:
            params.update(dict(orig_std=self.orig_std))
        if self.std_mean is not None:
            params.update(dict(std_mean=self.std_mean))
        return params


class LinearNonlinearPoisson(LinearNonlinear):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proj_weight_dist: Literal["normal", "uniform"] = "uniform",
        proj_weight_params: dict = {},
        normalize_proj: bool = False,
        nonlinearity: Literal["relu", "exp", "softplus", "sigmoid"] = "exp",
        nonlinearity_params: dict = {},
        sort_dims: bool = False,
        mean_center: Optional[Literal["neuron", "all"]] = None,
        rescale_variance: Optional[Literal["neuron", "all"]] = None,
        target_variance: float = 1.0,
        clip_val: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            proj_weight_dist=proj_weight_dist,
            proj_weight_params=proj_weight_params,
            normalize_proj=normalize_proj,
            nonlinearity=nonlinearity,
            nonlinearity_params=nonlinearity_params,
            noise_dist="poisson",
            sort_dims=sort_dims,
            mean_center=mean_center,
            rescale_variance=rescale_variance,
            target_variance=target_variance,
            seed=seed,
        )
        self.clip_val = clip_val

    def sample(self, trajectories: np.ndarray):
        data_dict = super().sample(trajectories=trajectories)
        data_dict["rates"] = data_dict.pop("means")
        spikes = data_dict.pop("observations")
        if self.clip_val is not None:
            spikes = np.clip(spikes, a_min=None, a_max=self.clip_val)
        data_dict["spikes"] = spikes
        return data_dict
