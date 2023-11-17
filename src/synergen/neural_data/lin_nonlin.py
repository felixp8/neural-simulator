import numpy as np
from typing import Optional, Union, Literal

from .base import DataSampler


def exp(
    arr: np.ndarray,  # (..., D)
    vscale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    hscale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    offset: Union[float, np.ndarray] = 0.0,  # float or (D,)
):
    expand = lambda x: (
        np.expand_dims(x, axis=tuple(range(arr.ndim - 1)))
        if isinstance(x, np.ndarray)
        else x
    )
    return expand(vscale) * np.exp(arr * expand(hscale) + expand(offset))


def relu(
    arr: np.ndarray,
    scale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    offset: Union[float, np.ndarray] = 0.0,  # float or (D,)
):
    expand = lambda x: (
        np.expand_dims(x, axis=tuple(range(arr.ndim - 1)))
        if isinstance(x, np.ndarray)
        else x
    )
    return expand(scale) * np.clip(arr + expand(offset), amin=0.0)


def softplus(
    arr: np.ndarray,
    scale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    beta: Union[float, np.ndarray] = 1.0,  # float or (D,)
    offset: Union[float, np.ndarray] = 0.0,  # float or (D,)
    # threshold: float = 20., # used in pytorch
):
    expand = lambda x: (
        np.expand_dims(x, axis=tuple(range(arr.ndim - 1)))
        if isinstance(x, np.ndarray)
        else x
    )
    return (
        expand(scale)
        / expand(beta)
        * np.log(1 + np.exp(expand(beta) * arr + expand(offset)))
    )


def sigmoid(
    arr: np.ndarray,
    vscale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    hscale: Union[float, np.ndarray] = 1.0,  # float or (D,)
    offset: Union[float, np.ndarray] = 0.0,  # float or (D,)
):
    expand = lambda x: (
        np.expand_dims(x, axis=tuple(range(arr.ndim - 1)))
        if isinstance(x, np.ndarray)
        else x
    )
    return expand(vscale) / (1.0 + np.exp(-expand(hscale) * arr + expand(offset)))


NONLINEARITIES = {"exp": exp, "relu": relu, "softplus": softplus, "sigmoid": sigmoid}


class LinearNonlinear(DataSampler):
    """General linear-nonlinear data sampler with choice of noise distribution.
    Applies a linear transformation followed by an element-wise non-linear
    function before adding noise"""

    def __init__(
        self,
        output_dim: int,
        input_dim: Optional[int] = None,
        proj_weights: Optional[np.ndarray] = None,
        proj_weight_dist: Literal["normal", "uniform", "eye"] = "uniform",
        proj_weight_params: dict = {},
        normalize_proj: bool = False,
        nonlinearity: Literal["relu", "exp", "softplus", "sigmoid"] = "exp",
        nonlinearity_params: dict = {},
        obs_noise_dist: Literal["poisson", "normal"] = "poisson",
        obs_noise_params: dict = {},
        mean_center: Optional[Literal["channel", "all"]] = None,
        rescale_variance: Optional[Literal["channel", "all"]] = None,
        target_variance: float = 1.0,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Initialize LinearNonlinear

        Parameters
        ----------
        output_dim : int
            Number of output channels to produce
        input_dim : int, optional
            Number of input channels to expect. If not provided,
            `input_dim` is set dynamically on the first call to
            `sample` based on the input trajectories
        proj_weights : np.ndarray, optional
            Weights to use for readout projection weights, can be provided
            if they are not to be sampled randomly.
        proj_weight_dist : {"normal", "uniform", "eye"}, default: "uniform"
            Noise distribution to use for sampling readout projection weights.
            "eye" simply initializes the projection weights as a (likely truncated)
            identity matrix. "eye" is not recommended if `output_dim > input_dim`,
            as there will then be output channels with all zero projection weights
        proj_weight_params : dict, default: {}
            Any kwargs for sampling projection weights with `proj_weight_dist`. For
            "normal" and "uniform" distributions, these should be kwargs for
            `np.random.Generator.normal` and `np.random.Generator.uniform`, respectively.
            For "eye", the only supported kwarg is `shuffle`, which shuffles rows of
            the projection weights
        normalize_proj : bool, default: False
            Whether to normalize projection weights to sum to 1 per-output channel
        nonlinearity : {"relu", "exp", "softplus", "sigmoid"}, default: "exp"
            Nonlinearity to apply to the linearly transformed trajectories
        nonlinearity_params : dict, default: {}
            Any kwargs to provide to the nonlinearity function, like offsets
        obs_noise_dist : {"poisson", "normal"}, default: "poisson"
            Observation noise distribution for generating data
        obs_noise_params : dict, default: {}
            Any kwargs for the observation noise distribution. Should
            be kwargs supported by `np.random.Generator.poisson` or
            `np.random.Generator.normal`
        mean_center : {"channel", "all"}, optional
            If provided, the data will be mean-centered before
            applying the non-linearity. `mean_center="channel"`
            mean-centers each channel separately. `mean_center="all"`
            mean-centers the data by its global mean, preserving
            differences between channel means
        rescale_variance : {"channel", "all"}, optional
            If provided, the data will be rescaled to a target variance
            before applying the non-linearity.
            `rescale_variance="channel"` does this for each channel
            separately. `rescale_variance="all"` rescales by the
            global variance, preserving differences between channel
            variances
        target_variance : float, default: 1
            The target variance for the rescaling
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        super().__init__(seed=seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj_weights = proj_weights
        self.proj_weight_dist = proj_weight_dist
        self.proj_weight_params = proj_weight_params
        self.normalize_proj = normalize_proj
        self.nonlinearity = nonlinearity
        self.nonlinearity_params = nonlinearity_params
        self.obs_noise_dist = obs_noise_dist
        self.obs_noise_params = obs_noise_params
        assert mean_center in [None, "channel", "all"]
        assert rescale_variance in [None, "channel", "all"]
        self.mean_center = mean_center
        self.rescale_variance = rescale_variance
        self.target_variance = target_variance
        self.orig_mean = None
        self.orig_std = None
        self.std_mean = None

    def initialize_projection(self) -> None:
        """Initialize projection weight matrix"""
        assert (
            self.input_dim is not None
        ), "`self.input_dim` must be configured before initializing projection"
        if self.proj_weight_dist == "eye":
            proj_weights = np.eye(self.input_dim, self.output_dim)
            if self.proj_weight_params.get("shuffle", False):
                self.rng.shuffle(proj_weights, axis=0)
        else:
            proj_weights = getattr(self.rng, self.proj_weight_dist)(
                size=(self.input_dim, self.output_dim), **self.proj_weight_params
            )
        if self.normalize_proj:
            proj_weights /= np.linalg.norm(proj_weights, axis=0, keepdims=True) + 1e-5
        self.proj_weights = proj_weights

    def sample(self, trajectories: np.ndarray) -> dict:
        """Samples neural data given latent trajectories

        Parameters
        ----------
        trajectories : np.ndarray
            A batch x time x latent_dim array of latent
            trajectories

        Returns
        -------
        dict
            A dictionary with two entries: "means", which
            contains the pre-observation-noise data;
            and "observations", which contains the noised
            observation data. Both data arrays have
            shape batch x time x output_dim
        """
        if self.proj_weights is None:
            self.input_dim = trajectories.shape[-1]
            self.initialize_projection()
        assert trajectories.shape[-1] == self.proj_weights.shape[0]
        activity = trajectories @ self.proj_weights
        if self.mean_center is not None:
            if self.orig_mean is None:
                axes = tuple(
                    range(trajectories.ndim - int(self.mean_center == "channel"))
                )
                self.orig_mean = np.mean(activity, axis=axes, keepdims=True)
            activity = activity - self.orig_mean
        if self.rescale_variance is not None:
            if self.orig_std is None:
                axes = tuple(
                    range(trajectories.ndim - int(self.rescale_variance == "channel"))
                )
                self.std_mean = np.mean(activity, axis=axes, keepdims=True)
                self.orig_std = np.std(activity, axis=axes, keepdims=True)
            activity = self.std_mean + (activity - self.std_mean) / (
                self.orig_std / self.target_variance + 1e-5
            )
        nonlinearity_fn = NONLINEARITIES.get(self.nonlinearity)
        activity = nonlinearity_fn(activity, **self.nonlinearity_params)
        noise_fn = getattr(self.rng, self.obs_noise_dist)
        data = noise_fn(activity, **self.obs_noise_params).astype(float)
        data_dict = {
            "means": activity,
            "observations": data,
        }
        return data_dict

    def get_params(self):
        params = super().get_params()
        params.update(
            dict(
                nonlinearity=self.nonlinearity,
                obs_noise_dist=self.obs_noise_dist,
                output_dim=self.output_dim,
                input_dim=self.input_dim,
                proj_weight_dist=self.proj_weight_dist,
                proj_weight_params=self.proj_weight_params,
                normalize_proj=self.normalize_proj,
            )
        )
        if self.proj_weights is not None:
            params.update(dict(proj_weights=self.proj_weights))
        if self.orig_mean is not None:
            params.update(dict(orig_mean=self.orig_mean))
        if self.orig_std is not None:
            params.update(dict(orig_std=self.orig_std))
        if self.std_mean is not None:
            params.update(dict(std_mean=self.std_mean))
        return params

    def set_params(self, params: dict) -> None:
        super().set_params(params)
        # can honestly just unpack params into __dict__
        self.nonlinearity = params["nonlinearity"]
        self.obs_noise_dist = params["obs_noise_dist"]
        self.output_dim = params["output_dim"]
        self.input_dim = params["input_dim"]
        self.proj_weight_dist = params["proj_weight_dist"]
        self.proj_weight_params = params["proj_weight_params"]
        self.normalize_proj = params["normalize_proj"]
        self.proj_weights = params.get("proj_weights")
        self.orig_mean = params.get("orig_mean")
        self.orig_std = params.get("orig_std")
        self.std_mean = params.get("std_mean")
        return


class LinearNonlinearPoisson(LinearNonlinear):
    """Convenience class for Poisson spiking with a linear-nonlinear
    encoding model"""

    def __init__(
        self,
        output_dim: int,
        input_dim: Optional[int] = None,
        proj_weight_dist: Literal["normal", "uniform", "eye"] = "uniform",
        proj_weight_params: dict = {},
        normalize_proj: bool = False,
        nonlinearity: Literal["relu", "exp", "softplus", "sigmoid"] = "exp",
        nonlinearity_params: dict = {},
        mean_center: Optional[Literal["channel", "all"]] = None,
        rescale_variance: Optional[Literal["channel", "all"]] = None,
        target_variance: float = 1.0,
        clip_val: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        """Initialize LinearNonlinearPoisson

        Parameters
        ----------
        output_dim : int
            Number of output channels to produce
        input_dim : int, optional
            Number of input channels to expect. If not provided,
            `input_dim` is set dynamically on the first call to
            `sample` based on the input trajectories
        proj_weight_dist : {"normal", "uniform", "eye"}, default: "uniform"
            Noise distribution to use for sampling readout projection weights.
            "eye" simply initializes the projection weights as a (likely truncated)
            identity matrix. "eye" is not recommended if `output_dim > input_dim`,
            as there will then be output channels with all zero projection weights
        proj_weight_params : dict, default: {}
            Any kwargs for sampling projection weights with `proj_weight_dist`. For
            "normal" and "uniform" distributions, these should be kwargs for
            `np.random.Generator.normal` and `np.random.Generator.uniform`, respectively.
            For "eye", the only supported kwarg is `shuffle`, which shuffles rows of
            the projection weights
        normalize_proj : bool, default: False
            Whether to normalize projection weights to sum to 1 per-output channel
        nonlinearity : {"relu", "exp", "softplus", "sigmoid"}, default: "exp"
            Nonlinearity to apply to the linearly transformed trajectories
        nonlinearity_params : dict, default: {}
            Any kwargs to provide to the nonlinearity function, like offsets
        mean_center : {"channel", "all"}, optional
            If provided, the data will be mean-centered before
            applying the non-linearity. `mean_center="channel"`
            mean-centers each channel separately. `mean_center="all"`
            mean-centers the data by its global mean, preserving
            differences between channel means
        rescale_variance : {"channel", "all"}, optional
            If provided, the data will be rescaled to a target variance
            before applying the non-linearity.
            `rescale_variance="channel"` does this for each channel
            separately. `rescale_variance="all"` rescales by the
            global variance, preserving differences between channel
            variances
        target_variance : float, default: 1
            The target variance for the rescaling
        clip_val : int, optional
            If provided, the maximum allowable spike count per bin. Counts
            exceeding `clip_val` will be clipped to that value
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            proj_weight_dist=proj_weight_dist,
            proj_weight_params=proj_weight_params,
            normalize_proj=normalize_proj,
            nonlinearity=nonlinearity,
            nonlinearity_params=nonlinearity_params,
            obs_noise_dist="poisson",
            mean_center=mean_center,
            rescale_variance=rescale_variance,
            target_variance=target_variance,
            seed=seed,
        )
        self.clip_val = clip_val

    def sample(self, trajectories: np.ndarray):
        """Samples spiking data given latent trajectories

        Parameters
        ----------
        trajectories : np.ndarray
            A batch x time x latent_dim array of latent
            trajectories

        Returns
        -------
        dict
            A dictionary with two entries: "rates", which
            contains the pre-observation-noise firing rate data;
            and "spikes", which contains the Poisson-sampled
            observed spiking data. Both data arrays have
            shape batch x time x output_dim
        """
        data_dict = super().sample(trajectories=trajectories)
        data_dict["rates"] = data_dict.pop("means")
        spikes = data_dict.pop("observations")
        if self.clip_val is not None:
            spikes = np.clip(spikes, a_min=None, a_max=self.clip_val)
        data_dict["spikes"] = spikes
        return data_dict
