import numpy as np
import itertools
from typing import Union, Optional, Any


ALL_DISTRIBUTIONS = [
    "uniform",
    "loguniform",
    "normal",
    "categorical",
]


class Distribution:
    def __init__(self, seed: Optional[Union[int, np.random.Generator]] = None):
        super().__init__()
        self.seed(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

    def sample(self, n: int = 1):
        raise NotImplementedError


class Uniform(Distribution):
    def __init__(
        self,
        low: float,
        high: float,
        shape: tuple = (),
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self, n: int = 1):
        return self.rng.uniform(low=self.low, high=self.high, size=(n,) + self.shape)


class LogUniform(Distribution):
    def __init__(
        self,
        low: float,
        high: float,
        shape: tuple = (),
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        assert (low > 0) and (
            high > 0
        ), "Sampling bounds must be greater than 0 for log-uniform sampling"
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self, n: int = 1):
        return np.exp(
            self.rng.uniform(
                low=np.log(self.low), high=np.log(self.high), size=(n,) + self.shape
            )
        )


class Normal(Distribution):
    def __init__(
        self,
        mean: float,
        std: float,
        shape: tuple = (),
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        self.mean = mean
        self.std = std
        self.shape = shape

    def sample(self, n: int = 1):
        return self.rng.normal(loc=self.mean, scale=self.std, size=(n,) + self.shape)


class Categorical(Distribution):
    def __init__(
        self,
        categories: Union[list[Any], np.ndarray],
        probs: Optional[Union[list[float], np.ndarray]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        self.categories = categories
        self.probs = probs

    def sample(self, n: int = 1):
        return self.rng.choice(a=self.categories, p=self.probs, size=n)


class SampleSpace:
    def __init__(
        self,
        distributions: dict[str, Union[dict, Distribution]] = {},
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__()
        self.seed(seed)
        self._distributions = {
            key: val
            if isinstance(val, Distribution)
            else self._build_distribution(**val)
            for key, val in distributions.items()
        }

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

    def _build_distribution(self, dist_name: str, dist_params: dict = {}):
        assert dist_name.lower() in ALL_DISTRIBUTIONS
        if dist_params.pop("seed", None) is not None:
            # print("warning - overriding seed")
            pass
        if dist_name.lower() == "uniform":
            return Uniform(seed=self.rng, **dist_params)
        if dist_name.lower() == "loguniform":
            return LogUniform(seed=self.rng, **dist_params)
        if dist_name.lower() == "normal":
            return Normal(seed=self.rng, **dist_params)
        if dist_name.lower() == "categorical":
            return Categorical(seed=self.rng, **dist_params)

    def add_distribution(self, name: str, distribution: Union[dict, Distribution]):
        assert name not in self._distributions, f"{name} already exists in sample space"
        if isinstance(distribution, dict):
            distribution = self._build_distribution(distribution)
        self._distributions[name] = distribution

    def sample(self, n: int = 1, stratified: bool = False):
        sampled_dict = {
            key: val.sample(n)
            for key, val in self._distributions.items()
            if not (isinstance(val, Categorical) and stratified)
        }
        if stratified:  # only do stratified sampling over categorical fields
            categorical_keys = [
                key
                for key, val in self._distributions.items()
                if isinstance(val, Categorical)
            ]
            all_categories = [
                self._distributions[key].categories for key in categorical_keys
            ]
            all_combinations = np.array(list(itertools.product(*all_categories)))
            stratified_sample = np.concatenate(
                [
                    np.tile(all_combinations, (n // len(all_combinations), 1)),
                    all_combinations[
                        self.rng.choice(
                            a=len(all_combinations), size=(n % len(all_combinations))
                        )
                    ],
                ],
                axis=0,
            )
            stratified_sample = stratified_sample[self.rng.permutation(n)]
            for i, key in enumerate(categorical_keys):
                sampled_dict[key] = stratified_sample[:, i]
        return sampled_dict
