import numpy as np
from typing import Optional, Union, Literal

from .base import Embedding


class Subsample(Embedding):
    def __init__(
        self,
        n_dim: int,
        subsample_method: Literal["random", "first", "last"] = "random",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        self.n_dim = n_dim
        self.subsample_method = subsample_method
        self.subsampled_dims = None
    
    def transform(self, X: np.ndarray, resample: bool = False):
        assert self.n_dim <= X.shape[-1], \
            f"Data dimensionality smaller than subsample size: {X.shape[-1]} < {self.n_dim}"
        if self.subsampled_dims is None or resample:
            if self.subsample_method == "first":
                self.subsampled_dims = np.arange(self.n_dim)
            elif self.subsample_method == "last":
                self.subsampled_dims = np.arange(X.shape[-1] - self.n_dim, X.shape[-1])
            else:
                self.subsampled_dims = self.rng.choice(X.shape[-1], size=self.n_dim, replace=False)
        assert self.subsampled_dims.max() < X.shape[-1], \
            f"Subsampled dimensions do not match data dimensionality"
        embedded = X[..., self.subsampled_dims]
        return embedded