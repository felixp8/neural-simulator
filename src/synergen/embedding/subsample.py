import numpy as np
from typing import Optional, Union, Literal

from .base import Embedding


class Subsample(Embedding):
    """Subsample input coordinate dimensions"""

    def __init__(
        self,
        n_dim: int,
        subsample_method: Optional[Literal["random", "first", "last"]] = "random",
        subsampled_dims: Optional[np.ndarray] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Initialize subsample embedding

        Parameters
        ----------
        n_dim : int
            Number of dimensions to choose for subsampling
        subsample_method : {"random", "first", "last"}, default: "random"
            How to choose dimensions for subsampling. "random" chooses
            randomly with `np.random.Generator.choice`. "first" chooses
            the first `n_dim` dimensions. "last" chooses the last `n_dim`
            dimensions. If None, `subsampled_dims` must be directly provided
        subsampled_dims : np.ndarray, optional
            An array containing indices of the dimensions to keep. Only
            used if `subsample_method=None`
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        if subsample_method is None:
            assert (
                subsampled_dims is not None
            ), "One of `subsample_method` or `subsampled_dims` must be provided"
        else:
            assert subsample_method in ["random", "first", "last"]
            if subsampled_dims is not None:
                subsampled_dims = None  # print warning?
        super().__init__(seed=seed)
        self.n_dim = n_dim
        self.subsample_method = subsample_method
        self.subsampled_dims = subsampled_dims

    def transform(self, X: np.ndarray, resample: bool = False) -> np.ndarray:
        """Transforms data points in X

        Parameters
        ----------
        X : np.ndarray
            An array of shape (..., D), where D is
            the dimensionality of each data point
        resample : bool, default: False
            Whether to resample new dimensions for subsampling.
            New dimensions are always chosen on the first call
            to `transform` but subsequent calls to transform
            will use those same dimensions if `resample=False`

        Returns
        -------
        np.ndarray
            An array of shape (..., E), where E is
            the dimensionality of the transformed/embedded
            data points
        """
        assert (
            self.n_dim <= X.shape[-1]
        ), f"Data dimensionality smaller than subsample size: {X.shape[-1]} < {self.n_dim}"
        if self.subsampled_dims is None or resample:
            if self.subsample_method == "first":
                self.subsampled_dims = np.arange(self.n_dim)
            elif self.subsample_method == "last":
                self.subsampled_dims = np.arange(X.shape[-1] - self.n_dim, X.shape[-1])
            elif self.subsample_method == "random":
                self.subsampled_dims = self.rng.choice(
                    X.shape[-1], size=self.n_dim, replace=False
                )
            # else print warning?
        assert (
            self.subsampled_dims.max() < X.shape[-1]
        ), f"Subsampled dimensions do not match data dimensionality"
        embedded = X[..., self.subsampled_dims]
        return embedded

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(dict(n_dim=self.n_dim, subsample_method=self.subsample_method))
        if self.subsampled_dims is not None:
            params.update(dict(subsampled_dims=self.subsampled_dims))
        return params

    def set_params(self, params: dict) -> None:
        super().set_params(params)
        self.n_dim = params["n_dim"]
        self.subsample_method = params["subsample_method"]
        self.subsampled_dims = params.get("subsampled_dims")
        return
