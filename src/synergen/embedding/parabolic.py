import numpy as np
from scipy.stats import special_ortho_group
from typing import Optional, Union

from .base import Embedding


class Parabolic2(Embedding):
    """Embed 2-d coordinates on a paraboloid"""

    def __init__(
        self,
        center: Union[list[float], np.ndarray] = [0.0, 0.0],
        scale: Union[list[float], np.ndarray] = [1.0, 1.0],
        rotate: bool = False,
        offset: Union[list[float], np.ndarray] = [0.0, 0.0, 0.0],
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Initialize parabolic embedding

        Parameters
        ----------
        center : array-like, default: [0., 0.]
            Per-dimension value to subtract to center the input data before coordinate transformation
        scale : array-like, default: [1., 1.]
            Per-dimension scaling to apply to the input data before coordinate transformation
        rotate : bool, default: False
            Whether to apply a random 2-d rotation to the input data before coordinate transformation
        offset : array-like, default: [0., 0., 0.]
            Offset to apply to the torus after the coordinate transformation
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        super().__init__(seed=seed)
        self.scale = np.asarray(scale)
        self.center = np.asarray(center)
        self.offset = np.asarray(offset)
        self.rotation = (
            special_ortho_group.rvs(dim=2, random_state=seed) if rotate else None
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert (
            X.shape[-1] == 2
        ), f"Size of last dimension of `points` must be 2, got {X.shape[-1]}"
        if self.rotation is not None:
            X = X @ self.rotation
        expand = lambda arr: np.expand_dims(arr, axis=tuple(range(X.ndim - 1)))
        z = np.sum(np.square(expand(self.scale) * (X - expand(self.center))), axis=-1)
        embedded = np.concatenate([X, z[..., None]], axis=-1)
        return embedded

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            dict(
                scale=self.scale,
                center=self.center,
            )
        )
        if self.rotation is not None:
            params.update(dict(rotation=self.rotation))
        return params
