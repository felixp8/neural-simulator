import numpy as np
from scipy.stats import special_ortho_group
from typing import Optional, Union

from .base import Embedding


class Parabolic3D(Embedding):
    def __init__(
        self, 
        center: Union[list[float], np.ndarray] = [0., 0.],
        scale: Union[list[float], np.ndarray] = [1., 1.],
        rotate: bool = False,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        self.scale = np.array(scale)
        self.center = np.array(center)
        self.rotation = special_ortho_group.rvs(dim=2, random_state=seed) if rotate else None
    
    def transform(self, X: np.ndarray):
        assert X.shape[-1] == 2, f"Size of last dimension of `points` must be 2, got {X.shape[-1]}"
        if self.rotation is not None:
            X = X @ self.rotation
        expand = lambda arr: np.expand_dims(arr, axis=tuple(range(X.ndim - 1)))
        z = np.sum(np.square(expand(self.scale) * (X - expand(self.center))), axis=-1)
        embedded = np.concatenate([X, z[..., None]], axis=-1)
        return embedded