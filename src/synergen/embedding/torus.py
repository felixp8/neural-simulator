import numpy as np
from scipy.stats import special_ortho_group
from typing import Optional, Union

from .base import Embedding


class Torus(Embedding):
    def __init__(
        self,
        major_radius: float = 2.,
        minor_radius: float = 1.,
        offset: Union[list[float], np.ndarray] = [0., 0., 0.],
        center: Union[list[float], np.ndarray] = [0., 0.],
        scale: Union[list[float], np.ndarray] = [1., 1.],
        rotate: bool = False,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.offset = np.array(offset)
        self.center = np.array(center)
        self.scale = np.array(scale)
        self.rotation = special_ortho_group.rvs(dim=2, random_state=seed) if rotate else None
    
    def transform(self, X: np.ndarray):
        assert X.shape[-1] == 2, f"Size of last dimension of `X` must be 2, got {X.shape[-1]}"
        if self.rotation is not None:
            X = X @ self.rotation
        expand = lambda arr: np.expand_dims(arr, axis=tuple(range(X.ndim - 1)))
        X = expand(self.scale) * (X - expand(self.center))
        if np.any(X < 0) or np.any(X >= (2*np.pi)):
            print("warning: X outside of domain, no longer injective mapping")
        x = (self.major_radius + self.minor_radius * np.cos(X[..., 0])) * np.cos(X[..., 1])
        y = (self.major_radius + self.minor_radius * np.cos(X[..., 0])) * np.sin(X[..., 1])
        z = self.minor_radius * np.sin(X[..., 0])
        embedded = np.stack([x, y, z], axis=-1)
        embedded += expand(self.offset)
        return embedded

        