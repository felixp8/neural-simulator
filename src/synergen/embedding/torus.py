import numpy as np
from scipy.stats import special_ortho_group
from typing import Optional, Union

from .base import Embedding


class Torus1(Embedding):
    """Embed 1-d coordinates on a 1-Torus (aka a circle)"""

    def __init__(
        self,
        radius: float = 1.0,
        offset: Union[list[float], np.ndarray] = [0.0, 0.0],
        center: float = 0.0,
        scale: float = 1.0,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """initialize 1-Torus embedding

        Parameters
        ----------
        radius : float, default: 1
            Radius of the circle
        center : float, default: 0
            Value to subtract to center the input data before coordinate transformation
        scale : float, default: 1
            Scaling to apply to the input data before coordinate transformation
        offset : array-like, default: [0., 0.]
            Offset to apply to the circle after the coordinate transformation
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        super().__init__(seed)
        self.radius = radius
        self.offset = np.asarray(offset)
        self.center = center
        self.scale = scale

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert (
            X.shape[-1] == 1
        ), f"Size of last dimension of `X` must be 1, got {X.shape[-1]}"
        expand = lambda arr: np.expand_dims(arr, axis=tuple(range(X.ndim - 1)))
        X = self.scale * (X - self.center)
        if np.any(X < 0) or np.any(X >= (2 * np.pi)):
            print("warning: X outside of domain [0,2*pi), no longer injective mapping")
        x = self.radius * np.cos(X)
        y = self.radius * np.sin(X)
        embedded = np.stack([x, y], axis=-1)
        embedded += expand(self.offset)
        return embedded

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            dict(
                radios=self.radius,
                offset=self.offset,
                center=self.center,
                scale=self.scale,
            )
        )
        return params


class Torus2(Embedding):
    """Embed 2-d coordinates on a 2-Torus (aka a donut)"""

    def __init__(
        self,
        major_radius: float = 2.0,
        minor_radius: float = 1.0,
        center: Union[list[float], np.ndarray] = [0.0, 0.0],
        scale: Union[list[float], np.ndarray] = [1.0, 1.0],
        rotate: bool = False,
        offset: Union[list[float], np.ndarray] = [0.0, 0.0, 0.0],
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Initialize 2-Torus embedding

        Parameters
        ----------
        major_radius : float, default: 2
            Major radius of the torus, i.e. the distance from the center of the torus
            itself to the center of the tube
        minor_radius : float, default: 1
            Minor radius of the torus, i.e. the radius of the tube itself
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
        super().__init__(seed)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.offset = np.array(offset)
        self.center = np.array(center)
        self.scale = np.array(scale)
        self.rotation = (
            special_ortho_group.rvs(dim=2, random_state=seed) if rotate else None
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert (
            X.shape[-1] == 2
        ), f"Size of last dimension of `X` must be 2, got {X.shape[-1]}"
        if self.rotation is not None:
            X = X @ self.rotation
        expand = lambda arr: np.expand_dims(arr, axis=tuple(range(X.ndim - 1)))
        X = expand(self.scale) * (X - expand(self.center))
        if np.any(X < 0) or np.any(X >= (2 * np.pi)):
            print("warning: X outside of domain [0,2*pi), no longer injective mapping")
        x = (self.major_radius + self.minor_radius * np.cos(X[..., 0])) * np.cos(
            X[..., 1]
        )
        y = (self.major_radius + self.minor_radius * np.cos(X[..., 0])) * np.sin(
            X[..., 1]
        )
        z = self.minor_radius * np.sin(X[..., 0])
        embedded = np.stack([x, y, z], axis=-1)
        embedded += expand(self.offset)
        return embedded

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            dict(
                major_radios=self.major_radius,
                minor_radius=self.minor_radius,
                offset=self.offset,
                center=self.center,
                scale=self.scale,
            )
        )
        if self.rotation is not None:
            params.update(dict(rotation=self.rotation))
        return params
