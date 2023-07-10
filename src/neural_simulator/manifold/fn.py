import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from typing import Optional


def parabolic3d(
    points: np.ndarray,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    x_center: float = 0.0,
    y_center: float = 0.0,
    rotate: bool = False,
    seed: Optional[int] = None,
):
    # embeds 2d coordinates in 3d on a parabolic surface
    # just an example for what kind of things might be in this file
    assert points.shape[-1] = 2
    embedded = np.concatenate([
        points,
        x_scale * np.square(points[..., 0] - x_center) + y_scale * np.square(points[..., 1] - y_center),
    ], axis=-1)
    if rotate:
        r = special_ortho_group.rvs(dim=3, seed=seed)
        embedded = embedded @ r
    return embedded
