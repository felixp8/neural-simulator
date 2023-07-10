import numpy as np
import dysts
from typing import Optional

from .base import System, AutonomousSystem

class DystsSystem(AutonomousSystem):
    """Subclass for dynamical systems implemented in dysts"""

    def __init__(
        self, 
        name: str, 
        params: dict = {}, 
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.system = getattr(dysts.flows, name)(**params)
        self.system.random_state = seed
        self.rng = np.random.default_rng(seed=None)
        self.n_dims = self.system.embedding_dimension

    def seed(self, seed: Optional[int] = Non):
        self.system.random_state = seed
        self.rng = np.random.default_rng(seed=None)

    def sample_ics(
        self,
        ics: Optional[np.ndarray] = None,
        n_trials: Optional[int] = None,
        dist: Optional[str] = None,
        dist_params: dict = {},
    ):
        if ics is not None:
            return ics
        assert n_trials is not None, "If `ics = None`, `n_trials` must be provided"
        assert dist is not None, "If `ics = None`, `dist` must be provided"
        ics = getattr(self.rng, dist)(size=(n_trials, self.n_dims), **dist_params)
        return ics

    def simulate_system(
        self,
        ics: np.ndarray,
        trial_len: int,
        burn_in: int = 0,
        method: str = "Radau",
        resample: bool = True,
        pts_per_period: int = 100,
        standardize: bool = False,
        postprocess: bool = True,
        noise: float = 0.0,
    ):
        trajectories = []
        for ic in ics:
            self.system.ic = ic
            traj = self.system.make_trajectory(
                n=(trial_len + burn_in),
                method=method,
                resample=resample,
                pts_per_period=pts_per_period,
                standardize=standardize,
                postprocess=postprocess,
                noise=noise,
            )[burn_in:]
            trajectories.append(traj)
        trajectories = np.stack(trajectories, axis=0)
        return trajectories

    
