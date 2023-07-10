import numpy as np
import pandas as pd
from typing import Union, Optional
from collections.abc import Callable


class System:
    """Generic dynamical system base class"""
    
    def __init__(self, n_dim: int, seed=None) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.rng = np.random.default_rng(seed)

    def seed(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_trajectories(self, *args, **kwargs) -> tuple[np.ndarray, pd.DataFrame, Union[np.ndarray, None]]:
        """Main class method to implement, samples trajectories from system.

        Returns
        -------
        trajectories : np.ndarray
            Array of sampled trajectories, with shape (B, T, D)
        trial_info : pd.DataFrame
            DataFrame containing information about each trial, with shape
            (B, C)
        inputs : np.ndarray or None
            Array of inputs to system at each timestep, with shape
            (B, T, I). If there are no inputs, `inputs = None`
        """
        raise NotImplementedError
        
    def sample_ics(
        self,
        ics: Optional[np.ndarray] = None,
        n_trials: Optional[int] = None,
        dist: Optional[Union[str, Callable]] = None,
        dist_params: dict = {},
    ) -> np.ndarray:
        if ics is not None:
            return ics
        assert n_trials is not None, "If `ics = None`, `n_trials` must be provided"
        assert dist is not None, "If `ics = None`, `dist` must be provided"
        if dist == "zeros":
            return np.zeros((n_trials, self.n_dim))
        elif hasattr(self.rng, dist):
            ics = getattr(self.rng, dist)(size=(n_trials, self.n_dim), **dist_params)
        elif callable(dist):
            ics = dist(**dist_params) # TODO: and pass size args?
        else:
            raise ValueError
        return ics


class AutonomousSystem(System):
    """Dynamical system that does not receive external inputs"""

    def __init__(self, n_dim, seed=None) -> None:
        super().__init__(n_dim=n_dim, seed=seed)

    def sample_trajectories(
        self, 
        ic_kwargs: dict = {}, 
        simulation_kwargs: dict = {}
    ) -> tuple[np.ndarray, pd.DataFrame, None]:
        ics = self.sample_ics(**ic_kwargs) # b x d
        trajectories = self.simulate_system(ics, **simulation_kwargs) # b x t x d
        trial_info = pd.DataFrame(ics, columns=[f'ic_dim{i}' for i in range(ics.shape[1])])
        return trajectories, trial_info, None

    def simulate_system(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class NonAutonomousSystem(System):
    """Dynamical system that receives external inputs"""

    def __init__(self, n_dim, n_input_dim, seed=None) -> None:
        super().__init__(n_dim=n_dim, seed=seed)
        self.n_input_dim = n_input_dim

    def sample_trajectories(self, *args, **kwargs) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        raise NotImplementedError


class CoupledSystem(NonAutonomousSystem):
    """Dynamical system that receives inputs from an environment and acts on the environment"""

    def __init__(self, n_dim, n_input_dim, seed=None) -> None:
        super().__init__(n_dim=n_dim, n_input_dim=n_input_dim, seed=seed)

    def sample_trajectories(
        self,
        ic_kwargs: dict = {},
        trial_kwargs: dict = {},
        simulation_kwargs: dict = {},
    ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        ics = self.sample_ics(**ic_kwargs) # b x d
        trial_info = self.sample_trials(**trial_kwargs)
        trajectories, inputs = self.simulate_system(ics, trial_info, **simulation_kwargs) # b x t x d
        for i in range(ics.shape[1]):
            trial_info[f'ic_dim{i}'] = ics[:, i]
        return trajectories, trial_info, inputs

    def sample_trials(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def simulate_system(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class UncoupledSystem(NonAutonomousSystem):
    """Dynamical system that receives inputs from an environment but doesn't act on the environment.
    Not really "uncoupled" - just not bidirectionally coupled, so maybe rename"""

    def __init__(self, n_dim, n_input_dim, seed=None) -> None:
        super().__init__(n_dim=n_dim, n_input_dim=n_input_dim, seed=seed)
    
    def sample_trajectories(
        self,
        ic_kwargs: dict = {},
        trial_kwargs: dict = {},
        input_kwargs: dict = {},
        simulation_kwargs: dict = {},
    ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        ics = self.sample_ics(**ic_kwargs) # b x d
        trial_info = self.sample_trials(**trial_kwargs)
        inputs = self.sample_inputs(trial_info, **input_kwargs)
        trajectories = self.simulate_system(ics, inputs, **simulation_kwargs) # b x t x d
        for i in range(ics.shape[1]):
            trial_info[f'ic_dim{i}'] = ics[:, i]
        return trajectories, trial_info, inputs

    def sample_trials(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def sample_inputs(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def simulate_system(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
