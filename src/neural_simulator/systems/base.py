import numpy as np
import pandas as pd
from typing import Union

class System:
    """Generic dynamical system base class"""
    
    def __init__(self):
        super().__init__()

    def sample_trajectories(self) -> tuple[np.ndarray, pd.DataFrame, Union[np.ndarray, None]]:
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


class AutonomousSystem(System):
    """Dynamical system that does not receive external inputs"""

    def __init__(self):
        super().__init__()

    def sample_trajectories(
        self, 
        ic_kwargs: dict = {}, 
        simulation_kwargs: dict = {}
    ) -> tuple[np.ndarray, pd.DataFrame]:
        ics = self.sample_ics(**ic_kwargs) # b x d
        trajectories = self.simulate_system(ics, **simulation_kwargs) # b x t x d
        trial_info = pd.DataFrame(ics, columns=[f'ic_dim{i}' for i in range(ics.shape[1])])
        return trajectories, trial_info, None

    def sample_ics(self, **kwargs):
        raise NotImplementedError

    def simulate_system(self, **kwargs):
        raise NotImplementedError


class NonAutonomousSystem(System):
    """Dynamical system that receives external inputs"""

    def __init__(self):
        super().__init__()

    def sample_trajectories(self):
        pass
        
