import numpy as np
import pandas as pd
import gymnasium as gym
import neurogym as ngym
import ivy
import itertools

from .base import System, UncoupledSystem
from .gym import sample_trial_params


class TrialEnvSystem(UncoupledSystem):
    """Dynamical system receiving input from neurogym.TrialEnv"""
    
    def __init__(
        self, 
        system,
        env: ngym.TrialEnv,
        n_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_dim=n_dim, n_input_dim=env.observation_space.n, seed=seed)
        self.system = system
        self.env = env
        self.env.seed(seed)

    def sample_trials( # can be moved to like a shared gym Mixin class
        self,
        n_trials: int,
        sample_space: dict = {},
        stratified: bool = False,
    ) -> list[dict]:
        if not sample_space:
            return [{}] * n_trials
        return sample_trial_params(sample_space=sample_space, n_trials=n_trials, stratified=stratified)

    def sample_inputs(
        self,
        trial_info: list[dict],
    ):
        # TODO: handle ragged trial lengths
        self.env.reset()
        inputs = []
        for trial_params in trial_info:
            _ = self.env.new_trial(**trial_params)
            inputs.append(self.env.ob) # TODO: doesn't include modifications by wrappers ugh
        return np.stack(inputs, axis=0)

    def simulate_system(
        self,
        ics: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # model forward pass
        raise NotImplementedError
