import numpy as np
import pandas as pd
import itertools
from typing import Optional, Union, Callable, Any


def sample_dict( # TODO: make this a separate class or put it in utils or something
    sample_space: dict,
    overrides: dict = {},
    rng: np.random.Generator = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    sampled_dict = {}
    for key, val in sample_space:
        if key in overrides:
            sampled_dict[key] = overrides[key]
        else:
            if val['dist'] == 'discrete':
                sampled_dict[key] = rng.choice(**val['dist_params'])
            else:
                sampled_dict[key] = getattr(rng, val['dist'])(**val['dist_params'])
    return sampled_dict


class Environment:
    """Environment base class"""

    def __init__(self, n_dim: int, max_batch_size: Optional[int] = None, seed=None):
        super().__init__()
        self.n_dim = n_dim
        self.max_batch_size = max_batch_size
        self.rng = np.random.default_rng(seed)

    def seed(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_trial_info(
        self, 
        trial_info: Optional[pd.DataFrame] = None,
        n: Optional[int] = None,
        sample_space: dict = {},
        stratified: bool = False,
        *args, 
        **kwargs,
    ) -> pd.DataFrame:
        """Sample trial conditions to generate trials from

        Parameters
        ----------
        trial_info : pd.DataFrame
            If provided, overrides the sampling operation and
            just returns the provided trial info
        n : int, optional
            Number of trials to sample
        sample_space : dict
            Dict of dicts specifying the space to sample
            trial conditions from. The upper level dict
            maps trial info field names to sampling parameter
            dicts. Sampling parameter dicts contain keys
            'dist' for sampling distribution name and 
            'dist_params' for distribution parameters
        stratified : bool, default: False
            Whether to perform stratified sampling, to
            guarantee that conditions are roughly equally
            represented

        Returns
        -------
        trial_info : pd.DataFrame
            Dataframe where each row corresponds to a trial
            to simulate
        """
        if trial_info is not None:
            return trial_info
        trial_info = []
        if stratified: # only do stratified sampling over discrete trial info fields
            discrete_keys = [key for key, val in sample_space.items() if val['dist'] == 'discrete']
            discrete_val_counts = [len(sample_space[key]['dist_params']['a']) for key in discrete_keys]
            n_combinations = np.prod(discrete_val_counts)
            trials_per_comb = np.full((n_combinations,), n)
            trials_per_comb = trials_per_comb // n_combinations + (np.arange(n_combinations) < (n % n_combinations))
            change_idx = np.roll(np.cumsum(trials_per_comb))
            change_idx[0] = 0
            comb_generator = itertools.product(*[sample_space[key]['dist_params'] for key in discrete_keys])
        overrides = {}
        for i in range(n):
            if stratified:
                if i in change_idx:
                    overrides = dict(zip(discrete_keys, next(comb_generator)))
            trial_info.append(sample_dict(sample_space, overrides, self.rng))
        trial_info = [trial_info[i] for i in self.rng.permutation(n)]
        return pd.DataFrame(trial_info)
    
    def sample_inputs(
        self, 
        trial_info: Optional[pd.DataFrame] = None,
        inputs: Optional[np.ndarray] = None,
        n: Optional[int] = None,
        *args, 
        **kwargs,
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], Any]:
        """Sample time-varying inputs to the dynamics model

        Returns
        -------
        trial_info : pd.DataFrame
            Dataframe where each row corresponds to a trial
            to simulate
        inputs : np.ndarray
            Sampled inputs to dynamics model, with 
            shape (B,T,I) where B is batch size/trial
            count, T is number of timesteps, and I is
            input dimensionality
        other : dict of np.ndarray
            Optional additional information about the 
            environment that is not input to the dynamics
            model. The dict should map name of the field
            to a (B,T,D) array
        """
        if inputs is not None:
            if trial_info is None:
                trial_info = [{} for _ in range(len(inputs))]
            return trial_info, inputs, None
        raise NotImplementedError

    def simulate(
        self, 
        trial_info: Optional[pd.DataFrame] = None,
        actions: Optional[np.ndarray] = None,
        env_state: Optional[Any] = None,
        *args, 
        **kwargs
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], Any]:
        """Simulate the environment forward in time,
        given trial info to initialize the environment
        or actions from the dynamics model

        Parameters
        ----------
        trial_info : pd.DataFrame, optional
            If provided, should re-initialize the environment(s)
            to match the sampled trial information
        actions : np.ndarray, optional
            If provided, should be a (B,T,I) array of 
            actions on the environment
        env_state : any, optional
            Any state information about the env that
            is useful to provide
        
        Returns
        -------
        info : pd.DataFrame
            Information about each trial, such as cumulative
            reward, to be used to update trial info
        inputs : np.ndarray
            Observations from the environment to be input
            to the dynamics model, with shape (B,T,I), where
            B is the batch size/trial count, T is simulation length,
            and I is the input dimensionality
        other : dict of np.ndarray
            Any other time-varying state information about 
            the environment that should not be passed to the 
            model, such as reward in non-RL settings. The
            dict should map name of the field to a (B,T,D) array
        env_state : any, optional
            Any state information about the env that
            is useful to provide
        """
        raise NotImplementedError