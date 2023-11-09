import numpy as np
import pandas as pd
import itertools
from typing import Optional, Union, Callable, Any

from ...utils.trial_sampling import SampleSpace


class Environment:
    """Environment base class"""

    def __init__(
        self,
        max_batch_size: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.seed(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed=seed)

    def sample_trial_info(
        self,
        trial_info: Optional[pd.DataFrame] = None,
        n: Optional[int] = None,
        sample_space: Union[SampleSpace, dict] = {},
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
        sample_space : dict or SampleSpace
            Dict or object specifying the space to sample
            trial conditions from. see `utils.trial_sampling`
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
        if isinstance(sample_space, dict):
            sample_space = SampleSpace(distributions=sample_space, seed=self.rng)
        trial_info = sample_space.sample(n=n, stratified=stratified)
        return pd.DataFrame(trial_info, index=np.arange(n))

    def sample_inputs(
        self,
        trial_info: Optional[pd.DataFrame] = None,
        inputs: Optional[np.ndarray] = None,
        n: Optional[int] = None,
        *args,
        **kwargs,
    ) -> tuple[pd.DataFrame, np.ndarray, Optional[dict]]:
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
        temporal_data : dict, optional
            Optional additional information about the
            environment that is not input to the dynamics
            model. The dict should map name of the field
            to a (B,T,-1) array
        """
        if inputs is not None:
            if trial_info is None:
                trial_info = pd.DataFrame([{} for _ in range(len(inputs))])
            return trial_info, inputs, None
        raise NotImplementedError

    def simulate(
        self,
        trial_info: Optional[pd.DataFrame] = None,
        actions: Optional[np.ndarray] = None,
        env_state: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> tuple[pd.DataFrame, np.ndarray, Optional[dict], Any]:
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
            reward, to be used to update trial info. Importantly,
            this must also contain a "done" field that indicates
            when the environment has finished a given trial
        inputs : np.ndarray
            Observations from the environment to be input
            to the dynamics model, with shape (B,T,I), where
            B is the batch size/trial count, T is simulation length,
            and I is the input dimensionality
        temporal_data : dict of np.ndarray, optional
            Any other time-varying state information about
            the environment that should not be passed to the
            model, such as reward in non-RL settings. The
            dict should map name of the field to a (B,T,D) array
        env_state : any, optional
            Any state information about the env that
            is useful to provide
        """
        raise NotImplementedError
