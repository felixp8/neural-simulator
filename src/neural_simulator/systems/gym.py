import numpy as np
import pandas as pd
import gymnasium as gym
import ivy
import itertools

from .base import System, CoupledSystem


def sample_dict(
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
                sampled_dict[key] = rng.choice()
            else:
                sampled_dict[key] = getattr(rng, val['dist'])(**val['distribution_params'])
    return sampled_dict


def sample_trial_params(
    sample_space: dict,
    n_trials: int,
    stratified: bool,
    rng: np.random.Generator = None,
) -> list[dict]:
    if rng is None:
        rng = np.random.default_rng()
    trial_params = []
    if stratified:
        discrete_keys = [key for key, val in sample_space.items() if val['dist'] == 'discrete']
        discrete_val_counts = [len(sample_space[key]['distribution_params']) for key in discrete_keys]
        n_combinations = np.prod(discrete_val_counts)
        trials_per_comb = np.full((n_combinations,), n_trials)
        trials_per_comb = trials_per_comb // n_combinations + (np.arange(n_combinations) < (n_trials % n_combinations))
        change_idx = np.roll(np.cumsum(trials_per_comb))
        change_idx[0] = 0
        comb_generator = itertools.product(*[sample_space[key]['distribution_params'] for key in discrete_keys])
    overrides = {}
    for i in range(n_trials):
        if stratified:
            if i in change_idx:
                overrides = dict(zip(discrete_keys, next(comb_generator)))
        trial_params.append(sample_dict(sample_space, overrides, rng))
    trial_params = [trial_params[i] for i in rng.permutation(n_trials)]
    return trial_params


class EnvSystem(CoupledSystem):
    """Dynamical system interacting with a Gym environment"""
    def __init__(
        self, 
        system,
        env: gym.Env,
        n_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_dim=n_dim, n_input_dim=env.observation_space.n, seed=seed)
        self.env = env
        self.env.seed(seed)

    def sample_trials(
        self,
        n_trials: int,
        sample_space: dict = {},
        stratified: bool = False,
    ) -> list[dict]:
        # sample space is dict of dicts like so:
        # {'trial_info_field': {'dist': 'distribution_name', 'dist_params': 'distribution_params'}}
        # where 'distribution_name' is some random distribution (e.g. normal, poisson, discrete)
        # for most distributions, `distribution_params` is the numpy args for that distribution
        # for discrete, `distribution_params` is a list of possible values. not supporting weighted sampling right now
        if not sample_space:
            return [{}] * n_trials
        return sample_trial_params(sample_space=sample_space, n_trials=n_trials, stratified=stratified)

    def simulate_system(
        self,
        ics: np.ndarray,
        trial_params: list[dict],
    ) -> tuple[ivy.Array, ivy.Array]:
        assert ics.shape[0] == len(trial_params)
        trajectories = []
        inputs = []
        for ic, params in zip(ics, trial_params):
            self.obs, _ = self.env.reset(**params)
            done = False
            state = ivy.array(ic)
            trajectory = []
            obs_list = []
            while not done:
                state, obs, done = self.step(state)
                trajectory.append(state)
                obs_list.append(obs)
            trajectory = ivy.stack(trajectory, axis=0)
            obs_list = ivy.stack(obs_list, axis=0)
            trajectories.append(trajectory)
            inputs.append(obs_list)
        trajectories = ivy.stack(trajectories, axis=0)
        inputs = ivy.stack(inputs, axis=0)
        return trajectories, inputs

    def step(self, state: ivy.Array) -> tuple[ivy.Array, ivy.Array, bool]:
        input = ivy.array(self.obs)
        state, action = self.system_step(state, input)
        self.obs, reward, term, trunc, info = self.env.step(action)
        return state, input, (term or trunc)

    def system_step(self, *args, **kwargs):
        raise NotImplementedError
