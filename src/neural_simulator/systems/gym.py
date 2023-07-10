import numpy as np
import pandas as pd
import gym
import ivy
import itertools

from .base import System


def sample_dict(
    sample_space: dict,
    overrides: dict = {},
    rng: np.random.Generator = None,
):
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
):
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


class RNNGymSystem(System):
    def __init__(
        self, 
        model,
        env: gym.Env,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.env = env
        self.env.seed(seed)
        self.rng = np.random.default_rng(seed)

    def sample_trajectories(self, ic_kwargs={}, trial_kwargs={}, simulation_kwargs={}):
        ics = self.sample_ics(**ic_kwargs)
        trial_params = self.sample_trials(**trial_kwargs)
        trajectories, inputs = self.simulate_system(ics, trial_params)
        trial_info = pd.DataFrame(trial_params)
        for i in range(ics.shape[1]):
            trial_info[f'ic_dim{i}'] = ics[:, i]
        return trajectories, trial_info, inputs

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

    def sample_trials(
        self,
        n_trials: int,
        sample_space: dict = {},
        stratified: bool = False,
    ):
        # sample space is dict of dicts like so:
        # {'trial_info_field': {'dist': 'distribution_name', 'dist_params': 'distribution_params'}}
        # where 'distribution_name' is some random distribution (e.g. normal, poisson, discrete)
        # for most distributions, `distribution_params` is the numpy args for that distribution
        # for discrete, `distribution_params` is a list of possible values. not supporting weighted sampling right now
        if not sample_space:
            return None
        return sample_trial_params(sample_space=sample_space, n_trials=n_trials, stratified=stratified)

    def simulate_system(
        self,
        ics: np.ndarray,
        trial_params: list[dict],
    ):
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
            trajectory = ivy.stack(trajectory, axis=0).to_numpy()
            obs_list = ivy.stack(obs_list, axis=0).to_numpy()
            trajectories.append(trajectory)
            inputs.append(obs_list)
        trajectories = np.stack(trajectories, axis=0)
        inputs = np.stack(inputs, axis=0)
        return trajectories, inputs

    def step(self, state: ivy.Array):
        input = ivy.array(self.obs)
        state, output = self.model(state, input)
        action = ivy.argmax(output)
        self.obs, reward, term, trunc, info = self.env.step(action)
        return state, input, (term or trunc)
