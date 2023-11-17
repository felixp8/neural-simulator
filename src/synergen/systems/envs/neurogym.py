import copy
import numpy as np
import pandas as pd
import inspect
from typing import Optional, Any, Union

from synergen.utils.trial_sampling import SampleSpace

from .base import Environment
from .gym import GymEnvironment

import gym  # still on legacy gym for now
import neurogym as ngym


class NeurogymEnvironment(GymEnvironment):
    def __init__(self, env: ngym.TrialEnv, seed=None, **kwargs):
        super().__init__(env=env, max_batch_size=1, seed=seed, legacy=True, **kwargs)

    def sample_inputs(
        self,
        trial_info: Optional[pd.DataFrame] = None,
        inputs: Optional[np.ndarray] = None,
        n: Optional[int] = None,
        sample_space: Union[SampleSpace, dict] = {},
        stratified: bool = False,
        temporal_data_fields: list[str] = [],
    ):
        if inputs is not None:
            return super().sample_inputs(trial_info=trial_info, ipnuts=inputs, n=n)
        if trial_info is None:
            trial_info = self.sample_trial_info(
                n=n, sample_space=sample_space, stratified=stratified
            )
        inputs = []
        targets = []
        temporal_data = {field: [] for field in temporal_data_fields}
        self.make_batch_envs(batch_size=min(len(trial_info), self.max_batch_size))
        for i in trial_info.index:
            obs, info = self.reset_envs(trial_info.loc[[i], :])
            if info:
                trial_info.loc[i, list(info.keys())] = list(info.values())
            inputs.append(self.batch_envs.ob)
            targets.append(self.batch_envs.gt)
            for field in temporal_data_fields:
                temporal_data[field].append(getattr(self.batch_envs, field))
        inputs = np.stack(inputs, axis=0)
        targets = np.stack(targets, axis=0)
        for field in temporal_data_fields:
            temporal_data[field] = np.stack(temporal_data[field], axis=0)
        if len(targets.shape) == 2:
            targets = targets[:, :, None]
        temporal_data["targets"] = targets
        return trial_info, inputs, temporal_data

    def reset_envs(
        self,
        trial_info: pd.DataFrame,
        option_fields: list[str] = [],
        kwarg_fields: list[str] = [],
    ):
        obs, info = super().reset_envs(
            trial_info=trial_info,
            option_fields=option_fields,
            kwarg_fields=kwarg_fields,
        )
        for key, val in self.batch_envs.start_t.items():
            info[f"{key}_time"] = val / 1000.0
        return obs, info

    def postprocess_step(self, *args):
        if len(args) == 4:  # for old gym API
            return args[0], args[1], args[2], args[2], args[3]
        else:
            assert len(args) == 5
            return tuple(args)
