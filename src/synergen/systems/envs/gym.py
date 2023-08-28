import copy
import numpy as np
import pandas as pd
from typing import Optional, Any, Union

from .base import Environment

try:
    import gymnasium
    HAS_GYMNASIUM = True
except:
    HAS_GYMNASIUM = False
try:
    import gym
    HAS_LEGACY = True
except:
    HAS_LEGACY = False

if HAS_GYMNASIUM and HAS_LEGACY:
    EnvType = Union[gym.Env, gymnasium.Env]
elif HAS_GYMNASIUM:
    EnvType = gymnasium.Env
elif HAS_LEGACY:
    EnvType = gym.Env

assert HAS_GYMNASIUM or HAS_LEGACY, "At least one of Gym or Gymnasium must be installed"


def convert_df_to_dict(trial_info: pd.DataFrame):
    vector_info = trial_info.to_dict(orient='list')
    for key in vector_info.keys():
        vector_info[key] = np.array(vector_info[key])
    return vector_info


class GymEnvironment(Environment):
    def __init__(
        self, 
        env: EnvType, 
        max_batch_size: int = 1, 
        info_fields: list[str] = [], 
        other_fields: list[str] = [],
        done_fields: list[str] = [],
        seed=None,
        legacy=False,
        asynchronous=False,
    ):
        super().__init__(
            seed=seed,
            max_batch_size=max_batch_size,
        )
        self.env = env
        self.env.seed(seed)
        self.batch_envs = None
        self.info_fields = info_fields
        self.other_fields = other_fields
        self.done_fields = done_fields
        if legacy:
            assert HAS_LEGACY
        else:
            assert HAS_GYMNASIUM
        self.legacy = legacy
        self.asynchronous = asynchronous

    def simulate(
        self, 
        trial_info: Optional[pd.DataFrame] = None,
        actions: Optional[Union[int, np.ndarray]] = None,
        env_state: Optional[Any] = None,
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], Any]:
        if trial_info is not None:
            self.batch_envs = None
            if self.legacy:
                VectorEnv = gym.vector.AsyncVectorEnv if self.asynchronous else gym.vector.SyncVectorEnv
            else:
                VectorEnv = gymnasium.vector.AsyncVectorEnv if self.asynchronous else gymnasium.vector.SyncVectorEnv
            self.batch_envs = VectorEnv([lambda: copy.deepcopy(self.env) for _ in range(len(trial_info))])
            obs, env_infos = self.batch_envs.reset(options=convert_df_to_dict(trial_info))
            info = {}
            other = {}
            for field in self.info_fields:
                if field in env_infos:
                    info[field] = env_infos[field]
            return pd.DataFrame(info), obs, other, None
        
        assert self.batch_envs is not None, f"You must initialize the envs by first passing in trial info"
        obs, reward, term, trunc, env_infos = self.batch_envs.step(actions)
        info = {'done': np.any([term, trunc] + [env_infos.get(key) for key in self.done_fields], axis=0)}
        other = {}
        if 'reward' in self.other_fields:
            other['reward'] = reward
        for field in self.info_fields:
            if field in env_infos:
                info[field] = env_infos[field]
        for field in self.other_fields:
            if field in env_infos:
                other[field] = env_infos[field]
        return pd.DataFrame(info), obs, other, None