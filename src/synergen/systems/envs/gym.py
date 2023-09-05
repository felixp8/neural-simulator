import copy
import numpy as np
import pandas as pd
import inspect
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


class GymEnvironment(Environment):
    def __init__(
        self,
        env: EnvType,
        max_batch_size: Optional[int] = None,
        info_kwargs: dict = {},
        done_kwargs: dict = {},
        reset_kwargs: dict = {},
        postprocess_kwargs: dict = {},
        seed=None,
        legacy=False,
        asynchronous=False,
    ):
        self.env = env
        super().__init__(
            seed=seed,
            max_batch_size=max_batch_size,
        )
        self.batch_envs = None
        self.info_kwargs = info_kwargs
        self.done_kwargs = done_kwargs
        self.reset_kwargs = reset_kwargs
        self.postprocess_kwargs = postprocess_kwargs
        if legacy:
            assert HAS_LEGACY
        else:
            assert HAS_GYMNASIUM
        self.legacy = legacy
        self.asynchronous = asynchronous
        self.validate_kwargs()

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        super().seed(seed)
        self.env.reset(seed=seed)

    def simulate(
        self,
        trial_info: Optional[pd.DataFrame] = None,
        actions: Optional[Union[int, np.ndarray]] = None,
        env_state: Optional[Any] = None,
        initialize: bool = False,
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], Any]:
        if initialize:
            assert (
                trial_info is not None
            ), f"`trial_info` must be provided for env initialization"
            self.make_batch_envs(batch_size=len(trial_info))
            obs, env_infos = self.reset_envs(trial_info=trial_info, **self.reset_kwargs)
            reward = np.zeros(len(trial_info))
            term = np.zeros(len(trial_info), dtype=np.bool_)
            trunc = np.zeros(len(trial_info), dtype=np.bool_)
            info = self.parse_info(
                obs=obs,
                reward=reward,
                term=term,
                trunc=trunc,
                env_infos=env_infos,
                **self.info_kwargs,
            )[0]
            info["done"] = self.check_done(
                obs=obs,
                reward=reward,
                term=term,
                trunc=trunc,
                env_infos=env_infos,
                **self.done_kwargs,
            )
            return pd.DataFrame(info), obs, {}, None

        assert (
            self.batch_envs is not None
        ), f"You must initialize the envs by first before stepping"
        step_output = self.batch_envs.step(actions)
        obs, reward, term, trunc, env_infos = self.postprocess_step(
            *step_output, **self.postprocess_kwargs
        )
        info, other = self.parse_info(
            obs=obs,
            reward=reward,
            term=term,
            trunc=trunc,
            env_infos=env_infos,
            **self.info_kwargs,
        )
        info["done"] = self.check_done(
            obs=obs,
            reward=reward,
            term=term,
            trunc=trunc,
            env_infos=env_infos,
            **self.done_kwargs,
        )
        return pd.DataFrame(info), obs, other, None

    def validate_kwargs(self):
        def validate_kwargs_on_func(kwargs, func):
            keys = set(kwargs.keys())
            signature = inspect.signature(func)
            for key in keys:
                assert (
                    key in signature.parameters.keys()
                ), f"kwarg {key} is not supported for func {func.__name__}"
                # could even check type correctness but skipping for now

        # doesn't check overlap between hard-coded args and kwargs, since people might
        # override simulate
        validate_kwargs_on_func(self.info_kwargs, self.parse_info)
        validate_kwargs_on_func(self.done_kwargs, self.check_done)
        validate_kwargs_on_func(self.reset_kwargs, self.reset_envs)
        validate_kwargs_on_func(self.postprocess_kwargs, self.postprocess_step)

    def make_batch_envs(self, batch_size: int):
        # may need kwargs one day?
        self.batch_envs = None
        if batch_size == 1:
            self.batch_envs = copy.deepcopy(self.env)
            return
        if self.legacy:
            VectorEnv = (
                gym.vector.AsyncVectorEnv
                if self.asynchronous
                else gym.vector.SyncVectorEnv
            )
        else:
            VectorEnv = (
                gymnasium.vector.AsyncVectorEnv
                if self.asynchronous
                else gymnasium.vector.SyncVectorEnv
            )
        self.batch_envs = VectorEnv(
            [lambda: copy.deepcopy(self.env) for _ in range(batch_size)]
        )

    def reset_envs(
        self,
        trial_info: pd.DataFrame,
        option_fields: list[str] = [],
        kwarg_fields: list[str] = [],
    ):
        assert self.batch_envs is not None, f"Envs must be initialized before resetting"
        reset_options = {}
        reset_kwargs = {}
        for field in option_fields:
            if field in trial_info.columns:
                reset_options[field] = trial_info.loc[:, field].to_numpy()
        for field in kwarg_fields:
            if field in trial_info.columns:
                reset_kwargs[field] = trial_info.loc[:, field].to_numpy()
        if reset_options:
            reset_kwargs["options"] = reset_options
        if len(trial_info) > 1:
            seed = self.rng.integers(
                low=0, high=4294967295, size=len(trial_info)
            ).tolist()
        else:
            seed = self.rng.integers(low=0, high=4294967295).item()
        obs, env_infos = self.batch_envs.reset(
            seed=seed,
            **reset_kwargs,
        )
        return obs, env_infos

    def postprocess_step(
        self,
        obs: np.ndarray,
        reward: np.ndarray,
        term: np.ndarray,
        trunc: np.ndarray,
        env_infos: dict,
    ):
        # for e.g. converting torch tensors to numpy
        return obs, reward, term, trunc, env_infos

    def parse_info(
        self,
        obs: np.ndarray,
        reward: np.ndarray,
        term: np.ndarray,
        trunc: np.ndarray,
        env_infos: dict,
        info_fields: list[str] = [],
        other_fields: list[str] = [],
    ):
        info = {}
        other = {}
        for field in info_fields:
            if field in env_infos:
                info[field] = env_infos[field]
        for field in other_fields:
            if field == "reward":
                other["reward"] = reward
            elif field in env_infos:
                other[field] = env_infos[field]
        return info, other

    def check_done(
        self,
        obs: np.ndarray,
        reward: np.ndarray,
        term: np.ndarray,
        trunc: np.ndarray,
        env_infos: dict,
        fields: list[str] = [],
    ):
        done = np.any(
            [term, trunc]
            + [
                env_infos.get(key, np.zeros(term.shape, dtype=np.bool_))
                for key in fields
            ],
            axis=0,
        )
        return done
