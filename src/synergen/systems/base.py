import numpy as np
import pandas as pd
from typing import Union, Optional
from dataclasses import replace

from .models.base import Model
from .envs.base import Environment
from ..utils.types import DataBatch, stack_data_batches


class System:
    """Generic system base class"""

    def __init__(self, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().__init__()
        self.seed(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed=seed)

    def sample_trajectories(
        self,
        n_traj: int,
        batch_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> DataBatch:
        """Main class method to implement, samples states from system.

        Returns
        -------
        DataBatch
            NamedTuple containing sampled states, as well as other
            data like trial info, inputs, etc. if available
        """
        raise NotImplementedError

    def get_params(self):
        return dict(name=self.__class__.__name__)

    def set_params(self, params: dict) -> None:
        assert "name" in params
        assert params["name"] == self.__class__.__name__
        return


class AutonomousSystem(System):
    """Dynamical system that does not receive external inputs"""

    def __init__(
        self, model: Model, seed: Optional[Union[int, np.random.Generator]] = None
    ) -> None:
        self.model = model
        super().__init__(seed=seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().seed(seed)
        self.model.seed(seed)

    def sample_trajectories(
        self,
        n_traj: int,
        ic_kwargs: dict = {},
        simulation_kwargs: dict = {},
        batch_size: Optional[int] = None,
    ) -> DataBatch:
        batch_size = batch_size or n_traj
        if self.model.max_batch_size is not None:
            batch_size = min(batch_size, self.model.max_batch_size)

        ics = self.model.sample_ics(n=n_traj, **ic_kwargs)  # b x d

        data_batches = []
        for i in range(0, n_traj, batch_size):
            batch_ics = ics[i : (i + batch_size)]
            states, outputs, actions, td = self.model.simulate(
                ics=batch_ics, **simulation_kwargs
            )  # b x t x d
            data_batches.append(
                DataBatch(
                    states=states,
                    outputs=outputs,
                    temporal_data=td,
                )
            )

        data_batch = stack_data_batches(data_batches)
        return data_batch


class NonAutonomousSystem(System):
    """Dynamical system that receives external inputs"""

    def __init__(
        self,
        model: Model,
        env: Environment,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        self.model = model
        self.env = env
        super().__init__(seed=seed)

    def seed(self, seed=None) -> None:
        super().seed(seed)
        self.model.seed(seed)
        self.env.seed(seed)


class CoupledSystem(NonAutonomousSystem):
    """Dynamical system that receives inputs from an environment and acts on the environment"""

    def __init__(
        self,
        model: Model,
        env: Environment,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        super().__init__(model=model, env=env, seed=seed)

    def sample_trajectories(
        self,
        n_traj: int,
        ic_kwargs: dict = {},
        trial_kwargs: dict = {},
        simulation_kwargs: dict = {},
        batch_size: Optional[int] = None,
        max_steps: int = 1000,
    ) -> DataBatch:
        batch_size = batch_size or n_traj
        if self.model.max_batch_size is not None:
            batch_size = min(batch_size, self.model.max_batch_size)
        if self.env.max_batch_size is not None:
            batch_size = min(batch_size, self.env.max_batch_size)

        ics = self.model.sample_ics(n=n_traj, **ic_kwargs)  # b x d
        trial_info = self.env.sample_trial_info(n=n_traj, **trial_kwargs)

        data_batches = []
        for n in range(0, len(trial_info), batch_size):
            batch_trial_info = trial_info.iloc[n : (n + batch_size)].copy()
            batch_ics = ics[n : (n + batch_size)]
            data_batch = self.simulate_coupled_system(
                ics=batch_ics,
                trial_info=batch_trial_info,
                simulation_kwargs=simulation_kwargs,
                max_steps=max_steps,
            )

            data_batches.append(data_batch)

        data_batch = stack_data_batches(data_batches)
        return data_batch

    def simulate_coupled_system(
        self,
        ics: np.ndarray,
        trial_info: pd.DataFrame,
        simulation_kwargs: dict = {},
        max_steps: int = 1000,
    ) -> DataBatch:
        info, inputs, env_td, env_state = self.env.simulate(
            trial_info=trial_info,
            actions=None,
            env_state=None,
            initialize=True,
        )
        if "done" in info.columns:
            env_done = info["done"].to_numpy()
            info.drop("done", axis=1, inplace=True)
        else:
            env_done = np.full((len(trial_info),), False)
        info.index = trial_info.index
        trial_info.loc[trial_info.index, info.columns] = info

        step = 0
        states = ics
        batch_done = False
        data_list = []
        mask_list = [env_done]
        while not batch_done:
            states, outputs, actions, model_td = self.model.simulate(
                ics=states,
                inputs=inputs,
                **simulation_kwargs,
            )  # b x t x d
            info, next_inputs, env_td, env_state = self.env.simulate(
                trial_info=None,
                actions=actions,
                env_state=env_state,
            )

            env_done = np.logical_or(mask_list[-1], info["done"].to_numpy())
            mask_list.append(env_done)
            batch_done = np.all(env_done)
            info.drop("done", axis=1, inplace=True)

            td = {**model_td, **env_td}
            data_list.append((states, inputs, outputs, td))
            info.index = trial_info.index
            trial_info.loc[trial_info.index, info.columns] = info

            inputs = next_inputs
            step += 1
            if step >= max_steps:
                print(
                    f"Warning: Some envs failed to terminate in {max_steps} steps. Forcing termination..."
                )
                break

        base_mask = np.stack(mask_list[:-1], axis=1)
        use_mask = np.any(base_mask)

        def stack_data(arr_list: Union[list, tuple]):
            if arr_list[0] is None:
                return None
            elif isinstance(arr_list[0], dict):
                return {
                    key: stack_data([al[key] for al in arr_list])
                    for key in arr_list[0].keys()
                }
            else:
                stacked = np.stack(arr_list, axis=1)
                if stacked.ndim == 2:
                    stacked = stacked[:, :, None]
                if use_mask:
                    stacked = np.ma.masked_array(
                        stacked, mask=np.tile(base_mask, (1, 1, stacked.shape[-1]))
                    )
                return stacked

        data_batch = DataBatch(
            states=stack_data(data_list[0]),
            inputs=stack_data(data_list[1]),
            outputs=stack_data(data_list[2]),
            temporal_data=stack_data(data_list[3]),
            trial_info=trial_info,
        )
        return data_batch


class UncoupledSystem(NonAutonomousSystem):
    """Dynamical system that receives inputs from an environment but doesn't act on the environment.
    Not really "uncoupled" - just not bidirectionally coupled, so maybe rename"""

    def __init__(
        self,
        model: Model,
        env: Environment,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        super().__init__(model=model, env=env, seed=seed)

    def sample_trajectories(
        self,
        n_traj: int,
        ic_kwargs: dict = {},
        trial_kwargs: dict = {},
        simulation_kwargs: dict = {},
        batch_size: Optional[int] = None,
    ) -> DataBatch:
        batch_size = batch_size or n_traj
        if self.model.max_batch_size is not None:
            batch_size = min(batch_size, self.model.max_batch_size)

        ics = self.model.sample_ics(n=n_traj, **ic_kwargs)  # b x d
        trial_info, inputs, env_td = self.env.sample_inputs(n=n_traj, **trial_kwargs)

        data_batches = []
        for i in range(0, n_traj, batch_size):
            batch_ics = ics[i : (i + batch_size)]
            batch_inputs = inputs[i : (i + batch_size)]
            batch_trial_info = trial_info.iloc[i : (i + batch_size)]
            states, outputs, actions, model_td = self.model.simulate(
                ics=batch_ics, inputs=batch_inputs, **simulation_kwargs
            )  # b x t x d
            data_batches.append(
                DataBatch(
                    states=states,
                    outputs=outputs,
                    temporal_data=model_td,
                )
            )

        data_batch = stack_data_batches(data_batches)
        data_batch = replace(data_batch, trial_info=trial_info, inputs=inputs)
        data_batch.temporal_data.update(env_td)
        return data_batch
