import numpy as np
import pandas as pd
from typing import Union, Optional, NamedTuple
from collections.abc import Callable

from .models.base import Model
from .envs.base import Environment


NumpyArray = Union[np.ndarray, np.ma.MaskedArray]

class TrajectoryBatch(NamedTuple):
    trajectories: NumpyArray
    trial_info: Optional[pd.DataFrame] = None
    inputs: Optional[NumpyArray] = None
    outputs: Optional[NumpyArray] = None
    targets: Optional[NumpyArray] = None
    other: Optional[dict[str, NumpyArray]] = None
    neural_data: Optional[dict[str, NumpyArray]] = None

def stack_trajectory_batches(trajectory_batches: list[TrajectoryBatch]):
    def cat(obj_list: list):
        if isinstance(obj_list[0], (np.ndarray, np.ma.MaskedArray)):
            return np.concatenate(obj_list, axis=0)
        elif isinstance(obj_list[0], pd.DataFrame):
            return pd.concat(obj_list, axis=0, ignore_index=True)
        elif isinstance(obj_list[0], dict):
            return {
                key: cat([obj[key] for obj in obj_list])
                for key in obj_list[0].keys()
            }
    stacked = [cat(list(zipped)) for zipped in zip(*trajectory_batches)]
    return TrajectoryBatch(*stacked)


class System:
    """Generic system base class"""
    
    def __init__(self, seed=None) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def seed(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_trajectories(self, n_traj: int, *args, **kwargs) -> TrajectoryBatch:
        """Main class method to implement, samples trajectories from system.

        Returns
        -------
        TrajectoryBatch
            NamedTuple containing sampled trajectories, as well as other
            data like trial info, inputs, etc. if available
        """
        raise NotImplementedError


class AutonomousSystem(System):
    """Dynamical system that does not receive external inputs"""

    def __init__(self, model: Model, seed=None) -> None:
        super().__init__(seed=seed)
        self.model = model

    def seed(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)
        self.model.seed(seed)

    def sample_trajectories(
        self, 
        n_traj: int, 
        ic_kwargs: dict = {}, 
        simulation_kwargs: dict = {}
    ) -> TrajectoryBatch:
        ics = self.model.sample_ics(n=n_traj, **ic_kwargs) # b x d
        trajectories, outputs, _ = self.model.simulate(ics=ics, **simulation_kwargs) # b x t x d
        trial_info = None
        if np.unique(ics, axis=0).shape[0] > 0:
            format_str = f"0{int(np.log10(ics.shape[0]).round())}d"
            trial_info = pd.DataFrame(ics, columns=[f'ic_dim{i:{format_str}}' for i in range(ics.shape[1])])
        trajectory_batch = TrajectoryBatch(
            trajectories=trajectories,
            trial_info=trial_info,
            outputs=outputs,
        )
        return trajectory_batch


class NonAutonomousSystem(System):
    """Dynamical system that receives external inputs"""

    def __init__(self, model: Model, env: Environment, seed=None) -> None:
        super().__init__(seed=seed)
        self.model = model
        self.env = env

    def seed(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)
        self.model.seed(seed)
        self.env.seed(seed)


class CoupledSystem(NonAutonomousSystem):
    """Dynamical system that receives inputs from an environment and acts on the environment"""

    def __init__(self, model, env, seed=None) -> None:
        super().__init__(model=model, env=env, seed=seed)

    def sample_trajectories(
        self,
        n_traj: int,
        ic_kwargs: dict = {},
        trial_kwargs: dict = {},
        simulation_kwargs: dict = {},
        max_steps: int = 100,
    ) -> TrajectoryBatch:
        ics = self.model.sample_ics(n=n_traj, **ic_kwargs) # b x d
        trial_info = self.env.sample_trial_info(n=n_traj, **trial_kwargs)
        batch_size = self.env.max_batch_size or len(trial_info)
        trajectory_batches = []
        for n in range(0, len(trial_info), batch_size):
            batch_trial_info = trial_info.iloc[n:(n+batch_size)].copy()
            batch_states = ics[n:(n+batch_size)]
            step = 0
            actions = None
            env_state = None
            batch_done = False
            trajectory_list = []
            output_list = []
            input_list = []
            mask_list = [np.full((len(batch_trial_info),), False)]
            target_list = []
            other_list = []
            while not batch_done:
                info, inputs, other, env_state = self.env.simulate(
                    trial_info=(batch_trial_info if step == 0 else None), 
                    actions=(actions if step > 0 else None),
                    env_state=env_state,
                )
                trajectories, outputs, actions = self.model.simulate(ics=batch_states, inputs=inputs, **simulation_kwargs) # b x t x d
                env_done = np.logical_or(mask_list[-1], info['done'].to_numpy())
                batch_done = np.all(env_done)
                trajectory_list.append(trajectories)
                output_list.append(outputs)
                input_list.append(inputs)
                mask_list.append(env_done)
                target_list.append(None if (other is None) else other.pop('targets', None))
                other_list.append(other)
                info.drop('done', axis=1, inplace=True)
                batch_trial_info.loc[batch_trial_info.index, info.columns] = info
                step += 1
                if step >= max_steps:
                    print(f"Warning: Some envs failed to terminate in {max_steps} steps. Forcing termination...")
                    break
            base_mask = np.stack(mask_list[:-1], axis=1)
            if np.any(base_mask):
                trajectories = np.ma.masked_array(np.stack(trajectory_list, axis=1), mask=np.tile(base_mask, (1,1,trajectory_list[0].shape[-1])))
                outputs = np.ma.masked_array(np.stack(output_list, axis=1), mask=np.tile(base_mask, (1,1,output_list[0].shape[-1]))) if output_list[0] is not None else None
                inputs = np.ma.masked_array(np.stack(input_list, axis=1), mask=np.tile(base_mask, (1,1,input_list[0].shape[-1]))) if input_list[0] is not None else None
                targets = np.ma.masked_array(np.stack(target_list, axis=1), mask=np.tile(base_mask, (1,1,target_list[0].shape[-1]))) if target_list[0] is not None else None
                other = {
                    key: np.ma.masked_array(
                        np.stack([o[key] for o in other_list], axis=1), 
                        mask=np.tile(base_mask, (1,1,other_list[0][key].shape[-1]))
                    )
                    for key in other_list[0].keys()
                } if other_list[0] is not None else None
            else:
                trajectories = np.stack(trajectory_list, axis=1)
                outputs = np.stack(output_list, axis=1) if output_list[0] is not None else None
                inputs = np.stack(input_list, axis=1) if input_list[0] is not None else None
                targets = np.stack(target_list, axis=1) if target_list[0] is not None else None
                other = {
                    key: np.stack([o[key] for o in other_list], axis=1)
                    for key in other_list[0].keys()
                } if other_list[0] is not None else None
            trajectory_batches.append(
                TrajectoryBatch(
                    trajectories=trajectories,
                    trial_info=batch_trial_info,
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets,
                    other=other,
                )
            )
        trajectory_batch = stack_trajectory_batches(trajectory_batches)
        trial_info = trajectory_batch.trial_info
        if np.unique(ics, axis=0).shape[0] > 0:
            format_str = f"0{int(np.log10(ics.shape[0]).round())}d"
            cols = [f'ic_dim{i:{format_str}}' for i in range(ics.shape[1])]
            trial_info_ic = pd.DataFrame(ics, columns=cols, index=trial_info.index)
            trial_info = pd.concat([trial_info, trial_info_ic], axis=1)
        trajectory_batch = TrajectoryBatch(
            trajectories=trajectory_batch.trajectories,
            trial_info=trial_info,
            inputs=trajectory_batch.inputs,
            outputs=trajectory_batch.outputs,
            targets=trajectory_batch.targets,
            other=trajectory_batch.other,
        )
        return trajectory_batch


class UncoupledSystem(NonAutonomousSystem):
    """Dynamical system that receives inputs from an environment but doesn't act on the environment.
    Not really "uncoupled" - just not bidirectionally coupled, so maybe rename"""

    def __init__(self, model, env, seed=None) -> None:
        super().__init__(model=model, env=env, seed=seed)
    
    def sample_trajectories(
        self,
        n_traj: int,
        ic_kwargs: dict = {},
        trial_kwargs: dict = {},
        simulation_kwargs: dict = {},
    ) -> TrajectoryBatch:
        ics = self.model.sample_ics(n=n_traj, **ic_kwargs) # b x d
        trial_info, inputs, other = self.env.sample_inputs(n=n_traj, **trial_kwargs)
        trajectories, outputs = self.model.simulate(ics=ics, inputs=inputs, **simulation_kwargs) # b x t x d
        if np.unique(ics, axis=0).shape[0] > 0:
            format_str = f"0{int(np.log10(ics.shape[0]).round())}d"
            cols = [f'ic_dim{i:{format_str}}' for i in range(ics.shape[1])]
            trial_info_ic = pd.DataFrame(ics, columns=cols, index=trial_info.index)
            trial_info = pd.concat([trial_info, trial_info_ic], axis=1)
        targets = other.pop('targets', None)
        trajectory_batch = TrajectoryBatch(
            trajectories=trajectories,
            trial_info=trial_info,
            inputs=inputs,
            outputs=outputs,
            targets=targets,
            other=other,
        )
        return trajectory_batch
