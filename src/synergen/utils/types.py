import numpy as np
import pandas as pd
from typing import Union, Optional
from dataclasses import dataclass


NumpyArray = Union[np.ndarray, np.ma.MaskedArray]


@dataclass
class TrajectoryBatch:
    trajectories: NumpyArray
    trial_info: Optional[pd.DataFrame] = None
    inputs: Optional[NumpyArray] = None
    outputs: Optional[NumpyArray] = None
    other: Optional[dict[str, NumpyArray]] = None
    neural_data: Optional[dict[str, NumpyArray]] = None

    def __post_init__(self, *args, **kwargs):
        self._validate()

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, key):
        expand_to_3d = (
            lambda arr: np.expand_dims(arr, axis=tuple(range(3 - arr.ndim)))
            if arr.ndim < 3
            else arr
        )
        return TrajectoryBatch(
            trajectories=expand_to_3d(self.trajectories[key]),
            trial_info=None if self.trial_info is None else self.trial_info.iloc[key],
            inputs=None if self.inputs is None else expand_to_3d(self.inputs[key]),
            outputs=None if self.outputs is None else expand_to_3d(self.outputs[key]),
            other=None
            if self.other is None
            else {
                other_key: expand_to_3d(other_val[key])
                for other_key, other_val in self.other.items()
            },
            neural_data=None
            if self.neural_data is None
            else {
                neural_key: expand_to_3d(neural_val[key])
                for neural_key, neural_val in self.neural_data.items()
            },
        )

    def _validate(self):
        batch_size = self.trajectories.shape[0]
        trial_len = self.trajectories.shape[1]
        if self.trial_info is not None:
            assert (
                self.trial_info.shape[0] == batch_size
            ), "trial_info must have same number of trials as trajectories"
        if self.inputs is not None:
            assert (
                self.inputs.shape[0] == batch_size
            ), "inputs must have same number of trials as trajectories"
            assert (
                self.inputs.shape[1] == trial_len
            ), "inputs must have same trial length as trajectories"
        if self.outputs is not None:
            assert (
                self.outputs.shape[0] == batch_size
            ), "outputs must have same number of trials as trajectories"
            assert (
                self.outputs.shape[1] == trial_len
            ), "outputs must have same trial length as trajectories"
        if self.other is not None:
            for key, val in self.other.items():
                assert (
                    val.shape[0] == batch_size
                ), f"other['{key}'] must have same number of trials as trajectories"
                assert (
                    val.shape[1] == trial_len
                ), f"other['{key}'] have same trial length as trajectories"
        if self.neural_data is not None:
            for key, val in self.neural_data.items():
                assert (
                    val.shape[0] == batch_size
                ), f"neural_data['{key}'] must have same number of trials as trajectories"
                assert (
                    val.shape[1] == trial_len
                ), f"neural_data['{key}'] have same trial length as trajectories"


def stack_trajectory_batches(trajectory_batches: list[TrajectoryBatch]):
    assert (
        len(trajectory_batches) > 0
    ), "Cannot stack trajectory batch list with length 0"
    if len(trajectory_batches) == 1:
        return trajectory_batches[0]

    def cat(obj_list: Union[list, tuple]):
        if obj_list[0] is None:
            return None
        elif isinstance(obj_list[0], (np.ndarray, np.ma.MaskedArray)):
            return np.concatenate(obj_list, axis=0)
        elif isinstance(obj_list[0], pd.DataFrame):
            return pd.concat(obj_list, axis=0, ignore_index=True)
        elif isinstance(obj_list[0], dict):
            return {
                key: cat([obj[key] for obj in obj_list]) for key in obj_list[0].keys()
            }

    stacked = {
        field: cat([tb.__dict__[field] for tb in trajectory_batches])
        for field in trajectory_batches[0].__dict__.keys()
    }
    return TrajectoryBatch(**stacked)


def shuffle_trajectory_batch(
    trajectory_batch: TrajectoryBatch,
    seed: Optional[Union[int, np.random.Generator]] = None,
):
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    perm = rng.permutation(len(trajectory_batch))
    shuffled_data = {}
    for field in trajectory_batch.__dict__.keys():
        if trajectory_batch.__dict__[field] is None:
            continue
        if isinstance(trajectory_batch.__dict__[field], dict):
            shuffled_data[field] = {
                key: val[perm] for key, val in trajectory_batch.__dict__[field].items()
            }
        else:
            shuffled_data[field] = trajectory_batch.__dict__[field][perm]
    return TrajectoryBatch(**shuffled_data)
