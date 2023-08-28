import numpy as np
import pandas as pd
from typing import Union, Optional, NamedTuple


NumpyArray = Union[np.ndarray, np.ma.MaskedArray]


class TrajectoryBatch(NamedTuple):
    trajectories: NumpyArray
    trial_info: Optional[pd.DataFrame] = None
    inputs: Optional[NumpyArray] = None
    outputs: Optional[NumpyArray] = None
    targets: Optional[NumpyArray] = None
    other: Optional[dict[str, NumpyArray]] = None
    neural_data: Optional[dict[str, NumpyArray]] = None

    def __len__(self):
        return self.trajectories.shape[0]
    
    def __getitem__(self, key):
        expand_to_3d = lambda arr: np.expand_dims(arr, axis=tuple(range(3 - arr.ndim))) if arr.ndim < 3 else arr
        return TrajectoryBatch(
            trajectories=expand_to_3d(self.trajectories[key]),
            trial_info=None if self.trial_info is None else self.trial_info.iloc[key], 
            inputs=None if self.inputs is None else expand_to_3d(self.inputs[key]), 
            outputs=None if self.outputs is None else expand_to_3d(self.outputs[key]), 
            targets=None if self.targets is None else expand_to_3d(self.targets[key]), 
            other=None if self.other is None else {
                other_key: expand_to_3d(other_val[key]) for other_key, other_val in self.other.items()
            },
            neural_data=None if self.neural_data is None else {
                neural_key: expand_to_3d(neural_val[key]) for neural_key, neural_val in self.neural_data.items()
            }
        )


def stack_trajectory_batches(trajectory_batches: list[TrajectoryBatch]):
    assert len(trajectory_batches) > 0, "Cannot stack trajectory batch list with length 0"
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
                key: cat([obj[key] for obj in obj_list])
                for key in obj_list[0].keys()
            }
    stacked = [cat(zipped) for zipped in zip(*trajectory_batches)]
    return TrajectoryBatch(*stacked)