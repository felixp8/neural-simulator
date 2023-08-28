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


def stack_trajectory_batches(trajectory_batches: list[TrajectoryBatch]):
    assert len(trajectory_batches) > 0, "Cannot stack trajectory batch list with length 0"
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