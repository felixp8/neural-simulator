import numpy as np
import pandas as pd
from typing import Optional, Union, Literal

from .base import System
from ..utils.types import DataBatch, stack_data_batches


class ArraySystem(System):
    def __init__(
        self,
        states: Optional[np.ndarray] = None,
        trial_info: Optional[pd.DataFrame] = None,
        inputs: Optional[np.ndarray] = None,
        outputs: Optional[np.ndarray] = None,
        temporal_data: Optional[dict[str, np.ndarray]] = None,
        data_batch: Optional[DataBatch] = None,
        sample_method: Literal["random", "first", "last"] = "random",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        assert (states is not None) ^ (data_batch is not None)
        if states is not None:
            self.data_batch = DataBatch(
                states=states,
                trial_info=trial_info,
                inputs=inputs,
                outputs=outputs,
                temporal_data=temporal_data,
            )
        else:
            self.data_batch = data_batch
        self.sample_method = sample_method

    def sample_trajectories(
        self,
        n_traj: int,
    ) -> DataBatch:
        data_batches = []
        sampled = 0
        while sampled < n_traj:
            batch_size = min(n_traj - sampled, len(self.data_batch))
            if self.sample_method == "first":
                data_batches.append(self.data_batch[:batch_size])
            elif self.sample_method == "last":
                data_batches.append(self.data_batch[-batch_size:])
            else:
                indices = self.rng.choice(
                    len(self.data_batch, batch_size, replace=False)
                )
                data_batches.append(self.data_batch[indices])
            sampled += batch_size
        return stack_data_batches(data_batches)
