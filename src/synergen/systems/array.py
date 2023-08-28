import numpy as np
import pandas as pd
from typing import Optional, Union, Literal

from .base import System
from ..utils.types import TrajectoryBatch, stack_trajectory_batches


class ArraySystem(System):
    def __init__(
        self, 
        trajectories: Optional[np.ndarray] = None,
        trial_info: Optional[pd.DataFrame] = None,
        inputs: Optional[np.ndarray] = None,
        outputs: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        other: Optional[dict[str, np.ndarray]] = None,
        trajectory_batch: Optional[TrajectoryBatch] = None,
        sample_method: Literal["random", "first", "last"] = "random",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        assert (trajectories is not None) ^ (trajectory_batch is not None)
        if trajectories is not None:
            self.trajectory_batch = TrajectoryBatch(
                trajectories=trajectories,
                trial_info=trial_info,
                inputs=inputs,
                outputs=outputs,
                targets=targets,
                other=other,
            )
        else:
            self.trajectory_batch = trajectory_batch
        self.sample_method = sample_method
    
    def sample_trajectories(
        self, 
        n_traj: int, 
    ):
        trajectory_batches = []
        sampled = 0
        while sampled < n_traj:
            batch_size = min(n_traj - sampled, len(self.trajectory_batch))
            if self.sample_method == "first":
                trajectory_batches.append(self.trajectory_batch[:batch_size])
            elif self.sample_method == "last":
                trajectory_batches.append(self.trajectory_batch[-batch_size:])
            else:
                indices = self.rng.choice(len(self.trajectory_batch, batch_size, replace=False))
                trajectory_batches.append(self.trajectory_batch[indices])
            sampled += batch_size
        return stack_trajectory_batches(trajectory_batches)
