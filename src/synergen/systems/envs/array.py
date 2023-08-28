import numpy as np
import pandas as pd
from typing import Optional, Union, Literal

from .base import Environment


class ArrayEnvironment(Environment):
    def __init__(
        self, 
        inputs: np.ndarray,
        trial_info: Optional[pd.DataFrame] = None,
        other: Optional[dict[str, np.ndarray]] = None,
        sample_method: Literal["first", "last", "random"] = "random",
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__(seed=seed)
        self.inputs = inputs
        self.trial_info = trial_info
        self.other = other
        self.sample_method = sample_method
    
    def sample_inputs(self, n: int):
        input_list = []
        trial_info_list = []
        other_list = []
        sampled = 0
        while sampled < n:
            batch_size = min(n - sampled, len(self.inputs))
            if self.sample_method == "first":
                indices = np.arange(batch_size)
            elif self.sample_method == "last":
                indices = np.arange(len(self.inputs) - batch_size, len(self.inputs))
            else:
                indices = self.rng.choice(len(self.trajectory_batch, batch_size, replace=False))
            input_list.append(self.inputs[indices])
            if self.trial_info is not None:
                trial_info_list.append(self.trial_info.iloc[indices])
            if self.other is not None:
                other_list.append({key: val[indices] for key, val in self.other.items()})
            sampled += batch_size
        inputs = np.concatenate(input_list, axis=0)
        trial_info = None if len(trial_info_list) == 0 else pd.concat(trial_info_list, axis=0, ignore_index=True)
        other = None if len(other_list) == 0 else {
            key: np.concatenate([ol[key] for ol in other_list], axis=0)
            for key in self.other.keys()
        }
        return inputs, trial_info, other