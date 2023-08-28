import numpy as np
from typing import Optional, Union


class DataSampler:
    def __init__(self, seed: Optional[Union[int, np.random.Generator]] = None):
        super().__init__()
        self.seed(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)
    
    def sample(self, trajectories: np.ndarray, **kwargs):
        raise NotImplementedError
