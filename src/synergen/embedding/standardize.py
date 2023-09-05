import numpy as np
import sklearn.preprocessing as StandardScaler
from typing import Optional, Union

from .base import SklearnEmbedding


class StandardScaler(SklearnEmbedding):
    def __init__(
        self,
        n_components: int,
        seed: Optional[Union[int, np.random.Generator]] = None,
        **kwargs,
    ):
        estimator = StandardScaler(**kwargs)
        super().__init__(estimator=estimator, seed=seed)
