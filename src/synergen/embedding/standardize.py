import numpy as np
import sklearn.preprocessing as StandardScaler
from typing import Optional, Union

from .base import SklearnEmbedding


class StandardScaler(SklearnEmbedding):
    """Standardize input coordinates with StandardScaler"""

    def __init__(
        self,
        seed: Optional[Union[int, np.random.Generator]] = None,
        **kwargs,
    ) -> None:
        """Initialize StandardScaler embedding

        Parameters
        ----------
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        kwargs
            Any other kwargs to provide to initialize sklearn.preprocessing.StandardScaler
        """
        estimator = StandardScaler(**kwargs)
        super().__init__(estimator=estimator, seed=seed)
        self.kwargs = kwargs
