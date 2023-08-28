import numpy as np
import sklearn.decomposition as skdecomp
from typing import Optional, Union

from .base import SklearnEmbedding


class PCA(SklearnEmbedding):
    def __init__(
        self,
        n_components: int,
        seed: Optional[Union[int, np.random.Generator]] = None,
        **kwargs,
    ):
        estimator = skdecomp.PCA(n_components=n_components, **kwargs)
        super().__init__(estimator=estimator, seed=seed)
