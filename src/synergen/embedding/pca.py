import numpy as np
import sklearn.decomposition as skdecomp
from typing import Optional, Union

from .base import SklearnEmbedding


class PCA(SklearnEmbedding):
    """Dim-reduce input coordinates with PCA"""

    def __init__(
        self,
        n_components: int,
        seed: Optional[Union[int, np.random.Generator]] = None,
        **kwargs,
    ) -> None:
        """Initialize PCA embedding

        Parameters
        ----------
        n_components : int
            Number of dimensions, or top PCs, to keep
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        kwargs
            Any other kwargs to provide to initialize sklearn.decomposition.PCA
        """
        estimator = skdecomp.PCA(n_components=n_components, **kwargs)
        super().__init__(estimator=estimator, seed=seed)
        self.n_components = n_components
        self.kwargs = kwargs

    def set_params(self, params: dict) -> None:
        self.n_components = params["n_components_"]
        super().set_params(params)
        return
