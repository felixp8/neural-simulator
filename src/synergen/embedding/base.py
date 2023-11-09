import numpy as np
import inspect
from typing import Optional, Union
from sklearn.base import BaseEstimator


class Embedding:
    """Base embedding class"""

    def __init__(self, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        """Initialize Embedding

        Parameters
        ----------
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        super().__init__()
        self.seed(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        """Seed embedding object and initialize random generator

        Parameters
        ----------
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed=seed)

    def transform(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Transforms data points in X

        Parameters
        ----------
        X : np.ndarray
            An array of shape (..., D), where D is
            the dimensionality of each data point

        Returns
        -------
        np.ndarray
            An array of shape (..., E), where E is
            the dimensionality of the transformed/embedded
            data points
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """Returns various parameters of Embedding object to save"""
        return dict(embedding_name=self.__class__.__name__)


class SklearnEmbedding(Embedding):
    """Base class to support using any sklearn estimator as an Embedding"""

    def __init__(
        self,
        estimator: BaseEstimator,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Initialize SklearnEmbedding

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            An initialized sklearn estimator instance
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        self.estimator = estimator
        self.fit = False  # flag indicating whether the estimator's been fit yet
        super().__init__(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().seed(seed)
        if (
            "random_state"
            in inspect.signature(self.estimator.__init__).parameters.keys()
        ):
            if isinstance(seed, int):
                random_state = seed
            else:  # dumb workaround while sklearn lacks support for np.random.Generator
                random_state = self.rng.integers(low=0, high=4294967295, size=1)
            self.estimator.set_params(random_state=random_state)

    def transform(self, X: np.ndarray, refit: bool = False) -> np.ndarray:
        """Transforms data points in X with sklearn estimator

        Parameters
        ----------
        X : np.ndarray
            An array of shape (..., D), where D is
            the dimensionality of each data point
        refit : bool, default: False
            Whether to re-fit the sklearn estimator on
            the new input data points. If this is the first
            call to `transform`, the estimator will be
            fit to the data anyway

        Returns
        -------
        np.ndarray
            An array of shape (..., E), where E is
            the dimensionality of the transformed/embedded
            data points
        """
        if refit or not self.fit:
            self.estimator.fit(X.reshape(-1, X.shape[-1]))
            self.fit = True
        embedded = self.estimator.transform(X.reshape(-1, X.shape[-1]))
        embedded = embedded.reshape(*X.shape[:-1], embedded.shape[-1])
        return embedded


class EmbeddingStack(Embedding):
    """Utility class to apply multiple embedding transformations sequentially"""

    def __init__(
        self,
        embeddings: list[Embedding],
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Initialize EmbeddingStack

        Parameters
        ----------
        embeddings : list of Embedding
            List containing Embedding instances to apply
            in the provided order
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        self.embeddings = embeddings
        super().__init__(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        """Seed embedding object and initialize random generator.
        Seeds each separate embedding in the stack with the same seed

        Parameters
        ----------
        seed : int or np.random.Generator, optional
            Seed or generator for reproducibility
        """
        super().seed(seed)
        for embedding in self.embeddings:
            embedding.seed(seed)

    def transform(
        self, X: np.ndarray, kwarg_list: Optional[list[dict]] = None
    ) -> np.ndarray:
        """Transforms data points in X with embedding stack

        Parameters
        ----------
        X : np.ndarray
            An array of shape (..., D), where D is
            the dimensionality of each data point
        kwarg_list : list of dict, optional
            List of kwargs to apply to each embedding
            object. If provided, its length must be
            equal to the length of the list of embeddings.
            If some embeddings need kwargs but a particular
            embedding does not, provide an empty dict at
            the corresponding index in kwarg_list

        Returns
        -------
        np.ndarray
            An array of shape (..., E), where E is
            the dimensionality of the transformed/embedded
            data points
        """
        if kwarg_list is None:
            kwarg_list = [{}] * len(self.embeddings)
        assert len(kwarg_list) == len(
            self.embeddings
        ), f"Number of kwarg dicts must match number of embedding layers"
        for embedding, kwargs in zip(self.embeddings, kwarg_list):
            X = embedding(X, **kwargs)
        return X

    def get_params(self) -> dict:
        """Returns various parameters of embedding object to save.
        Creates nested dictionary with params for each embedding
        under an upper-level key "Embedding{i}" where `i` is the
        position of that embedding in the list
        """
        params = super().get_params()
        params.update(
            {f"Embedding{i:02d}": e.get_params() for i, e in enumerate(self.embeddings)}
        )
        return params
