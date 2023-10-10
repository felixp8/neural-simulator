import numpy as np
import inspect
from typing import Optional, Union
from sklearn.base import BaseEstimator


class Embedding:
    def __init__(self, seed: Optional[Union[int, np.random.Generator]] = None):
        super().__init__()
        self.seed(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

    def transform(self, X: np.ndarray, *args, **kwargs):
        raise NotImplementedError

    def get_params(self):
        return dict(embedding_name=self.__class__.__name__)


class SklearnEmbedding(Embedding):
    def __init__(
        self,
        estimator: BaseEstimator,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.estimator = estimator
        self.fit = False
        super().__init__(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        super().seed(seed)
        if (
            "random_state"
            in inspect.signature(self.estimator.__init__).parameters.keys()
        ):
            # TODO: need to figure out if Generator is supported by sklearn in latest releases
            # For now, using dumb workaround instead
            if isinstance(seed, int):
                random_state = seed
            else:
                random_state = self.rng.integers(low=0, high=4294967295, size=1)
            self.estimator.set_params(random_state=random_state)

    def transform(self, X: np.ndarray, refit: bool = False):
        if refit or not self.fit:
            self.estimator.fit(X.reshape(-1, X.shape[-1]))
            self.fit = True
        embedded = self.estimator.transform(X.reshape(-1, X.shape[-1]))
        embedded = embedded.reshape(*X.shape[:-1], embedded.shape[-1])
        return embedded


class EmbeddingStack(Embedding):
    def __init__(
        self,
        embeddings: list[Embedding],
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.embeddings = embeddings
        super().__init__(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        super().seed(seed)
        for embedding in self.embeddings:
            embedding.seed(seed)

    def transform(self, X: np.ndarray, kwarg_list: Optional[list[dict]] = None):
        if kwarg_list is None:
            kwarg_list = [{}] * len(self.embeddings)
        assert len(kwarg_list) == len(
            self.embeddings
        ), f"Number of kwarg dicts must match number of embedding layers"
        for embedding, kwargs in zip(self.embeddings, kwarg_list):
            X = embedding(X, **kwargs)
        return X
