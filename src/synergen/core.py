import numpy as np
from typing import Optional, Union
from dataclasses import replace

from .systems.base import System
from .embedding.base import Embedding
from .neural_data.base import DataSampler  # TODO: rename this??
from .utils.data_io import write_file
from .utils.types import (
    DataBatch,
    stack_data_batches,
    shuffle_data_batch,
)


class NeuralDataGenerator:
    def __init__(
        self,
        data_sampler: DataSampler,
        system: Optional[System] = None,
        embedding: Optional[Embedding] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__()
        self.data_sampler = data_sampler
        self.system = system
        self.embedding = embedding
        self.seed(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed=seed)
        self.data_sampler.seed(seed)
        if self.system is not None:
            self.system.seed(seed)
        if self.embedding is not None:
            self.embedding.seed(seed)

    def generate_dataset(
        self,
        data_batch: Optional[DataBatch] = None,
        trajectory_kwargs: dict = {},
        embedding_kwargs: dict = {},
        sampling_kwargs: dict = {},
        export_kwargs: dict = {},
        n_repeats: int = 1,
        shuffle: bool = False,
    ):
        # make latent trajectories
        if data_batch is None:
            assert self.system is not None
            data_batch = self.system.sample_trajectories(**trajectory_kwargs)
            data_batch.general_data.update(dict(system=self.system.get_params()))
        # copy trajectories if desired
        if n_repeats != 1:
            assert n_repeats > 1, "`n_repeats` must be >= 1"
            data_batch = stack_data_batches([data_batch] * n_repeats)
        if shuffle:
            data_batch = shuffle_data_batch(data_batch, self.rng)
        states = data_batch.states
        # embedding
        if self.embedding is not None:
            states = self.embedding.transform(states, **embedding_kwargs)
            data_batch.temporal_data.update(dict(embedded_states=states))
            data_batch.general_data.update(dict(embedding=self.embedding.get_params()))
        # finally, sample neural data
        neural_data = self.data_sampler.sample(states, **sampling_kwargs)
        data_batch = replace(data_batch, neural_data=neural_data)
        data_batch.general_data.update(
            dict(data_sampler=self.data_sampler.get_params())
        )
        # export to file if desired
        if export_kwargs:
            write_file(data_batch=data_batch, **export_kwargs)
        return data_batch
