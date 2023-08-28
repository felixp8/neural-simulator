import numpy as np
from typing import Optional, Union

from .systems.base import System
from .embedding.base import Embedding
from .synthetic_data.base import DataSampler # TODO: rename this??
from .utils.data_export import export_to_nwb


class NeuralDataGenerator:
    def __init__(
        self, 
        system: System, 
        data_sampler: DataSampler,
        embedding: Optional[Embedding] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        super().__init__()
        self.system = system
        self.data_sampler = data_sampler
        self.embedding = embedding
        self.seed(seed)

    def seed(self, seed: Optional[Union[int, np.random.Generator]] = None):
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)
        self.system.seed(seed)
        self.data_sampler.seed(seed)
        if self.embedding is not None:
            self.embedding.seed(seed)

    def generate_dataset(
        self, 
        trajectory_kwargs: dict = {}, 
        embedding_kwargs: dict = {},
        sampling_kwargs: dict = {},
        nwb_export: bool = False,
    ):
        # make latent trajectories
        trajectory_batch = self.system.sample_trajectories(**trajectory_kwargs) # traj: B x T x D
        trajectories = trajectory_batch.trajectories
        # embedding
        if self.embedding is not None:
            trajectories = self.embedding.transform(trajectories, **embedding_kwargs)
        # finally, sample neural data
        neural_data = self.data_sampler.sample(trajectories, **sampling_kwargs)
        trajectory_batch = trajectory_batch._replace(neural_data=neural_data) # TODO: decide on a good data format
        # export to NWB if desired
        if nwb_export:
            export_to_nwb(trajectory_batch)
        return trajectory_batch

    # TODO: think about making torch/tf dataloaders
