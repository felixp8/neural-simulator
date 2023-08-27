import numpy as np
from typing import Optional, Union, Literal

from .systems.base import System, TrajectoryBatch
from .utils import (
    get_dim_reduction_function, 
    get_manifold_embedding_function, 
    get_neural_sampling_function, 
    export_to_nwb
)


class NeuralDataGenerator: # TODO: unsure if this really warrants its own class
    def __init__(self, system: System, seed=None):
        super().__init__()
        self.system = system
        self.random_seed = seed
        self.system.seed(seed)
        self.rng = np.random.default_rng(self.random_seed)

    def seed(self, seed=None):
        self.random_seed = seed
        self.system.seed(seed)
        self.rng = np.random.default_rng(self.random_seed)

    def generate_dataset(
        self, 
        trajectory_kwargs: dict = {}, 
        reduced_dim: Optional[int] = None,
        dim_reduction_method: Optional[str] = "PCA",
        dim_reduction_kwargs: dict = {},
        subsampled_dim: Optional[int] = None,
        manifold_embedding: Optional[str] = None,
        manifold_embedding_kwargs: dict = {},
        # TODO: want to figure out how to span multiple data modalities here
        neural_sampling: Literal["lin_nonlin", "depasquale"] = "lin_nonlin", 
        neural_sampling_kwargs: dict = {},
        nwb_export: bool = False,
    ):
        # make latent trajectories
        trajectory_batch = self.system.sample_trajectories(**trajectory_kwargs) # traj: B x T x D
        trajectories = trajectory_batch.trajectories
        # dim reduce if desired
        if reduced_dim is not None:
            dim_reduction_fn = get_dim_reduction_function(reduced_dim, dim_reduction_method, dim_reduction_kwargs)
            trajectories = dim_reduction_fn(trajectories)
        # partially observe dimensions if desired
        if subsampled_dim is not None:
            if subsampled_dim < trajectories.shape[-1]:
                subsampled_idx = self.rng.choice(trajectories.shape[-1], size=subsampled_dim, replace=False)
                trajectories = trajectories[:,:,subsampled_idx]
        # embed latents in nonlinear manifold if desired
        if manifold_embedding is not None:
            manifold_embedding_fn = get_manifold_embedding_function(manifold_embedding, manifold_embedding_kwargs)
            trajectories = manifold_embedding_fn(trajectories)
        # finally, sample neural data
        neural_sampling_fn = get_neural_sampling_function(neural_sampling, neural_sampling_kwargs)
        neural_data = neural_sampling_fn(trajectories) # could return dict with keys like 'rates', 'spikes' mapping to tensors?
        trajectory_batch = TrajectoryBatch(
            trajectories=trajectory_batch.trajectories,
            trial_info=trajectory_batch.trial_info,
            inputs=trajectory_batch.inputs,
            outputs=trajectory_batch.outputs,
            targets=trajectory_batch.targets,
            other=trajectory_batch.other,
            neural_data=neural_data,
        ) # TODO: decide on a good data format
        # export to NWB if desired
        if nwb_export:
            export_to_nwb(trajectory_batch)
        return trajectory_batch

    # TODO: think about making torch/tf dataloaders
