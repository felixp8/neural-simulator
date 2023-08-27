import numpy as np

from neural_simulator.neural_data_generator import NeuralDataGenerator
from neural_simulator.systems.base import AutonomousSystem
from neural_simulator.systems.models.dysts import DystsModel


seed = 0

datagen = NeuralDataGenerator(
    system=AutonomousSystem(
        model=DystsModel(name="Lorenz", params={}),
    ),
    seed=seed,
)

trajectory_kwargs = dict(
    n_traj=500,
    ic_kwargs=dict(
        dist="normal",
        dist_params=dict(
            loc=np.array([0., 0., 20]),
            scale=np.array([3., 3., 3.]),
        )
    ),
    simulation_kwargs=dict(
        trial_len=100,
        burn_in=100,
        pts_per_period=100,
        standardize=True,
    )
)

neural_sampling_kwargs = dict(
    projection_dim=30,
    proj_weight_dist="uniform",
    proj_weight_params=dict(
        low=-1.0,
        high=1.0,
    ),
    nonlinearity="exp",
    nonlinearity_params=dict(),
    noise_dist="poisson",
    noise_params=dict(),
    sort_dims=True,
    standardize=True,
    seed=seed,
)

manifold_embedding_kwargs = dict(
    x_scale=1.0,
    y_scale=2.0,
    x_center=0.0,
    y_center=0.0,
    rotate=True,
    seed=seed,
)

output = datagen.generate_dataset(
    trajectory_kwargs=trajectory_kwargs,
    subsampled_dim=2,
    manifold_embedding="parabolic3d",
    manifold_embedding_kwargs=manifold_embedding_kwargs,
    neural_sampling="lin_nonlin",
    neural_sampling_kwargs=neural_sampling_kwargs,
)
