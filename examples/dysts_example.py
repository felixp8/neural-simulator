import numpy as np

from synergen.core import NeuralDataGenerator
from synergen.systems.base import AutonomousSystem
from synergen.systems.models.dysts import DystsModel
from synergen.synthetic_data.lin_nonlin import LinearNonlinearPoisson


seed = 0

datagen = NeuralDataGenerator(
    system=AutonomousSystem(
        model=DystsModel(name="Lorenz", params={}),
    ),
    data_sampler=LinearNonlinearPoisson(
        input_dim=3,
        output_dim=30,
        proj_weight_dist="uniform",
        proj_weight_params=dict(
            low=-1.0,
            high=1.0,
        ),
        nonlinearity="exp",
        nonlinearity_params=dict(
            offset=-3.0,
        ),
        sort_dims=True,
        standardize=True,
        clip_val=5,
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

output = datagen.generate_dataset(
    trajectory_kwargs=trajectory_kwargs,
)
