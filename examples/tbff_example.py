import sys

sys.path.insert(1, "/home/fpei2/interp/InterpretabilityBenchmark/")
# sys.path.insert(1, '/home/csverst/Github/InterpretabilityBenchmark/')

from interpretability.task_modeling.task_wrapper.decoupled import TaskTrainedDecoupled
from interpretability.task_modeling.model.task_networks import NODE
from interpretability.task_modeling.task_env.task_env import NBitFlipFlop

import numpy as np
import pandas as pd
import torch
from typing import Optional, Any

from synergen.core import NeuralDataGenerator
from synergen.systems.base import UncoupledSystem
from synergen.systems.models.base import Model
from synergen.systems.envs.base import Environment
from synergen.embedding.pca import PCA
from synergen.synthetic_data.lin_nonlin import LinearNonlinearPoisson


# load stuff from the run

ckpt_path = "/snel/share/runs/dysts-learning/MultiDatasets/NODE/20230828_TBFF_NODE_Felix/train_2023-08-28_11-55-46/0_latent_size=3,seed=0,learning_rate=0.0005/last.ckpt"

task_env = NBitFlipFlop(
    n=3,
    n_timesteps=200,
    noise=0.05,
)

model = NODE(
    num_layers=3,
    layer_hidden_size=128,
    latent_size=3,
)
n_outputs = task_env.action_space.shape[0]
n_inputs = task_env.observation_space.shape[0]
model.init_model(n_inputs, n_outputs)

pl_module = TaskTrainedDecoupled(
    learning_rate=0.0005,
    weight_decay=1e-5,
    model=model,
)
pl_module.load_state_dict(torch.load(ckpt_path)["state_dict"])
model = pl_module.model


# define necessary classes


class NODEModel(Model):
    def __init__(self, model: NODE, seed: Optional[int] = None):
        super().__init__(
            n_dim=model.latent_size,
            seed=seed,
        )
        self.model = model
        self.dtype = list(model.parameters())[0].dtype

    def simulate(
        self,
        ics: np.ndarray,
        inputs: np.ndarray,
    ):
        # import pdb; pdb.set_trace()
        ics = torch.from_numpy(ics).to(self.dtype).unsqueeze(dim=1)
        inputs = torch.from_numpy(inputs).to(self.dtype)
        with torch.no_grad():
            output, latents = self.model(inputs=inputs, hidden=ics)
            output = output.numpy()[:, :, 0, :]
            latents = latents.numpy()[:, :, 0, :]
        return latents, output, None


class NBFFEnvironment(Environment):
    def __init__(self, env: NBitFlipFlop, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self.env = env  # currently no way to seed env it seems

    def sample_inputs(self, n: int):
        inputs, outputs = self.env.generate_dataset(n_samples=n)
        trial_info = pd.DataFrame([], index=np.arange(n))  # no trial info
        other = {"ff_state": outputs}
        return trial_info, inputs, other


# make objects

seed = 0

system = UncoupledSystem(
    model=NODEModel(model=model), env=NBFFEnvironment(env=task_env)
)

data_sampler = LinearNonlinearPoisson(
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
)

datagen = NeuralDataGenerator(
    system=system,
    data_sampler=data_sampler,
    seed=seed,
)


# run simulation

trajectory_kwargs = dict(
    n_traj=1000,
    ic_kwargs=dict(
        dist="zeros",
    ),
    simulation_kwargs=dict(),
)

export_kwargs = dict(
    file_format="hdf5",
    file_path="tbff_node_offset-3.h5",
    # file_format="nwb",
    # file_path="test.nwb",
    # overwrite=True,
    # dt=0.005,
    # inter_trial_interval=20,
)

output = datagen.generate_dataset(
    trajectory_kwargs=trajectory_kwargs,
    export_kwargs=export_kwargs,
)

import pdb

pdb.set_trace()
