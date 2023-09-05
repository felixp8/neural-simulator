import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Any, Union

from synergen.core import NeuralDataGenerator
from synergen.systems.base import UncoupledSystem
from synergen.systems.models.base import Model
from synergen.systems.envs.neurogym import NeurogymEnvironment
from synergen.embedding.pca import PCA
from synergen.synthetic_data.lin_nonlin import LinearNonlinearPoisson

import neurogym as ngym

seed = 0

# specify task
task = "PerceptualDecisionMaking-v0"
kwargs = {
    "dt": 20,
    "timing": {"fixation": 200, "stimulus": 400, "delay": 200, "response": 200},
}
seq_len = 50

env = ngym.make(task, **kwargs)
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

# define network
class Net(nn.Module):
    def __init__(self, num_h, ob_size, act_size):
        super(Net, self).__init__()
        self.rnn = nn.GRU(ob_size, num_h)
        self.linear = nn.Linear(num_h, act_size)

    def forward(self, x, h0=None):
        out = self.rnn(x, h0)[0]
        x = self.linear(out)
        return x, out


# define model class
class TorchModel(Model):
    def __init__(self, model: Net, seed=None):
        super().__init__(
            n_dim=model.rnn.hidden_size,
            seed=seed,
        )
        self.model = model
        self.dtype = list(model.parameters())[0].dtype

    def simulate(
        self,
        ics: np.ndarray,
        inputs: np.ndarray,
    ):
        with torch.no_grad():
            ics = torch.from_numpy(ics).to(self.dtype)
            inputs = torch.from_numpy(inputs).to(self.dtype)
            if self.model.rnn.batch_first:
                ics = ics.unsqueeze(dim=1)
            else:
                inputs = inputs.permute(1, 0, 2)
                ics = ics.unsqueeze(dim=0)
            outputs, hidden = self.model(inputs, ics)
            if not self.model.rnn.batch_first:
                outputs = outputs.permute(1, 0, 2)
                hidden = hidden.permute(1, 0, 2)
            outputs = outputs.numpy()
            hidden = hidden.numpy()
        return hidden, outputs, None


# load model from checkpoint
net = Net(num_h=64, ob_size=ob_size, act_size=act_size)
state_dict = torch.load("../../scripts/model.ckpt")["state_dict"]
net.load_state_dict(state_dict)
net.eval()

# instantiate objects
model = TorchModel(model=net, seed=seed)
env = NeurogymEnvironment(env=env, seed=seed)
system = UncoupledSystem(model=model, env=env, seed=seed)

data_sampler = LinearNonlinearPoisson(
    input_dim=64,
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

# kwargs
trajectory_kwargs = dict(
    n_traj=1000,
    ic_kwargs=dict(
        dist="zeros",
    ),
    simulation_kwargs=dict(),
)

export_kwargs = dict(
    file_format="nwb",
    file_path="test.nwb",
    overwrite=True,
    dt=0.020,
    inter_trial_interval=10,
)

# generate data
output = datagen.generate_dataset(
    trajectory_kwargs=trajectory_kwargs,
    export_kwargs=export_kwargs,
)

import pdb

pdb.set_trace()
