import sys

sys.path.insert(1, "/home/fpei2/learning/ttrnn/")

import numpy as np
from typing import Optional, Any

from synergen.core import NeuralDataGenerator
from synergen.systems.base import UncoupledSystem, CoupledSystem
from synergen.systems.models.base import Model
from synergen.systems.envs.gym import GymEnvironment
from synergen.embedding.pca import PCA
from synergen.synthetic_data.lin_nonlin import LinearNonlinearPoisson
from synergen.utils.data_io import read_file

import torch

from ttrnn.trainer import A2C
from ttrnn.models.actorcritic import ActorCritic
from ttrnn.tasks.harlow import HarlowMinimalDelay
from neurogym.wrappers import PassAction, PassReward, Noise
from ttrnn.tasks.wrappers import DiscreteToBoxWrapper, RingToBoxWrapper, ParallelEnvs

seed = 0

ckpt_path = "/home/fpei2/learning/harlow-rnn-analysis/runs/harlowdelay6_gru256/epoch=29999-step=30000.ckpt"

task = HarlowMinimalDelay(
    dt=100,
    obj_dim=11,
    obj_mode="kb",
    obj_init="normal",
    orthogonalize=True,
    abort=True,
    rewards={"abort": -0.1, "correct": 1.0, "fail": 0.0},
    timing={"fixation": 200, "stimulus": 400, "delay": 200, "decision": 200},
    num_trials_before_reset=6,
    r_tmax=-1.0,
)

std_noise = 0.1
wrappers = [
    (Noise, {"std_noise": std_noise}),
    (PassAction, {"one_hot": True}),
    (PassReward, {}),
    # (ParallelEnvs, {'num_envs': 8}),
]

if len(wrappers) > 0:
    for wrapper, wrapper_kwargs in wrappers:
        task = wrapper(task, **wrapper_kwargs)

pl_module = A2C.load_from_checkpoint(ckpt_path, env=task)

model = pl_module.model

env = GymEnvironment(
    env=task,
    max_batch_size=8,
    info_kwargs={"info_fields": ["performance"], "other_fields": ["gt", "reward"]},
    done_kwargs={"fields": ["block_done"]},
    legacy=True,
)


class A2CModel(Model):
    def __init__(self, model: ActorCritic, seed: Optional[int] = None):
        super().__init__(
            n_dim=model.rnn.hidden_size,
            seed=seed,
        )
        self.model = model
        self.model.rnn.update_cache()

    def simulate(
        self,
        ics: np.ndarray,
        inputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        with torch.no_grad():
            action_logits, value, hx = self.model(
                torch.from_numpy(inputs).float(),
                torch.from_numpy(ics).float(),
                cached=True,
            )
            hx = hx.numpy()
            action = action_logits.mode.numpy()
            action_logits = action_logits.logits.numpy()
        return hx, action_logits, action


system = CoupledSystem(
    model=A2CModel(model=model),
    env=env,
    seed=seed,
)

embedding = PCA(n_components=10, seed=seed)
data_sampler = LinearNonlinearPoisson(
    input_dim=10,
    output_dim=30,
    proj_weight_dist="uniform",
    proj_weight_params=dict(
        low=-1.0,
        high=1.0,
    ),
    nonlinearity="exp",
    nonlinearity_params=dict(),
    sort_dims=True,
    standardize=True,
    clip_val=5,
    seed=seed,
)

datagen = NeuralDataGenerator(
    system=system,
    data_sampler=data_sampler,
    embedding=embedding,
    seed=seed,
)

trajectory_kwargs = dict(
    n_traj=500,
    ic_kwargs=dict(
        dist="normal",
        dist_params=dict(
            loc=0.0,
            scale=0.1,
        ),
    ),
    trial_kwargs=dict(),
    simulation_kwargs=dict(),
    max_steps=60,
)

export_kwargs = dict(
    # file_format="nwb",
    # file_path="test.nwb",
    # overwrite=True,
)

output = datagen.generate_dataset(
    trajectory_kwargs=trajectory_kwargs,
    export_kwargs=export_kwargs,
)

import pdb

pdb.set_trace()
