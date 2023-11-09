import sys

sys.path.insert(
    1, "/home/fpei2/interp/InterpretabilityBenchmark/MotorNet-pytorch/MotorNet-pytorch/"
)
sys.path.insert(1, "/home/fpei2/interp/InterpretabilityBenchmark/")
# sys.path.insert(1, '/home/csverst/Github/InterpretabilityBenchmark/')

from interpretability.task_modeling.task_wrapper.coupled import TaskTrainedCoupled
from interpretability.task_modeling.model.policy_networks import NODEPolicy
from motornet.environment import RandomTargetReach
from motornet.effector import ReluPointMass24

import numpy as np
import pandas as pd
import torch
import copy
from typing import Optional, Any

from synergen.core import NeuralDataGenerator
from synergen.systems.base import CoupledSystem
from synergen.systems.models.base import Model
from synergen.systems.envs.gym import GymEnvironment
from synergen.embedding.pca import PCA
from synergen.neural_data.lin_nonlin import LinearNonlinearPoisson


# load stuff from the run

ckpt_path = "/snel/share/runs/dysts-learning/MultiDatasets/NODE/20230725_RandomTarget_NODE4/train_2023-07-25_15-07-38/0_seed=0,learning_rate=0.0010/last.ckpt"

task_env = RandomTargetReach(
    effector=ReluPointMass24(),
    max_ep_duration=1.0,
    differentiable=False,  # so it returns numpy
)

model = NODEPolicy(
    input_size=None,
    latent_size=20,
    output_size=None,
    num_layers=3,
    layer_hidden_size=128,
)
n_outputs = task_env.action_space.shape[0]
n_inputs = task_env.observation_space.shape[0]
model.init_model(n_inputs, n_outputs)

pl_module = TaskTrainedCoupled(
    task_env=task_env,
    input_size=12,
    output_size=4,
    state_label="fingertip",
    learning_rate=1.0e-3,
    weight_decay=0,
    model=model,
)
pl_module.load_state_dict(torch.load(ckpt_path)["state_dict"])
model = pl_module.model


# define necessary classes


class NODEModel(Model):
    def __init__(self, model: NODEPolicy, seed: Optional[int] = None):
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
        ics = torch.from_numpy(ics).to(self.dtype)
        inputs = torch.from_numpy(inputs).to(self.dtype)
        with torch.no_grad():
            output, latents = self.model(x=inputs, h0=ics)
            output = output.numpy()
            latents = latents.numpy()
        return latents, output, output


class RandomTargetEnvironment(GymEnvironment):
    def make_batch_envs(self, batch_size):
        self.batch_envs = self.env  # env already supports batching

    def reset_envs(self, trial_info):
        joints = np.stack(trial_info["joints"], axis=0)
        goal = np.stack(trial_info["goal"], axis=0)
        obs, info = self.batch_envs.reset(
            batch_size=len(trial_info),
            joint_state=torch.from_numpy(joints),
            goal=torch.from_numpy(goal),
        )
        return obs, info

    def parse_info(self, obs, reward, term, trunc, env_infos, state_label="fingertip"):
        other = {"state": env_infos["states"][state_label]}
        return {}, other

    def check_done(self, obs, reward, term, trunc, env_infos):
        to_bool_array = (
            lambda x: np.full((obs.shape[0],), x, dtype=np.bool_)
            if isinstance(x, bool)
            else x
        )
        return super().check_done(
            obs, reward, to_bool_array(term), to_bool_array(trunc), env_infos
        )


# make objects

seed = 0

env = RandomTargetEnvironment(
    env=task_env, max_batch_size=8, info_kwargs={"state_label": "fingertip"}
)

system = CoupledSystem(
    model=NODEModel(model=model),
    env=env,
)

data_sampler = LinearNonlinearPoisson(
    input_dim=20,
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


# manually make trial info for convenience

# task_env_cp = copy.deepcopy(task_env)
# n_traj = 1000
# joint_list = []
# goal_list = []
# for _ in range(n_traj):
#     obs, info = task_env_cp.reset(batch_size=1)
#     joint_list.append(np.squeeze(info["states"]["joint"]))
#     goal_list.append(np.squeeze(info["goal"]))
# trial_info = pd.DataFrame({"joints": joint_list, "goal": goal_list})


# run simulation

trajectory_kwargs = dict(
    n_traj=1000,
    ic_kwargs=dict(
        dist="zeros",
    ),
    simulation_kwargs=dict(),
    # trial_kwargs=dict(
    #     trial_info=trial_info,
    # ),
)

export_kwargs = dict(
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
