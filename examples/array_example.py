import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Optional, Any

from synergen.core import NeuralDataGenerator
from synergen.systems.array import ArraySystem
from synergen.embedding.pca import PCA
from synergen.neural_data.lin_nonlin import LinearNonlinearPoisson

seed = 0

# load data from file

# data_dir = Path("/home/csverst/Documents/tempData/")
# file_path = Path('3BFF_model_RNN_n_neurons_50_nonlin_embed_False_obs_noise_poisson_seed_0.h5')

# with h5py.File(data_dir / file_path, 'r') as h5f:
#     train_inputs = h5f['train_inputs'][()]
#     train_latents = h5f['train_latents'][()]
#     valid_inputs = h5f['valid_inputs'][()]
#     valid_latents = h5f['valid_latents'][()]
# inputs = np.concatenate([train_inputs, valid_inputs], axis=0)
# latents = np.concatenate([train_latents, valid_latents], axis=0)

# system = ArraySystem(
#     trajectories=latents,
#     inputs=inputs,
#     sample_method="first",
#     seed=seed,
# )

data_dir = Path("/home/fpei2/interp/scripts/")
file_path = Path("gru_tbff_20230917-193143.h5")

with h5py.File(data_dir / file_path, "r") as h5f:
    latents = h5f["latents"][()]
    inputs = h5f["inputs"][()]
    outputs = h5f["outputs"][()]
    targets = h5f["targets"][()]
    epoch_nums = h5f["epoch_nums"][()]
    trial_info = pd.DataFrame(epoch_nums[:, None], columns=["epoch_num"])

system = ArraySystem(
    states=latents,
    trial_info=trial_info,
    inputs=inputs,
    outputs=outputs,
    temporal_data={"targets": targets},
    sample_method="first",
    seed=seed,
)

# make other objects

data_sampler = LinearNonlinearPoisson(
    output_dim=50,
    proj_weight_dist="uniform",
    proj_weight_params=dict(
        low=-1.0,
        high=1.0,
    ),
    # proj_weight_dist="eye",
    normalize_proj=False,
    nonlinearity="exp",
    nonlinearity_params=dict(
        offset=0.0,
    ),
    # sort_dims=True,
    mean_center="all",
    rescale_variance="all",
    target_variance=1.0,
    # clip_val=5,
)

datagen = NeuralDataGenerator(
    system=system,
    data_sampler=data_sampler,
    seed=seed,
)


# run simulation

trajectory_kwargs = dict(
    n_traj=1000,
)

export_kwargs = dict(
    # file_format="lfads",
    # file_path="temp.h5",
    # overwrite=True,
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

from sklearn.decomposition import PCA

trajectories_all = output.states.reshape(-1, output.states.shape[-1]).copy()
trajectories_all -= trajectories_all.mean(axis=0, keepdims=True)
pca1 = PCA()
pca1.fit(trajectories_all)
orig_participation_ratio = np.square(np.sum(pca1.explained_variance_)) / np.sum(
    np.square(pca1.explained_variance_)
)
print(orig_participation_ratio)

rates_all = (
    output.neural_data["rates"]
    .reshape(-1, output.neural_data["rates"].shape[-1])
    .copy()
)
rates_all -= rates_all.mean(axis=0, keepdims=True)
pca2 = PCA()
pca2.fit(rates_all)
rate_participation_ratio = np.square(np.sum(pca2.explained_variance_)) / np.sum(
    np.square(pca2.explained_variance_)
)
print(rate_participation_ratio)

logrates_all = np.log(
    output.neural_data["rates"].reshape(-1, output.neural_data["rates"].shape[-1])
)
logrates_all -= logrates_all.mean(axis=0, keepdims=True)
pca3 = PCA()
pca3.fit(logrates_all)
lograte_participation_ratio = np.square(np.sum(pca3.explained_variance_)) / np.sum(
    np.square(pca3.explained_variance_)
)
print(lograte_participation_ratio)

import pdb

pdb.set_trace()
