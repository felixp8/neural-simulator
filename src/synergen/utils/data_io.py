import numpy as np
import pandas as pd
import h5py
import zipfile
from numpy.lib.recfunctions import repack_fields
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Optional, Union, Literal

from .types import TrajectoryBatch

# general


def write_file(
    file_path: Union[Path, str],
    trajectory_batch: TrajectoryBatch,
    file_format: Literal["npz", "hdf5", "nwb", "benchmark"],
    generator=None,
    overwrite: bool = False,
    **kwargs,
):
    if file_format == "hdf5":
        return write_to_hdf5(
            file_path=file_path,
            trajectory_batch=trajectory_batch,
            generator=generator,
            overwrite=overwrite,
            **kwargs,
        )
    if file_format == "npz":
        return write_to_npz(
            file_path=file_path,
            trajectory_batch=trajectory_batch,
            generator=generator,
            overwrite=overwrite,
            **kwargs,
        )
    if file_format == "nwb":
        return write_to_nwb(
            file_path=file_path,
            trajectory_batch=trajectory_batch,
            generator=generator,
            overwrite=overwrite,
            **kwargs,
        )
    if file_format == "benchmark":
        return write_to_benchmark(
            file_path=file_path,
            trajectory_batch=trajectory_batch,
            generator=generator,
            overwrite=overwrite,
            **kwargs,
        )


def read_file(
    file_path: Union[Path, str],
    read_slice: Union[list[int], np.ndarray[int], slice] = (),
):
    file_path = Path(file_path)
    assert file_path.suffix in [".h5", ".npz", ".nwb"]
    if file_path.suffix == ".h5":
        return read_from_hdf5(file_path=file_path, read_slice=read_slice)
    if file_path.suffix == ".npz":
        return read_from_npz(file_path=file_path, read_slice=read_slice)
    # if file_path.suffix == ".nwb":
    #     return read_from_nwb(file_path=file_path, read_slice=read_slice,)


def check_data_shape(
    file_path: Union[Path, str],
):
    file_path = Path(file_path)
    assert file_path.suffix in [".h5", ".npz", ".nwb"]
    if file_path.suffix == ".h5":
        return check_data_shape_hdf5(
            file_path=file_path,
        )
    if file_path.suffix == ".npz":
        return check_data_shape_npz(
            file_path=file_path,
        )


# hdf5


def write_to_hdf5(
    file_path: Union[Path, str],
    trajectory_batch: TrajectoryBatch,
    generator=None,
    include: Optional[list] = None,
    overwrite: bool = False,
):
    file_path = Path(file_path)
    assert (
        overwrite or not file_path.exists()
    ), f"Path {file_path} already exists. Please set `overwrite=True` if you wish to overwrite it."
    include = include or flatten_dict_keys(trajectory_batch.__dict__)
    with h5py.File(file_path, "w") as h5f:
        if "trajectories" in include:
            h5f.create_dataset(name="trajectories", data=trajectory_batch.trajectories)
        if "inputs" in include and trajectory_batch.inputs is not None:
            h5f.create_dataset(name="inputs", data=trajectory_batch.inputs)
        if "outputs" in include and trajectory_batch.outputs is not None:
            h5f.create_dataset(name="outputs", data=trajectory_batch.outputs)
        if trajectory_batch.other is not None:
            for key, val in trajectory_batch.other.items():
                if f"other.{key}" in include:
                    other = get_group(h5obj=h5f, group_name="other")
                    other.create_dataset(name=key, data=val)
        if trajectory_batch.neural_data is not None:
            for key, val in trajectory_batch.neural_data.items():
                if f"neural_data.{key}" in include:
                    neural_data = get_group(h5obj=h5f, group_name="neural_data")
                    neural_data.create_dataset(name=key, data=val)
        if "trial_info" in include and trajectory_batch.trial_info is not None:
            ti_as_array = df_to_sarray(trajectory_batch.trial_info)
            if ti_as_array is not None:
                drop_names = []
                for name in ti_as_array.dtype.names:
                    if ti_as_array.dtype[name] == np.dtype("O"):
                        drop_names.append(name)
                if len(drop_names) > 0:
                    print(
                        f"Excluding columns {drop_names} from H5 file as they are unsupported dtypes"
                    )
                keep_names = list(set(ti_as_array.dtype.names) - set(drop_names))
                ti_as_array = repack_fields(ti_as_array[keep_names])
                h5f.create_dataset(name="trial_info", data=ti_as_array)


def read_from_hdf5(
    file_path: Union[Path, str],
    read_slice: Union[list[int], np.ndarray[int], slice] = (),
):
    file_path = Path(file_path)
    assert file_path.exists(), f"File {file_path} not found"
    with h5py.File(file_path, "r") as h5f:
        trajectories = h5f["trajectories"][read_slice]
        inputs = h5f["inputs"][read_slice] if "inputs" in h5f.keys() else None
        outputs = h5f["outputs"][read_slice] if "outputs" in h5f.keys() else None
        other = (
            h5_to_dict(h5f["other"], read_slice=read_slice)
            if "other" in h5f.keys()
            else None
        )
        neural_data = (
            h5_to_dict(h5f["neural_data"], read_slice=read_slice)
            if "neural_data" in h5f.keys()
            else None
        )
        trial_info = (
            sarray_to_df(h5f["trial_info"][read_slice])
            if "trial_info" in h5f.keys()
            else None
        )
    trajectory_batch = TrajectoryBatch(
        trajectories=trajectories,
        trial_info=trial_info,
        inputs=inputs,
        outputs=outputs,
        other=other,
        neural_data=neural_data,
    )
    return trajectory_batch


def check_data_shape_hdf5(
    file_path: Union[Path, str],
):
    file_path = Path(file_path)
    assert file_path.exists(), f"File {file_path} not found"
    with h5py.File(file_path, "r") as h5f:
        b, t, _ = h5f["trajectories"].shape
    return b, t


# numpy


def write_to_npz(
    file_path: Union[Path, str],
    trajectory_batch: TrajectoryBatch,
    generator=None,
    # TODO: support `include`
    overwrite: bool = False,
):
    file_path = Path(file_path)
    assert (
        overwrite or not file_path.exists()
    ), f"Path {file_path} already exists. Please set `overwrite=True` if you wish to overwrite it."
    data_dict = {"trajectories": trajectory_batch.trajectories}
    if trajectory_batch.inputs is not None:
        data_dict["inputs"] = trajectory_batch.inputs
    if trajectory_batch.outputs is not None:
        data_dict["outputs"] = trajectory_batch.outputs
    if trajectory_batch.other is not None:
        for key, val in trajectory_batch.other.items():
            data_dict[f"other_{key}"] = val
    if trajectory_batch.neural_data is not None:
        for key, val in trajectory_batch.neural_data.items():
            data_dict[f"neural_data_{key}"] = val
    if trajectory_batch.trial_info is not None:
        ti_as_array = df_to_sarray(trajectory_batch.trial_info)
        if ti_as_array is not None:
            data_dict["trial_info"] = ti_as_array
    np.savez(file_path, **data_dict)


def read_from_npz(
    file_path: Union[Path, str],
    read_slice: Union[list[int], np.ndarray[int], slice] = (),
):
    file_path = Path(file_path)
    assert file_path.exists(), f"File {file_path} not found"
    if read_slice != ():
        print(
            "warning: specifying read slices when reading from npz does not save memory, as "
            "all the data are loaded when the file is read anyway"
        )
    npzfile = np.load(file_path)
    data_dict = {}
    for filename in npzfile.files:
        if filename == "trial_info":
            data_dict["trial_info"] = sarray_to_df(npzfile[filename]).iloc[read_slice]
        elif "other" in filename:
            if "other" not in data_dict:
                data_dict["other"] = {}
            key = filename.partition("other_")[-1]
            data_dict["other"][key] = npzfile[filename][read_slice]
        elif "neural_data" in filename:
            if "neural_data" not in data_dict:
                data_dict["neural_data"] = {}
            key = filename.partition("neural_data_")[-1]
            data_dict["neural_data"][key] = npzfile[filename][read_slice]
        else:
            data_dict[filename] = npzfile[filename][read_slice]
    trajectory_batch = TrajectoryBatch(**data_dict)
    return trajectory_batch


def check_data_shape_npz(
    file_path: Union[Path, str],
):
    file_path = Path(file_path)
    assert file_path.exists(), f"File {file_path} not found"
    with zipfile.ZipFile(file_path) as archive:
        npy = archive.open("trajectories.npy")
        version = np.lib.format.read_magic(npy)
        shape, *_ = np.lib.format._read_array_header(npy, version)
    b, t, *_ = shape
    return b, t


# nwb


def write_to_nwb(
    file_path: Union[Path, str],
    trajectory_batch: TrajectoryBatch,
    generator=None,
    # TODO: support `include`
    dt: float = 0.01,
    inter_trial_interval: int = 10,
    overwrite: bool = False,
):
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.file import Subject
    from pynwb.device import Device
    from pynwb.misc import Units
    from pynwb.ecephys import ElectrodeGroup
    from hdmf.backends.hdf5 import H5DataIO
    from datetime import datetime, timezone
    from uuid import uuid4

    file_path = Path(file_path)
    assert (
        overwrite or not file_path.exists()
    ), f"Path {file_path} already exists. Please set `overwrite=True` if you wish to overwrite it."

    # TODO: support metadata
    nwbfile = NWBFile(
        session_description="no description.",
        session_start_time=datetime.now(timezone.utc).astimezone(),
        identifier=str(uuid4()),
    )

    # TODO: support masked arrays
    trial_count, trial_length, _ = trajectory_batch.trajectories.shape
    trial_start_idxs = np.arange(trial_count) * (trial_length + inter_trial_interval)
    timestamps = np.concatenate(
        [(np.arange(trial_length) + start_idx) for start_idx in trial_start_idxs],
        axis=0,
    )
    timestamps = np.round(timestamps * dt, 9)
    flatten = lambda arr: arr.reshape(-1, arr.shape[-1])

    if trajectory_batch.trial_info is not None:
        for col in trajectory_batch.trial_info.columns:
            if col in ["start_time", "stop_time"]:
                raise AssertionError  # TODO: have a workaround
            nwbfile.add_trial_column(name=col, description=col)

    for i in range(trial_count):
        trial_data = {
            "start_time": round(trial_start_idxs[i] * dt, 9),
            "stop_time": round((trial_start_idxs[i] + trial_length) * dt, 9),
        }
        if trajectory_batch.trial_info is not None:
            trial_info_dict = trajectory_batch.trial_info.iloc[i].to_dict()
            for key in trial_info_dict.keys():
                if key.endswith("_time"):
                    trial_info_dict[key] = (
                        trial_info_dict[key] + trial_data["start_time"]
                    )
            trial_data.update(trial_info_dict)
        nwbfile.add_trial(**trial_data)

    simulation_data = []
    for field in ["trajectories", "inputs", "outputs"]:
        if trajectory_batch.__dict__.get(field, None) is not None:
            ts = TimeSeries(
                name=field,
                data=H5DataIO(
                    flatten(trajectory_batch.__dict__[field]),
                    compression="gzip",
                ),
                timestamps=timestamps,
                description=f"Simulated {field} data.",
                unit="n/a",
            )
            simulation_data.append(ts)
    if trajectory_batch.other is not None:
        for field in trajectory_batch.other.keys():
            ts = TimeSeries(
                name=field,
                data=H5DataIO(
                    flatten(trajectory_batch.other[field]),
                    compression="gzip",
                ),
                timestamps=timestamps,
                description=f"Simulated {field} data.",
                unit="n/a",
            )
            simulation_data.append(ts)
    simulation_module = nwbfile.create_processing_module(
        name="simulation", description="Simulation data."
    )
    simulation_module.add(simulation_data)

    neural_data = []
    if trajectory_batch.neural_data is not None:
        for field in trajectory_batch.neural_data.keys():
            if field == "spikes":
                spikes = trajectory_batch.neural_data["spikes"].astype(int)
                spikes = flatten(spikes)
                n_channels = spikes.shape[-1]
                device = nwbfile.create_device(
                    name="Device",
                    description=f"no description.",
                )
                electrode_group = nwbfile.create_electrode_group(
                    name="ElectrodeGroup",
                    description="no description.",
                    location="unknown",
                    device=device,
                )
                for i in range(n_channels):
                    nwbfile.add_electrode(
                        id=int(i),
                        x=np.nan,
                        y=np.nan,
                        z=np.nan,
                        imp=np.nan,
                        location="unknown",
                        group=electrode_group,
                        filtering="unknown",
                    )
                    indices = np.nonzero(spikes[:, i])[0]
                    counts = spikes[:, i][indices]
                    spike_times = np.repeat(timestamps[indices], counts)
                    nwbfile.add_unit(
                        id=i,
                        spike_times=spike_times,
                        electrodes=[i],
                    )
            else:
                ts = TimeSeries(
                    name=field,
                    data=H5DataIO(
                        flatten(trajectory_batch.neural_data[field]),
                        compression="gzip",
                    ),
                    timestamps=timestamps,
                    description=f"Simulated neural {field} data.",
                    unit="n/a",
                )
                neural_data.append(ts)
    if len(neural_data) > 0:
        neural_module = nwbfile.create_processing_module(
            name="neural_data", description="Synthetic neural data."
        )
        neural_module.add(neural_data)

    with NWBHDF5IO(file_path, "w") as io:
        io.write(nwbfile)


# TODO: def read_from_nwb


# benchmark


def write_to_benchmark(
    file_path: Union[Path, str],
    trajectory_batch: TrajectoryBatch,
    generator=None,
    overwrite: bool = False,
    trial_split_ratio: Union[list, tuple] = (0.8, 0.2),
    neuron_split_ratio: Union[list, tuple] = (1.0, 0.0),
    seed: Optional[int] = None,
):
    file_path = Path(file_path)
    assert (
        overwrite or not file_path.exists()
    ), f"Path {file_path} already exists. Please set `overwrite=True` if you wish to overwrite it."

    assert (
        len(trial_split_ratio) == 2
    ), f"Only supports 2 trial splits currently (train/valid) but received {len(trial_split_ratio)} values."
    trial_split_ratio = np.array(trial_split_ratio)
    trial_split_ratio /= trial_split_ratio.sum()
    assert trial_split_ratio[-1] > 0, f"Cannot have valid split ratio <= 0.0"

    assert (
        len(neuron_split_ratio) == 2
    ), f"Only supports 2 neuron splits currently (heldin/out) but received {len(neuron_split_ratio)} values."
    neuron_split_ratio = np.array(neuron_split_ratio)
    neuron_split_ratio /= neuron_split_ratio.sum()

    assert trajectory_batch.neural_data is not None
    assert (
        "spikes" in trajectory_batch.neural_data
    )  # TODO: allow mapping what fields to go recon data, etc.
    assert "rates" in trajectory_batch.neural_data

    trial_inds = np.arange(len(trajectory_batch))
    train_trial_inds, valid_trial_inds = train_test_split(
        trial_inds, test_size=trial_split_ratio[-1], random_state=seed
    )

    neuron_inds = np.arange(trajectory_batch.neural_data["spikes"].shape[-1])
    if neuron_split_ratio[-1] == 0.0:
        heldin_neuron_inds = all_neuron_inds = neuron_inds
    else:
        heldin_neuron_inds, heldout_neuron_inds = train_test_split(
            neuron_inds, test_size=neuron_split_ratio[-1], random_state=seed
        )
        all_neuron_inds = np.concatenate([heldin_neuron_inds, heldout_neuron_inds])

    with h5py.File(file_path, "w") as h5f:
        h5f.create_dataset(
            "train_encod_data",
            data=trajectory_batch.neural_data["spikes"][train_trial_inds][
                :, :, heldin_neuron_inds
            ],
        )
        h5f.create_dataset(
            "valid_encod_data",
            data=trajectory_batch.neural_data["spikes"][valid_trial_inds][
                :, :, heldin_neuron_inds
            ],
        )

        h5f.create_dataset(
            "train_recon_data",
            data=trajectory_batch.neural_data["spikes"][train_trial_inds][
                :, :, all_neuron_inds
            ],
        )
        h5f.create_dataset(
            "valid_recon_data",
            data=trajectory_batch.neural_data["spikes"][valid_trial_inds][
                :, :, all_neuron_inds
            ],
        )

        h5f.create_dataset(
            "train_activity",
            data=trajectory_batch.neural_data["rates"][train_trial_inds][
                :, :, all_neuron_inds
            ],
        )
        h5f.create_dataset(
            "valid_activity",
            data=trajectory_batch.neural_data["rates"][valid_trial_inds][
                :, :, all_neuron_inds
            ],
        )

        h5f.create_dataset(
            "train_latents", data=trajectory_batch.trajectories[train_trial_inds]
        )
        h5f.create_dataset(
            "valid_latents", data=trajectory_batch.trajectories[valid_trial_inds]
        )

        if trajectory_batch.inputs is not None:
            h5f.create_dataset(
                "train_inputs", data=trajectory_batch.inputs[train_trial_inds]
            )
            h5f.create_dataset(
                "valid_inputs", data=trajectory_batch.inputs[valid_trial_inds]
            )

        h5f.create_dataset("train_inds", data=train_trial_inds)
        h5f.create_dataset("valid_inds", data=valid_trial_inds)

        h5f.create_dataset("perm_neurons", data=all_neuron_inds)
        if generator is not None:
            sampler_params = generator.data_sampler.get_params()
            for key, val in sampler_params.items():
                h5f.create_dataset(
                    (key if key != "proj_weights" else "readout"), data=val
                )


# utils


def df_to_sarray(
    df: pd.DataFrame,
    index: bool = False,
):
    """Convert pandas DataFrame to structured array"""
    if df.empty:
        return None
    if index:
        df = df.reset_index(name="trial_id")
    dtypes = list(df.dtypes.items())
    for i, (col_name, col_dtype) in enumerate(dtypes):
        if col_dtype == np.dtype("O"):
            if isinstance(df[col_name].iloc[0], str):
                max_str_len = max([len(entry) for entry in df[col_name].values])
                dtypes[i] = (col_name, f"S{max_str_len}")
            elif isinstance(df[col_name].iloc[0], (list, tuple, np.ndarray)):
                fixed_len = all(
                    [len(df[col_name].iloc[0]) == len(row) for row in df[col_name]]
                )
                if fixed_len:
                    subdtype = np.dtype(type(df[col_name].iloc[0][0]))
                    if subdtype.kind == "U":
                        max_str_len = max(
                            [max([len(entry) for entry in row]) for row in df[col_name]]
                        )
                        subdtype = np.dtype(f"S{max_str_len}")
                    dtypes[i] = (col_name, subdtype, len(df[col_name].iloc[0]))
            # no handling for other dtypes here
    return np.array([tuple(x) for x in df.values], dtype=dtypes)


def sarray_to_df(
    sarray: np.array,
):
    normal_cols = []
    array_cols = []
    has_index = False
    for name in sarray.dtype.names:
        if sarray.dtype[name].shape != ():
            array_cols.append(name)
        elif name == "trial_id":
            has_index = True
        else:
            normal_cols.append(name)
    index = sarray["trial_id"] if has_index else None
    df = pd.DataFrame.from_records(sarray[normal_cols], index=index)
    array_df = pd.DataFrame(
        {name: list(sarray[name]) for name in array_cols}, index=df.index
    )
    df = pd.concat(
        [df, array_df], axis=1
    )  # column order is switched up, if that matters
    return df


def h5_to_dict(
    h5obj: Union[h5py.File, h5py.Group],
    read_slice: Union[list[int], np.ndarray[int], slice, tuple[slice]] = (),
):
    h5_dict = {
        key: h5_to_dict(h5obj[key])
        if isinstance(h5obj[key], h5py.Group)
        else h5obj[key][read_slice]
        for key in h5obj.keys()
    }
    return h5_dict


def flatten_dict_keys(d, prefix=""):
    keys = []
    for key in d.keys():
        if isinstance(d[key], dict):
            keys += flatten_dict_keys(d[key], prefix=f"{prefix}{key}.")
        else:
            keys.append(f"{prefix}{key}")
    return keys


def get_group(h5obj: h5py.Group, group_name: str, **kwargs):
    if group_name in h5obj.keys():
        assert isinstance(
            h5obj[group_name], h5py.Group
        ), f"{group_name} exists but is not a `h5py.Group`"
        return h5obj[group_name]
    else:
        return h5obj.create_group(group_name, **kwargs)
