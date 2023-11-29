import numpy as np
import pandas as pd
import h5py
import zipfile
from numpy.lib.recfunctions import repack_fields
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Optional, Union, Literal

from .types import DataBatch

# general


def write_file(
    file_path: Union[Path, str],
    data_batch: DataBatch,
    file_format: Literal["npz", "hdf5", "nwb", "lfads"],
    overwrite: bool = False,
    **kwargs,
):
    if file_format == "hdf5":
        return write_to_hdf5(
            file_path=file_path,
            data_batch=data_batch,
            overwrite=overwrite,
            **kwargs,
        )
    if file_format == "npz":
        return write_to_npz(
            file_path=file_path,
            data_batch=data_batch,
            overwrite=overwrite,
            **kwargs,
        )
    if file_format == "nwb":
        print("Please do not do this. Exporting to NWB is a bad idea (right now).")
        return write_to_nwb(
            file_path=file_path,
            data_batch=data_batch,
            overwrite=overwrite,
            **kwargs,
        )
    if file_format == "lfads":
        return write_to_lfads(
            file_path=file_path,
            data_batch=data_batch,
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
    data_batch: DataBatch,
    include: Optional[list] = None,
    trial_info_as_csv: bool = True,
    overwrite: bool = False,
):
    file_path = Path(file_path)
    assert (
        overwrite or not file_path.exists()
    ), f"Path {file_path} already exists. Please set `overwrite=True` if you wish to overwrite it."
    include = include or flatten_dict_keys(data_batch.__dict__)
    with h5py.File(file_path, "w") as h5f:
        if "states" in include:
            h5f.create_dataset(
                name="states", data=data_batch.states, compression="gzip"
            )
        if "inputs" in include and data_batch.inputs is not None:
            h5f.create_dataset(
                name="inputs", data=data_batch.inputs, compression="gzip"
            )
        if "outputs" in include and data_batch.outputs is not None:
            h5f.create_dataset(
                name="outputs", data=data_batch.outputs, compression="gzip"
            )
        if "trial_info" in include and data_batch.trial_info is not None:
            if trial_info_as_csv:
                data_batch.trial_info.to_csv(
                    file_path.parent / (file_path.stem + "_trial_info.csv"), index=False
                )
            else:
                ti_as_array = df_to_sarray(data_batch.trial_info)
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
                    h5f.create_dataset(
                        name="trial_info", data=ti_as_array, compression="gzip"
                    )
        for key, val in data_batch.temporal_data.items():
            if f"temporal_data.{key}" in include:
                temporal_data = get_group(h5obj=h5f, group_name="temporal_data")
                temporal_data.create_dataset(name=key, data=val, compression="gzip")
        for key, val in data_batch.neural_data.items():
            if f"neural_data.{key}" in include:
                neural_data = get_group(h5obj=h5f, group_name="neural_data")
                neural_data.create_dataset(name=key, data=val, compression="gzip")
        for key, val in data_batch.general_data.items():
            if f"general_data.{key}" in include:
                general_data = get_group(h5obj=h5f, group_name="general_data")
                general_data.create_dataset(name=key, data=val, compression="gzip")


def read_from_hdf5(
    file_path: Union[Path, str],
    read_slice: Union[list[int], np.ndarray[int], slice] = (),
):
    file_path = Path(file_path)
    assert file_path.exists(), f"File {file_path} not found"
    trial_info_as_csv = Path(
        file_path.parent / (file_path.stem + "_trial_info.csv")
    ).exists()
    with h5py.File(file_path, "r") as h5f:
        states = h5f["states"][read_slice]
        inputs = h5f["inputs"][read_slice] if "inputs" in h5f.keys() else None
        outputs = h5f["outputs"][read_slice] if "outputs" in h5f.keys() else None
        temporal_data = (
            h5_to_dict(h5f["temporal_data"], read_slice=read_slice)
            if "temporal_data" in h5f.keys()
            else None
        )
        neural_data = (
            h5_to_dict(h5f["neural_data"], read_slice=read_slice)
            if "neural_data" in h5f.keys()
            else None
        )
        if trial_info_as_csv:
            trial_info = pd.read_csv(
                file_path.parent / (file_path.stem + "_trial_info.csv")
            )
        else:
            trial_info = (
                sarray_to_df(h5f["trial_info"][read_slice])
                if "trial_info" in h5f.keys()
                else None
            )
    data_batch = DataBatch(
        states=states,
        trial_info=trial_info,
        inputs=inputs,
        outputs=outputs,
        temporal_data=temporal_data,
        neural_data=neural_data,
    )
    return data_batch


def check_data_shape_hdf5(
    file_path: Union[Path, str],
):
    file_path = Path(file_path)
    assert file_path.exists(), f"File {file_path} not found"
    with h5py.File(file_path, "r") as h5f:
        b, t, _ = h5f["states"].shape
    return b, t


# numpy


def write_to_npz(
    file_path: Union[Path, str],
    data_batch: DataBatch,
    include: Optional[list] = None,
    trial_info_as_csv: bool = True,
    overwrite: bool = False,
):
    file_path = Path(file_path)
    assert (
        overwrite or not file_path.exists()
    ), f"Path {file_path} already exists. Please set `overwrite=True` if you wish to overwrite it."
    include = include or flatten_dict_keys(data_batch.__dict__)
    data_dict = {}
    if "states" in include:
        data_dict["states"] = data_batch.states
    if "inputs" in include and data_batch.inputs is not None:
        data_dict["inputs"] = data_batch.inputs
    if "outputs" in include and data_batch.outputs is not None:
        data_dict["outputs"] = data_batch.outputs
    if "trial_info" in include and data_batch.trial_info is not None:
        if trial_info_as_csv:
            data_batch.trial_info.to_csv(
                file_path.parent / (file_path.stem + "_trial_info.csv"), index=False
            )
        else:
            ti_as_array = df_to_sarray(data_batch.trial_info)
            if ti_as_array is not None:
                data_dict["trial_info"] = ti_as_array
    for key, val in data_batch.temporal_data.items():
        if f"temporal_data.{key}" in include:
            data_dict[f"temporal_data_{key}"] = val
    for key, val in data_batch.neural_data.items():
        if f"neural_data.{key}" in include:
            data_dict[f"neural_data_{key}"] = val
    for key, val in data_batch.general_data.items():
        if f"general_data.{key}" in include:
            data_dict[f"general_data_{key}"] = val
    np.savez(file_path, **data_dict)


def read_from_npz(
    file_path: Union[Path, str],
    read_slice: Union[list[int], np.ndarray[int], slice] = (),
):
    file_path = Path(file_path)
    assert file_path.exists(), f"File {file_path} not found"
    trial_info_as_csv = Path(
        file_path.parent / (file_path.stem + "_trial_info.csv")
    ).exists()
    if read_slice != ():
        print(
            "warning: specifying read slices when reading from npz does not save memory, as "
            "all the data are loaded when the file is read anyway"
        )
    npzfile = np.load(file_path)
    data_dict = {}
    if trial_info_as_csv:
        data_dict["trial_info"] = pd.read_csv(
            file_path.parent / (file_path.stem + "_trial_info.csv")
        )
    for filename in npzfile.files:
        if filename == "trial_info":  # could overwrite `trial_info_as_csv`. warn?
            data_dict["trial_info"] = sarray_to_df(npzfile[filename]).iloc[read_slice]
        elif "temporal_data" in filename:
            if "temporal_data" not in data_dict:
                data_dict["temporal_data"] = {}
            key = filename.partition("temporal_data_")[-1]
            data_dict["temporal_data"][key] = npzfile[filename][read_slice]
        elif "neural_data" in filename:
            if "neural_data" not in data_dict:
                data_dict["neural_data"] = {}
            key = filename.partition("neural_data_")[-1]
            data_dict["neural_data"][key] = npzfile[filename][read_slice]
        else:
            data_dict[filename] = npzfile[filename][read_slice]
    data_batch = DataBatch(**data_dict)
    return data_batch


def check_data_shape_npz(
    file_path: Union[Path, str],
):
    file_path = Path(file_path)
    assert file_path.exists(), f"File {file_path} not found"
    with zipfile.ZipFile(file_path) as archive:
        npy = archive.open("states.npy")
        version = np.lib.format.read_magic(npy)
        shape, *_ = np.lib.format._read_array_header(npy, version)
    b, t, *_ = shape
    return b, t


# nwb

# EXPERIMENTAL!!! DO NOT RECOMMEND!!!


def write_to_nwb(
    file_path: Union[Path, str],
    data_batch: DataBatch,
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
    trial_count, trial_length, _ = data_batch.states.shape
    trial_start_idxs = np.arange(trial_count) * (trial_length + inter_trial_interval)
    timestamps = np.concatenate(
        [(np.arange(trial_length) + start_idx) for start_idx in trial_start_idxs],
        axis=0,
    )
    timestamps = np.round(timestamps * dt, 9)
    flatten = lambda arr: arr.reshape(-1, arr.shape[-1])

    if data_batch.trial_info is not None:
        for col in data_batch.trial_info.columns:
            if col in ["start_time", "stop_time"]:
                raise AssertionError  # TODO: have a workaround
            nwbfile.add_trial_column(name=col, description=col)

    for i in range(trial_count):
        trial_data = {
            "start_time": round(trial_start_idxs[i] * dt, 9),
            "stop_time": round((trial_start_idxs[i] + trial_length) * dt, 9),
        }
        if data_batch.trial_info is not None:
            trial_info_dict = data_batch.trial_info.iloc[i].to_dict()
            for key in trial_info_dict.keys():
                if key.endswith("_time"):
                    trial_info_dict[key] = (
                        trial_info_dict[key] + trial_data["start_time"]
                    )
            trial_data.update(trial_info_dict)
        nwbfile.add_trial(**trial_data)

    simulation_data = []
    for field in ["states", "inputs", "outputs"]:
        if data_batch.__dict__.get(field, None) is not None:
            ts = TimeSeries(
                name=field,
                data=H5DataIO(
                    flatten(data_batch.__dict__[field]),
                    compression="gzip",
                ),
                timestamps=timestamps,
                description=f"Simulated {field} data.",
                unit="n/a",
            )
            simulation_data.append(ts)
    if data_batch.temporal_data is not None:
        for field in data_batch.temporal_data.keys():
            ts = TimeSeries(
                name=field,
                data=H5DataIO(
                    flatten(data_batch.temporal_data[field]),
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
    if data_batch.neural_data is not None:
        for field in data_batch.neural_data.keys():
            if field == "spikes":
                spikes = data_batch.neural_data["spikes"].astype(int)
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
                        flatten(data_batch.neural_data[field]),
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


# lfads-torch


def write_to_lfads(
    file_path: Union[Path, str],
    data_batch: DataBatch,
    data_field: str = "spikes",
    truth_field: Optional[str] = "rates",
    latent_field: Optional[str] = "states",
    ext_input_field: Optional[str] = None,
    trial_info_as_csv: bool = False,
    overwrite: bool = False,
    trial_split_ratio: Union[list, tuple] = (0.8, 0.2),
    neuron_split_ratio: Union[list, tuple] = (1.0, 0.0),
    extra_fields: list[str] = [],
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

    assert data_batch.neural_data is not None
    assert data_field in data_batch.neural_data
    if truth_field is not None:
        assert truth_field in data_batch.neural_data
    if latent_field is not None:  # TODO: clean up nested access
        if "." in latent_field:
            group, _, field = latent_field.partition(".")
            assert group in data_batch.__dict__
            assert field in data_batch.__dict__[group]
        else:
            assert latent_field in data_batch.__dict__
    if ext_input_field is not None:
        if "." in ext_input_field:
            group, _, field = ext_input_field.partition(".")
            assert group in data_batch.__dict__
            assert field in data_batch.__dict__[group]
        else:
            assert ext_input_field in data_batch.__dict__

    trial_inds = np.arange(len(data_batch))
    train_trial_inds, valid_trial_inds = train_test_split(
        trial_inds, test_size=trial_split_ratio[-1], random_state=seed
    )
    train_trial_inds = np.sort(train_trial_inds)
    valid_trial_inds = np.sort(valid_trial_inds)

    neuron_inds = np.arange(data_batch.neural_data["spikes"].shape[-1])
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
            data=data_batch.neural_data[data_field][train_trial_inds][
                :, :, heldin_neuron_inds
            ],
            compression="gzip",
        )
        h5f.create_dataset(
            "valid_encod_data",
            data=data_batch.neural_data[data_field][valid_trial_inds][
                :, :, heldin_neuron_inds
            ],
            compression="gzip",
        )

        h5f.create_dataset(
            "train_recon_data",
            data=data_batch.neural_data[data_field][train_trial_inds][
                :, :, all_neuron_inds
            ],
            compression="gzip",
        )
        h5f.create_dataset(
            "valid_recon_data",
            data=data_batch.neural_data[data_field][valid_trial_inds][
                :, :, all_neuron_inds
            ],
            compression="gzip",
        )

        if truth_field is not None:
            h5f.create_dataset(
                "train_truth",
                data=data_batch.neural_data[truth_field][train_trial_inds][
                    :, :, all_neuron_inds
                ],
                compression="gzip",
            )
            h5f.create_dataset(
                "valid_truth",
                data=data_batch.neural_data[truth_field][valid_trial_inds][
                    :, :, all_neuron_inds
                ],
                compression="gzip",
            )
            h5f.create_dataset("conversion_factor", data=1.0)

        if latent_field is not None:
            if "." in latent_field:
                group, _, field = latent_field.partition(".")
                data = data_batch.__dict__[group][field]
            else:
                data = data_batch.__dict__[field]
            h5f.create_dataset(
                "train_latents", data=data[train_trial_inds], compression="gzip"
            )
            h5f.create_dataset(
                "valid_latents", data=data[valid_trial_inds], compression="gzip"
            )

        if ext_input_field is not None:
            if "." in ext_input_field:
                group, _, field = ext_input_field.partition(".")
                data = data_batch.__dict__[group][field]
            else:
                data = data_batch.__dict__[field]
            h5f.create_dataset(
                "train_ext_input", data=data[train_trial_inds], compression="gzip"
            )
            h5f.create_dataset(
                "valid_ext_input", data=data[valid_trial_inds], compression="gzip"
            )

        h5f.create_dataset("train_inds", data=train_trial_inds, compression="gzip")
        h5f.create_dataset("valid_inds", data=valid_trial_inds, compression="gzip")

        h5f.create_dataset("channel_order", data=all_neuron_inds, compression="gzip")

        sampler_params = data_batch.general_data.get("data_sampler", dict())
        if "proj_weights" in sampler_params:
            h5f.create_dataset(
                "readout", data=sampler_params["proj_weights"], compression="gzip"
            )
        if "orig_mean" in sampler_params:
            h5f.create_dataset(
                "orig_mean", data=sampler_params["orig_mean"], compression="gzip"
            )
        if "orig_std" in sampler_params:
            h5f.create_dataset(
                "orig_std", data=sampler_params["orig_std"], compression="gzip"
            )
        if "std_mean" in sampler_params:
            h5f.create_dataset(
                "std_mean", data=sampler_params["std_mean"], compression="gzip"
            )

        if data_batch.trial_info is not None and trial_info_as_csv:
            df = data_batch.trial_info.copy()
            split = [
                "valid" if i in valid_trial_inds else "train" for i in range(len(df))
            ]
            df["split"] = split
            df.to_csv(
                file_path.parent / (file_path.stem + "_trial_info.csv"), index=False
            )

        all_fields = flatten_dict_keys(data_batch.__dict__)
        for field in extra_fields:
            if field not in all_fields:
                raise ValueError(
                    f"Field {field} not found in DataBatch. Expecting one of {all_fields}"
                )
            if "." in field:
                group, _, field = field.partition(".")
                data = data_batch.__dict__[group][field]
                if field in h5f.keys():  # dumb
                    field = f"{group}_field"
                h5f.create_dataset(
                    "train_" + field, data=data[train_trial_inds], compression="gzip"
                )
                h5f.create_dataset(
                    "valid_" + field, data=data[valid_trial_inds], compression="gzip"
                )
            else:
                h5f.create_dataset(
                    "train_" + field,
                    data=data_batch.__dict__[field][train_trial_inds],
                    compression="gzip",
                )
                h5f.create_dataset(
                    "valid_" + field,
                    data=data_batch.__dict__[field][valid_trial_inds],
                    compression="gzip",
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


def dict_to_h5(
    data: dict,
    h5obj: Union[h5py.File, h5py.Group],
):
    for key, val in data.items():
        if isinstance(val, dict):
            h5group = h5obj.create_group(key)
            dict_to_h5(val, h5group)
        else:
            h5obj.create_dataset(key, data=val)


def flatten_dict_keys(d, prefix="", exclude=[]):
    keys = []
    for key in d.keys():
        if key in exclude:
            continue
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
