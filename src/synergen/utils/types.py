import numpy as np
import pandas as pd
from typing import Union, Optional, Any
from dataclasses import dataclass, field


NumpyArray = Union[np.ndarray, np.ma.MaskedArray]


@dataclass
class DataBatch:
    states: NumpyArray
    trial_info: Optional[pd.DataFrame] = None
    inputs: Optional[NumpyArray] = None
    outputs: Optional[NumpyArray] = None
    neural_data: dict[str, NumpyArray] = field(default_factory=dict)
    temporal_data: dict[str, NumpyArray] = field(default_factory=dict)
    general_data: dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, key):
        expand_to_nd = (
            lambda arr, n: np.expand_dims(arr, axis=tuple(range(n - arr.ndim)))
            if arr.ndim < n
            else arr
        )
        expand_iloc = lambda key: [key] if isinstance(key, int) else key
        return DataBatch(
            states=expand_to_nd(self.states[key], 3),
            trial_info=None
            if self.trial_info is None
            else self.trial_info.iloc[expand_iloc(key)],
            inputs=None if self.inputs is None else expand_to_nd(self.inputs[key], 3),
            outputs=None
            if self.outputs is None
            else expand_to_nd(self.outputs[key], 3),
            neural_data={
                neural_key: expand_to_nd(neural_val[key], 3)
                for neural_key, neural_val in self.neural_data.items()
            },
            temporal_data={
                temporal_key: expand_to_nd(temporal_val[key], 3)
                for temporal_key, temporal_val in self.temporal_data.items()
            },
            general_data=self.general_data,
        )


def validate_data_batch(data_batch: DataBatch):
    batch_size = len(data_batch)
    trial_len = data_batch.states.shape[1]

    def check_partial_shape(arr_name, arr, shape):
        if arr is None:
            return
        assert len(shape) == len(arr.shape), (
            f"Array {arr_name} has wrong number of dimensions. "
            f"Expected {len(shape)}, got {len(arr.shape)}: {arr.shape}"
        )
        for i, dim in enumerate(shape):
            if dim == -1:
                continue
            else:
                assert dim == arr.shape[i], (
                    f"Array {arr_name} has incorrect size for dimension {i}. "
                    f"Expected {dim}, got {arr.shape[i]}: {arr.shape}"
                )

    check_partial_shape("inputs", data_batch.inputs, (batch_size, trial_len, -1))
    check_partial_shape("outputs", data_batch.outputs, (batch_size, trial_len, -1))
    check_partial_shape("trial_info", data_batch.trial_info, (batch_size, -1))
    for key, val in data_batch.neural_data.items():
        check_partial_shape(f"neural_data.{key}", val, (batch_size, trial_len, -1))
    for key, val in data_batch.temporal_data.items():
        check_partial_shape(f"temporal_data.{key}", val, (batch_size, trial_len, -1))


def stack_data_batches(data_batches: list[DataBatch]):
    assert len(data_batches) > 0, "Cannot stack trajectory batch list with length 0"
    if len(data_batches) == 1:
        return data_batches[0]

    def cat(obj_list: Union[list, tuple]):
        if obj_list[0] is None:
            return None
        elif isinstance(obj_list[0], (np.ndarray, np.ma.MaskedArray)):
            return np.concatenate(obj_list, axis=0)
        elif isinstance(obj_list[0], pd.DataFrame):
            return pd.concat(obj_list, axis=0, ignore_index=True)
        elif isinstance(obj_list[0], dict):
            return {
                key: cat([obj[key] for obj in obj_list]) for key in obj_list[0].keys()
            }

    # currently assumes that all keys of neural_data, temporal_data, etc. are present
    # in every DataBatch
    stacked = {
        field: (
            dict(kv for tb in data_batches for kv in tb.__dict__[field].items())
            if field == "general_data"
            else cat([tb.__dict__[field] for tb in data_batches])
        )
        for field in data_batches[0].__dict__.keys()
    }
    return DataBatch(**stacked)


def shuffle_data_batch(
    data_batch: DataBatch,
    seed: Optional[Union[int, np.random.Generator]] = None,
):
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data_batch))
    shuffled_data = {}
    for field in data_batch.__dict__.keys():
        if data_batch.__dict__[field] is None:
            continue
        if isinstance(data_batch.__dict__[field], dict):
            if field == "general_data":
                shuffled_data[field] = data_batch.__dict__[field]
            else:
                shuffled_data[field] = {
                    key: val[perm] for key, val in data_batch.__dict__[field].items()
                }
        else:
            shuffled_data[field] = data_batch.__dict__[field][perm]
    return DataBatch(**shuffled_data)
