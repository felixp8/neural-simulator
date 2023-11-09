from pathlib import Path
from typing import Optional, Union
import torch.utils.data as data

from .types import DataBatch
from .data_io import read_file, check_data_shape


class SyntheticNeuralDataset(data.Dataset):
    def __init__(
        self,
        source_data: Union[str, Path, DataBatch],
        data_fields: Optional[list[str]],
        in_memory: bool = True,
    ):
        super().__init__()
        self.data_fields = data_fields
        if isinstance(source_data, DataBatch):
            self.source_data = source_data
            self.source_data_path = None
        else:
            source_data = Path(source_data)
            assert source_data.suffix in [".h5", ".npz", ".nwb"]
            self.source_data = None
            self.source_data_path = source_data
            if in_memory:
                self.source_data = read_file(self.source_data_path)

    def __len__(self):
        if self.source_data is not None:
            return len(self.source_data)
        else:
            return check_data_shape(self.source_data_path)[0]

    def __getitem__(self, idx):
        if self.source_data is None:
            data_item = read_file(self.source_data_path, read_slice=[idx])
        else:
            data_item = self.source_data[idx]
        data_list = []
        for field in self.data_fields:
            if field.startswith("temporal_data"):
                data_list.append(data_item.temporal_data[field.partition(".")[-1]])
            elif field.startswith("neural_data"):
                data_list.append(data_item.neural_data[field.partition(".")[-1]])
            else:
                data_list.append(data_item.__dict__[field])
        return tuple(data_list)
