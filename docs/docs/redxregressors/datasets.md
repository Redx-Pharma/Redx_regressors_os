# Module redxregressors.datasets

Module for classes and methods to store and data sets

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module for classes and methods to store and data sets

        """

        import torch

        import os

        from pathlib import Path

        from torch_geometric.data import InMemoryDataset

        from torch_geometric.utils import from_smiles

        import logging

        from torch_geometric.loader import DataLoader

        import numpy as np

        import pandas as pd

        from typing import List, Optional, Union, Callable

        from tqdm import tqdm

        from redxregressors import ml_featurization, utilities

        log = logging.getLogger(__name__)



        class RedxCSVGraphMolDataSet(InMemoryDataset):

            """

            This class is intended to read in a CSV file containing SMILES strings and labels/target values and

            convert them into a PyTorch Geometric dataset. This is useful for training graph neural networks on

            molecular data.

            The class is based on the PyTorch Geometric InMemoryDataset class and so inherits all of its methods. The

            pytorch geometric MoleculeNet dataset is used as a template for this class. The pytorch geometric code

            is licensed under the MIT license which is given below.

            Copyright (c) 2023 PyG Team <team@pyg.org>

            Permission is hereby granted, free of charge, to any person obtaining a copy

            of this software and associated documentation files (the "Software"), to deal

            in the Software without restriction, including without limitation the rights

            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell

            copies of the Software, and to permit persons to whom the Software is

            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in

            all copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,

            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE

            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER

            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN

            THE SOFTWARE.

            """

            def __init__(

                self,

                csv_file: Union[str, List[str], pd.DataFrame],

                smiles_column: str,

                property_columns: Union[str, List[str]],

                description: Optional[str] = None,

                id_column: Optional[str] = None,

                save_path: Optional[str] = os.getcwd(),

                save_name: Optional[str] = None,

                transform: Optional[Callable] = None,

                pre_transform: Optional[Callable] = from_smiles,

                overwrite_existing=True,

            ) -> None:

                """

                Initialize the class and read in the data from the CSV file and convert it to PyTorch Geometric Data objects.

                Note this is only valid for single objectives.

                Args:

                    root (string): Root directory where the dataset should be saved.

                    save_path (string): Root directory where the dataset should be saved.

                    csv_file (string): The name of the CSV file containing the data.

                    smiles_column (string): The name of the column in the CSV file containing the SMILES strings.

                    label_column (string): The name of the column in the CSV

                    file containing the target values.

                    transform (callable, optional): A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version.

                    The data object will be transformed before every access. (default: None)

                    pre_transform (callable, optional): A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version.

                    The data object will be transformed before being saved to disk. (default: None)

                """

                # store column headers that are needed and make sure the property columns is a list and if not convert it to a list

                self.csv_file = csv_file

                self.smiles_column = smiles_column

                if isinstance(property_columns, str):

                    self.property_columns = [property_columns]

                else:

                    self.property_columns = property_columns

                self.id_column = id_column

                self.description = description

                self.save_name = save_name

                self.save_path = Path(path=save_path)

                # call the parent class constructor which will run process() to convert the data to PyTorch Geometric Data objects and save them to disk

                super(RedxCSVGraphMolDataSet, self).__init__(

                    save_path, transform, pre_transform, force_reload=overwrite_existing

                )

                # Load the saved graph data from disk and store it in the class as an in memory data set

                self.load(self.processed_paths[0])

            @property

            def raw_file_names(self) -> List[str]:

                """

                Function to return the raw file names

                """

                if isinstance(self.csv_file, list):

                    return self.csv_file

                else:

                    return [self.csv_file]

            @property

            def processed_dir(self) -> str:

                """

                Function to return the processed directory path. Note that 'processed' is made in the call to process() if it does not exist

                """

                return str(self.save_path.joinpath("processed").absolute())

            @property

            def processed_file_names(self) -> str:

                """

                Function to return the processed file names

                """

                return "graph_data.pt"

            def process(self) -> None:

                """

                Function inherited from the base Dataset class to process the data and save it to disk for each graph entry

                In this case the function reads in the CSV file and converts the SMILES strings to PyTorch Geometric Data objects

                using the default from_smiles function. The target values are also added to the data object as a tensor. We

                can then use the pre_transform function to apply any transformations to the data object before saving it to disk. In the

                present example this is used to extract the smiles from the default data object gotten from the from_smiles function and

                passes these to the new featurization function for the AttentiveFP model. The data object is then saved to disk.

                Note that providing no pre_transform function will result in the default data object being saved to disk and to get a modified

                graph data object for each molecule we can provide a pre_transform function that modifies the data object before saving it to disk.

                """

                # check if we have a single csv file to load from disk or a list of csv files to load and concatenate

                if isinstance(self.csv_file, list):

                    data = pd.concat([pd.read_csv(f) for f in self.csv_file])

                elif isinstance(self.csv_file, str):

                    data = pd.read_csv(self.csv_file)

                elif isinstance(self.csv_file, pd.DataFrame):

                    data = self.csv_file

                else:

                    raise ValueError(

                        "csv_file must be a string, pandas dataframe or a list of strings"

                    )

                # drop any rows with missing values in the SMILES or target columns

                data = data.dropna(subset=[self.smiles_column] + self.property_columns, axis=0)

                # get the SMILES strings and target values from the data

                log.info(f"Data columns are as follows: {data.columns}")

                smiles = data[self.smiles_column].copy().tolist()

                properties = np.array(

                    [data[p].copy().to_list() for p in self.property_columns], dtype=np.float32

                )

                log.info(f"Properties targets are as follows: {properties}")

                # get the IDs if they are present in the data

                if self.id_column is not None:

                    self.ids = data[self.id_column].tolist()

                else:

                    self.ids = data.index.tolist()

                # create a list to store the data objects

                data_list = []

                # loop over the SMILES strings and target values and create a PyTorch Geometric Data object for each

                for ith in tqdm(range(len(smiles))):

                    # validate the SMILES string

                    smi = utilities.validate_smile(smiles[ith])

                    # create the PyTorch Geometric Data object

                    data = ml_featurization.from_smiles_without_default_graph_feature_gen(

                        smi

                    )  # avoid needless default featurization from_smiles(smi). We use from_smiles() as the defualt pre_transform.

                    data.y = torch.tensor(properties[:, ith], dtype=torch.float)

                    # check if the data object is empty and skip if it is

                    if data.num_edges == 0 or data.num_nodes == 0:

                        log.warning(

                            f"Skipping molecule {self.ids[ith]} graph featureization failed"

                        )

                        log.warning(f"Input smiles: '{smiles[ith]}' validated smiles '{smi}'")

                        continue

                    # apply the pre_transform function to the data object if we want graph featurization that is different from the pytorch geometric default for smiles

                    if self.pre_transform is not None:

                        data = self.pre_transform(data)

                    # append the data object to the list

                    data_list.append(data)

                # save the data objects to disk

                processed_dir_path = Path(self.processed_dir)

                if not os.path.isdir(self.processed_dir):

                    processed_dir_path.mkdir(parents=True, exist_ok=True)

                self.save(data_list, self.processed_paths[0])



        def split_redx_csv_data_set_into_train_test_validation(

            data,

            train_frac: float = 0.8,

            test_frac: float = 0.1,

            batch_size: int = 1,

        ) -> tuple[DataLoader, DataLoader, DataLoader]:

            """

            Function to split a PyTorch Geometric dataset into training, test, and validation sets. The function uses the DataLoader class

            from PyTorch to create the data loaders for each set. The function also ensures that the train, test, and validation sets do not

            have overlapping indices.

            Args:

                data (Dataset): The PyTorch Geometric dataset to split.

                train_frac (float): The fraction of the data to use for the training set.

                test_frac (float): The fraction of the data to use for the test set.

                batch_size (int): The batch size to use for the data loaders.

            Returns:

                tuple[DataLoader, DataLoader, DataLoader]: The training, test, and validation data loaders.

            """

            n_train = int(len(data) * train_frac)

            n_test = int(len(data) * test_frac)

            indexes = [ith for ith in range(len(data))]

            train_indices = np.random.choice(indexes, size=n_train, replace=False)

            test_indices = np.array(list(set(indexes) - set(train_indices)))

            test_indices = np.random.choice(test_indices, size=n_test, replace=False)

            validation_indices = np.array(

                list(set(indexes) - set(train_indices) - set(test_indices))

            )

            if any(ent in train_indices for ent in test_indices):

                raise RuntimeError("ERROR - The train and test sets have overlapping indices")

            if any(ent in train_indices for ent in validation_indices):

                raise RuntimeError(

                    "ERROR - The train and validation sets have overlapping indices"

                )

            g = torch.Generator()

            g.manual_seed(0)

            train = DataLoader(

                [data[ith] for ith in train_indices],

                batch_size=batch_size,

                shuffle=True,

                worker_init_fn=utilities.seed_worker,

                generator=g,

            )

            test = DataLoader(

                [data[ith] for ith in test_indices],

                batch_size=1,

                shuffle=False,

                worker_init_fn=utilities.seed_worker,

                generator=g,

            )

            validation = DataLoader(

                [data[ith] for ith in validation_indices],

                batch_size=1,

                shuffle=False,

                worker_init_fn=utilities.seed_worker,

                generator=g,

            )

            return train, test, validation



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### split_redx_csv_data_set_into_train_test_validation

```python3
def split_redx_csv_data_set_into_train_test_validation(
    data,
    train_frac: float = 0.8,
    test_frac: float = 0.1,
    batch_size: int = 1
) -> tuple[torch_geometric.loader.dataloader.DataLoader, torch_geometric.loader.dataloader.DataLoader, torch_geometric.loader.dataloader.DataLoader]
```

Function to split a PyTorch Geometric dataset into training, test, and validation sets. The function uses the DataLoader class

from PyTorch to create the data loaders for each set. The function also ensures that the train, test, and validation sets do not
have overlapping indices.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data | Dataset | The PyTorch Geometric dataset to split. | None |
| train_frac | float | The fraction of the data to use for the training set. | None |
| test_frac | float | The fraction of the data to use for the test set. | None |
| batch_size | int | The batch size to use for the data loaders. | None |

**Returns:**

| Type | Description |
|---|---|
| tuple[DataLoader, DataLoader, DataLoader] | The training, test, and validation data loaders. |

??? example "View Source"
        def split_redx_csv_data_set_into_train_test_validation(

            data,

            train_frac: float = 0.8,

            test_frac: float = 0.1,

            batch_size: int = 1,

        ) -> tuple[DataLoader, DataLoader, DataLoader]:

            """

            Function to split a PyTorch Geometric dataset into training, test, and validation sets. The function uses the DataLoader class

            from PyTorch to create the data loaders for each set. The function also ensures that the train, test, and validation sets do not

            have overlapping indices.

            Args:

                data (Dataset): The PyTorch Geometric dataset to split.

                train_frac (float): The fraction of the data to use for the training set.

                test_frac (float): The fraction of the data to use for the test set.

                batch_size (int): The batch size to use for the data loaders.

            Returns:

                tuple[DataLoader, DataLoader, DataLoader]: The training, test, and validation data loaders.

            """

            n_train = int(len(data) * train_frac)

            n_test = int(len(data) * test_frac)

            indexes = [ith for ith in range(len(data))]

            train_indices = np.random.choice(indexes, size=n_train, replace=False)

            test_indices = np.array(list(set(indexes) - set(train_indices)))

            test_indices = np.random.choice(test_indices, size=n_test, replace=False)

            validation_indices = np.array(

                list(set(indexes) - set(train_indices) - set(test_indices))

            )

            if any(ent in train_indices for ent in test_indices):

                raise RuntimeError("ERROR - The train and test sets have overlapping indices")

            if any(ent in train_indices for ent in validation_indices):

                raise RuntimeError(

                    "ERROR - The train and validation sets have overlapping indices"

                )

            g = torch.Generator()

            g.manual_seed(0)

            train = DataLoader(

                [data[ith] for ith in train_indices],

                batch_size=batch_size,

                shuffle=True,

                worker_init_fn=utilities.seed_worker,

                generator=g,

            )

            test = DataLoader(

                [data[ith] for ith in test_indices],

                batch_size=1,

                shuffle=False,

                worker_init_fn=utilities.seed_worker,

                generator=g,

            )

            validation = DataLoader(

                [data[ith] for ith in validation_indices],

                batch_size=1,

                shuffle=False,

                worker_init_fn=utilities.seed_worker,

                generator=g,

            )

            return train, test, validation

## Classes

### RedxCSVGraphMolDataSet

```python3
class RedxCSVGraphMolDataSet(
    csv_file: Union[str, List[str], pandas.core.frame.DataFrame],
    smiles_column: str,
    property_columns: Union[str, List[str]],
    description: Optional[str] = None,
    id_column: Optional[str] = None,
    save_path: Optional[str] = '/Users/j.mcdonagh/Documents/Projects/Software/For_Open_Sourcing/Redx_regressors_os/docs',
    save_name: Optional[str] = None,
    transform: Optional[Callable] = None,
    pre_transform: Optional[Callable] = <function from_smiles at 0x17d453560>,
    overwrite_existing=True
)
```

This class is intended to read in a CSV file containing SMILES strings and labels/target values and

convert them into a PyTorch Geometric dataset. This is useful for training graph neural networks on
molecular data.

The class is based on the PyTorch Geometric InMemoryDataset class and so inherits all of its methods. The
pytorch geometric MoleculeNet dataset is used as a template for this class. The pytorch geometric code
is licensed under the MIT license which is given below.
Copyright (c) 2023 PyG Team <team@pyg.org>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

#### Ancestors (in MRO)

* torch_geometric.data.in_memory_dataset.InMemoryDataset
* torch_geometric.data.dataset.Dataset
* torch.utils.data.dataset.Dataset
* typing.Generic

#### Static methods


#### collate

```python3
def collate(
    data_list: Sequence[torch_geometric.data.data.BaseData]
) -> Tuple[torch_geometric.data.data.BaseData, Optional[Dict[str, torch.Tensor]]]
```

Collates a list of :class:`~torch_geometric.data.Data` or

??? example "View Source"
            @staticmethod

            def collate(

                data_list: Sequence[BaseData],

            ) -> Tuple[BaseData, Optional[Dict[str, Tensor]]]:

                r"""Collates a list of :class:`~torch_geometric.data.Data` or

                :class:`~torch_geometric.data.HeteroData` objects to the internal

                storage format of :class:`~torch_geometric.data.InMemoryDataset`.

                """

                if len(data_list) == 1:

                    return data_list[0], None

                data, slices, _ = collate(

                    data_list[0].__class__,

                    data_list=data_list,

                    increment=False,

                    add_batch=False,

                )

                return data, slices


#### save

```python3
def save(
    data_list: Sequence[torch_geometric.data.data.BaseData],
    path: str
) -> None
```

Saves a list of data objects to the file path :obj:`path`.

??? example "View Source"
            @classmethod

            def save(cls, data_list: Sequence[BaseData], path: str) -> None:

                r"""Saves a list of data objects to the file path :obj:`path`."""

                data, slices = cls.collate(data_list)

                fs.torch_save((data.to_dict(), slices, data.__class__), path)

#### Instance variables

```python3
data
```

```python3
has_download
```

Checks whether the dataset defines a :meth:`download` method.

```python3
has_process
```

Checks whether the dataset defines a :meth:`process` method.

```python3
num_classes
```

```python3
num_edge_features
```

Returns the number of features per edge in the dataset.

```python3
num_features
```

Returns the number of features per node in the dataset.

Alias for :py:attr:`~num_node_features`.

```python3
num_node_features
```

Returns the number of features per node in the dataset.

```python3
processed_dir
```

Function to return the processed directory path. Note that 'processed' is made in the call to process() if it does not exist

```python3
processed_file_names
```

Function to return the processed file names

```python3
processed_paths
```

The absolute filepaths that must be present in order to skip

processing.

```python3
raw_dir
```

```python3
raw_file_names
```

Function to return the raw file names

```python3
raw_paths
```

The absolute filepaths that must be present in order to skip

downloading.

#### Methods


#### copy

```python3
def copy(
    self,
    idx: Union[slice, torch.Tensor, numpy.ndarray, collections.abc.Sequence, NoneType] = None
) -> 'InMemoryDataset'
```

Performs a deep-copy of the dataset. If :obj:`idx` is not given,

will clone the full dataset. Otherwise, will only clone a subset of the
dataset from indices :obj:`idx`.
Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or

??? example "View Source"
            def copy(self, idx: Optional[IndexType] = None) -> 'InMemoryDataset':

                r"""Performs a deep-copy of the dataset. If :obj:`idx` is not given,

                will clone the full dataset. Otherwise, will only clone a subset of the

                dataset from indices :obj:`idx`.

                Indices can be slices, lists, tuples, and a :obj:`torch.Tensor` or

                :obj:`np.ndarray` of type long or bool.

                """

                if idx is None:

                    data_list = [self.get(i) for i in self.indices()]

                else:

                    data_list = [self.get(i) for i in self.index_select(idx).indices()]

                dataset = copy.copy(self)

                dataset._indices = None

                dataset._data_list = None

                dataset.data, dataset.slices = self.collate(data_list)

                return dataset


#### cpu

```python3
def cpu(
    self,
    *args: str
) -> 'InMemoryDataset'
```

Moves the dataset to CPU memory.

??? example "View Source"
            def cpu(self, *args: str) -> 'InMemoryDataset':

                r"""Moves the dataset to CPU memory."""

                return self.to(torch.device('cpu'))


#### cuda

```python3
def cuda(
    self,
    device: Union[int, str, NoneType] = None
) -> 'InMemoryDataset'
```

Moves the dataset toto CUDA memory.

??? example "View Source"
            def cuda(

                self,

                device: Optional[Union[int, str]] = None,

            ) -> 'InMemoryDataset':

                r"""Moves the dataset toto CUDA memory."""

                if isinstance(device, int):

                    device = f'cuda:{int}'

                elif device is None:

                    device = 'cuda'

                return self.to(device)


#### download

```python3
def download(
    self
) -> None
```

Downloads the dataset to the :obj:`self.raw_dir` folder.

??? example "View Source"
            def download(self) -> None:

                r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""

                raise NotImplementedError


#### get

```python3
def get(
    self,
    idx: int
) -> torch_geometric.data.data.BaseData
```

Gets the data object at index :obj:`idx`.

??? example "View Source"
            def get(self, idx: int) -> BaseData:

                # TODO (matthias) Avoid unnecessary copy here.

                if self.len() == 1:

                    return copy.copy(self._data)

                if not hasattr(self, '_data_list') or self._data_list is None:

                    self._data_list = self.len() * [None]

                elif self._data_list[idx] is not None:

                    return copy.copy(self._data_list[idx])

                data = separate(

                    cls=self._data.__class__,

                    batch=self._data,

                    idx=idx,

                    slice_dict=self.slices,

                    decrement=False,

                )

                self._data_list[idx] = copy.copy(data)

                return data


#### get_summary

```python3
def get_summary(
    self
) -> Any
```

Collects summary statistics for the dataset.

??? example "View Source"
            def get_summary(self) -> Any:

                r"""Collects summary statistics for the dataset."""

                from torch_geometric.data.summary import Summary

                return Summary.from_dataset(self)


#### index_select

```python3
def index_select(
    self,
    idx: Union[slice, torch.Tensor, numpy.ndarray, collections.abc.Sequence]
) -> 'Dataset'
```

Creates a subset of the dataset from specified indices :obj:`idx`.

Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
long or bool.

??? example "View Source"
            def index_select(self, idx: IndexType) -> 'Dataset':

                r"""Creates a subset of the dataset from specified indices :obj:`idx`.

                Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a

                list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type

                long or bool.

                """

                indices = self.indices()

                if isinstance(idx, slice):

                    start, stop, step = idx.start, idx.stop, idx.step

                    # Allow floating-point slicing, e.g., dataset[:0.9]

                    if isinstance(start, float):

                        start = round(start * len(self))

                    if isinstance(stop, float):

                        stop = round(stop * len(self))

                    idx = slice(start, stop, step)

                    indices = indices[idx]

                elif isinstance(idx, Tensor) and idx.dtype == torch.long:

                    return self.index_select(idx.flatten().tolist())

                elif isinstance(idx, Tensor) and idx.dtype == torch.bool:

                    idx = idx.flatten().nonzero(as_tuple=False)

                    return self.index_select(idx.flatten().tolist())

                elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:

                    return self.index_select(idx.flatten().tolist())

                elif isinstance(idx, np.ndarray) and idx.dtype == bool:

                    idx = idx.flatten().nonzero()[0]

                    return self.index_select(idx.flatten().tolist())

                elif isinstance(idx, Sequence) and not isinstance(idx, str):

                    indices = [indices[i] for i in idx]

                else:

                    raise IndexError(

                        f"Only slices (':'), list, tuples, torch.tensor and "

                        f"np.ndarray of dtype long or bool are valid indices (got "

                        f"'{type(idx).__name__}')")

                dataset = copy.copy(self)

                dataset._indices = indices

                return dataset


#### indices

```python3
def indices(
    self
) -> collections.abc.Sequence
```

??? example "View Source"
            def indices(self) -> Sequence:

                return range(self.len()) if self._indices is None else self._indices


#### len

```python3
def len(
    self
) -> int
```

Returns the number of data objects stored in the dataset.

??? example "View Source"
            def len(self) -> int:

                if self.slices is None:

                    return 1

                for _, value in nested_iter(self.slices):

                    return len(value) - 1

                return 0


#### load

```python3
def load(
    self,
    path: str,
    data_cls: Type[torch_geometric.data.data.BaseData] = <class 'torch_geometric.data.data.Data'>
) -> None
```

Loads the dataset from the file path :obj:`path`.

??? example "View Source"
            def load(self, path: str, data_cls: Type[BaseData] = Data) -> None:

                r"""Loads the dataset from the file path :obj:`path`."""

                out = fs.torch_load(path)

                assert isinstance(out, tuple)

                assert len(out) == 2 or len(out) == 3

                if len(out) == 2:  # Backward compatibility.

                    data, self.slices = out

                else:

                    data, self.slices, data_cls = out

                if not isinstance(data, dict):  # Backward compatibility.

                    self.data = data

                else:

                    self.data = data_cls.from_dict(data)


#### print_summary

```python3
def print_summary(
    self,
    fmt: str = 'psql'
) -> None
```

Prints summary statistics of the dataset to the console.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| fmt | str | Summary tables format. Available table formats<br>can be found `here <https://github.com/astanin/python-tabulate?<br>tab=readme-ov-file#table-format>`__. (default: :obj:`"psql"`) | None |

??? example "View Source"
            def print_summary(self, fmt: str = "psql") -> None:

                r"""Prints summary statistics of the dataset to the console.

                Args:

                    fmt (str, optional): Summary tables format. Available table formats

                        can be found `here <https://github.com/astanin/python-tabulate?

                        tab=readme-ov-file#table-format>`__. (default: :obj:`"psql"`)

                """

                print(self.get_summary().format(fmt=fmt))


#### process

```python3
def process(
    self
) -> None
```

Function inherited from the base Dataset class to process the data and save it to disk for each graph entry

In this case the function reads in the CSV file and converts the SMILES strings to PyTorch Geometric Data objects
using the default from_smiles function. The target values are also added to the data object as a tensor. We
can then use the pre_transform function to apply any transformations to the data object before saving it to disk. In the
present example this is used to extract the smiles from the default data object gotten from the from_smiles function and
passes these to the new featurization function for the AttentiveFP model. The data object is then saved to disk.

Note that providing no pre_transform function will result in the default data object being saved to disk and to get a modified
graph data object for each molecule we can provide a pre_transform function that modifies the data object before saving it to disk.

??? example "View Source"
            def process(self) -> None:

                """

                Function inherited from the base Dataset class to process the data and save it to disk for each graph entry

                In this case the function reads in the CSV file and converts the SMILES strings to PyTorch Geometric Data objects

                using the default from_smiles function. The target values are also added to the data object as a tensor. We

                can then use the pre_transform function to apply any transformations to the data object before saving it to disk. In the

                present example this is used to extract the smiles from the default data object gotten from the from_smiles function and

                passes these to the new featurization function for the AttentiveFP model. The data object is then saved to disk.

                Note that providing no pre_transform function will result in the default data object being saved to disk and to get a modified

                graph data object for each molecule we can provide a pre_transform function that modifies the data object before saving it to disk.

                """

                # check if we have a single csv file to load from disk or a list of csv files to load and concatenate

                if isinstance(self.csv_file, list):

                    data = pd.concat([pd.read_csv(f) for f in self.csv_file])

                elif isinstance(self.csv_file, str):

                    data = pd.read_csv(self.csv_file)

                elif isinstance(self.csv_file, pd.DataFrame):

                    data = self.csv_file

                else:

                    raise ValueError(

                        "csv_file must be a string, pandas dataframe or a list of strings"

                    )

                # drop any rows with missing values in the SMILES or target columns

                data = data.dropna(subset=[self.smiles_column] + self.property_columns, axis=0)

                # get the SMILES strings and target values from the data

                log.info(f"Data columns are as follows: {data.columns}")

                smiles = data[self.smiles_column].copy().tolist()

                properties = np.array(

                    [data[p].copy().to_list() for p in self.property_columns], dtype=np.float32

                )

                log.info(f"Properties targets are as follows: {properties}")

                # get the IDs if they are present in the data

                if self.id_column is not None:

                    self.ids = data[self.id_column].tolist()

                else:

                    self.ids = data.index.tolist()

                # create a list to store the data objects

                data_list = []

                # loop over the SMILES strings and target values and create a PyTorch Geometric Data object for each

                for ith in tqdm(range(len(smiles))):

                    # validate the SMILES string

                    smi = utilities.validate_smile(smiles[ith])

                    # create the PyTorch Geometric Data object

                    data = ml_featurization.from_smiles_without_default_graph_feature_gen(

                        smi

                    )  # avoid needless default featurization from_smiles(smi). We use from_smiles() as the defualt pre_transform.

                    data.y = torch.tensor(properties[:, ith], dtype=torch.float)

                    # check if the data object is empty and skip if it is

                    if data.num_edges == 0 or data.num_nodes == 0:

                        log.warning(

                            f"Skipping molecule {self.ids[ith]} graph featureization failed"

                        )

                        log.warning(f"Input smiles: '{smiles[ith]}' validated smiles '{smi}'")

                        continue

                    # apply the pre_transform function to the data object if we want graph featurization that is different from the pytorch geometric default for smiles

                    if self.pre_transform is not None:

                        data = self.pre_transform(data)

                    # append the data object to the list

                    data_list.append(data)

                # save the data objects to disk

                processed_dir_path = Path(self.processed_dir)

                if not os.path.isdir(self.processed_dir):

                    processed_dir_path.mkdir(parents=True, exist_ok=True)

                self.save(data_list, self.processed_paths[0])


#### shuffle

```python3
def shuffle(
    self,
    return_perm: bool = False
) -> Union[ForwardRef('Dataset'), Tuple[ForwardRef('Dataset'), torch.Tensor]]
```

Randomly shuffles the examples in the dataset.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| return_perm | bool | If set to :obj:`True`, will also<br>return the random permutation used to shuffle the dataset.<br>(default: :obj:`False`) | None |

??? example "View Source"
            def shuffle(

                self,

                return_perm: bool = False,

            ) -> Union['Dataset', Tuple['Dataset', Tensor]]:

                r"""Randomly shuffles the examples in the dataset.

                Args:

                    return_perm (bool, optional): If set to :obj:`True`, will also

                        return the random permutation used to shuffle the dataset.

                        (default: :obj:`False`)

                """

                perm = torch.randperm(len(self))

                dataset = self.index_select(perm)

                return (dataset, perm) if return_perm is True else dataset


#### to

```python3
def to(
    self,
    device: Union[int, str]
) -> 'InMemoryDataset'
```

Performs device conversion of the whole dataset.

??? example "View Source"
            def to(self, device: Union[int, str]) -> 'InMemoryDataset':

                r"""Performs device conversion of the whole dataset."""

                if self._indices is not None:

                    raise ValueError("The given 'InMemoryDataset' only references a "

                                     "subset of examples of the full dataset")

                if self._data_list is not None:

                    raise ValueError("The data of the dataset is already cached")

                self._data.to(device)

                return self


#### to_datapipe

```python3
def to_datapipe(
    self
) -> Any
```

Converts the dataset into a :class:`torch.utils.data.DataPipe`.

The returned instance can then be used with :pyg:`PyG's` built-in

??? example "View Source"
            def to_datapipe(self) -> Any:

                r"""Converts the dataset into a :class:`torch.utils.data.DataPipe`.

                The returned instance can then be used with :pyg:`PyG's` built-in

                :class:`DataPipes` for baching graphs as follows:

                .. code-block:: python

                    from torch_geometric.datasets import QM9

                    dp = QM9(root='./data/QM9/').to_datapipe()

                    dp = dp.batch_graphs(batch_size=2, drop_last=True)

                    for batch in dp:

                        pass

                See the `PyTorch tutorial

                <https://pytorch.org/data/main/tutorial.html>`_ for further background

                on DataPipes.

                """

                from torch_geometric.data.datapipes import DatasetAdapter

                return DatasetAdapter(self)


#### to_on_disk_dataset

```python3
def to_on_disk_dataset(
    self,
    root: Optional[str] = None,
    backend: str = 'sqlite',
    log: bool = True
) -> 'torch_geometric.data.OnDiskDataset'
```

Converts the :class:`InMemoryDataset` to a :class:`OnDiskDataset`

variant. Useful for distributed training and hardware instances with
limited amount of shared memory.

root (str, optional): Root directory where the dataset should be saved.
    If set to :obj:`None`, will save the dataset in
    :obj:`root/on_disk`.
    Note that it is important to specify :obj:`root` to account for
    different dataset splits. (optional: :obj:`None`)
backend (str): The :class:`Database` backend to use.
    (default: :obj:`"sqlite"`)
log (bool, optional): Whether to print any console output while
    processing the dataset. (default: :obj:`True`)

??? example "View Source"
            def to_on_disk_dataset(

                self,

                root: Optional[str] = None,

                backend: str = 'sqlite',

                log: bool = True,

            ) -> 'torch_geometric.data.OnDiskDataset':

                r"""Converts the :class:`InMemoryDataset` to a :class:`OnDiskDataset`

                variant. Useful for distributed training and hardware instances with

                limited amount of shared memory.

                root (str, optional): Root directory where the dataset should be saved.

                    If set to :obj:`None`, will save the dataset in

                    :obj:`root/on_disk`.

                    Note that it is important to specify :obj:`root` to account for

                    different dataset splits. (optional: :obj:`None`)

                backend (str): The :class:`Database` backend to use.

                    (default: :obj:`"sqlite"`)

                log (bool, optional): Whether to print any console output while

                    processing the dataset. (default: :obj:`True`)

                """

                if root is None and (self.root is None or not osp.exists(self.root)):

                    raise ValueError(f"The root directory of "

                                     f"'{self.__class__.__name__}' is not specified. "

                                     f"Please pass in 'root' when creating on-disk "

                                     f"datasets from it.")

                root = root or osp.join(self.root, 'on_disk')

                in_memory_dataset = self

                ref_data = in_memory_dataset.get(0)

                if not isinstance(ref_data, Data):

                    raise NotImplementedError(

                        f"`{self.__class__.__name__}.to_on_disk_dataset()` is "

                        f"currently only supported on homogeneous graphs")

                # Parse the schema ====================================================

                schema: Dict[str, Any] = {}

                for key, value in ref_data.to_dict().items():

                    if isinstance(value, (int, float, str)):

                        schema[key] = value.__class__

                    elif isinstance(value, Tensor) and value.dim() == 0:

                        schema[key] = dict(dtype=value.dtype, size=(-1, ))

                    elif isinstance(value, Tensor):

                        size = list(value.size())

                        size[ref_data.__cat_dim__(key, value)] = -1

                        schema[key] = dict(dtype=value.dtype, size=tuple(size))

                    else:

                        schema[key] = object

                # Create the on-disk dataset ==========================================

                class OnDiskDataset(torch_geometric.data.OnDiskDataset):

                    def __init__(

                        self,

                        root: str,

                        transform: Optional[Callable] = None,

                    ):

                        super().__init__(

                            root=root,

                            transform=transform,

                            backend=backend,

                            schema=schema,

                        )

                    def process(self):

                        _iter = [

                            in_memory_dataset.get(i)

                            for i in in_memory_dataset.indices()

                        ]

                        if log:  # pragma: no cover

                            _iter = tqdm(_iter, desc='Converting to OnDiskDataset')

                        data_list: List[Data] = []

                        for i, data in enumerate(_iter):

                            data_list.append(data)

                            if i + 1 == len(in_memory_dataset) or (i + 1) % 1000 == 0:

                                self.extend(data_list)

                                data_list = []

                    def serialize(self, data: Data) -> Dict[str, Any]:

                        return data.to_dict()

                    def deserialize(self, data: Dict[str, Any]) -> Data:

                        return Data.from_dict(data)

                    def __repr__(self) -> str:

                        arg_repr = str(len(self)) if len(self) > 1 else ''

                        return (f'OnDisk{in_memory_dataset.__class__.__name__}('

                                f'{arg_repr})')

                return OnDiskDataset(root, transform=in_memory_dataset.transform)
