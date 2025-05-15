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
