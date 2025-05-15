#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of graph deep learning regressors
"""

import torch
import logging
import deepchem as dc
import numpy as np
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import aggr
from typing import List, Optional, Callable, Union, Tuple, Any
from tqdm import tqdm
from redxregressors import datasets, ml_featurization
import os
from pathlib import Path
from torch.nn import Linear
from redxregressors.utilities import seed_all

log = logging.getLogger(__name__)

torch.use_deterministic_algorithms(True)


def train_attentivefp_pyg(
    model, train_loader, optimizer, epochs: int = 100
) -> list[Any]:
    """
    Function to train the AttentiveFP model. The function takes a model, a training loader, an optimizer and the number of epochs to train the model for. The function returns the losses of the model during training.
    Args:
        model (redxregressors.graph_models.AttentiveFP): The model to train
        train_loader (torch_geometric.loader.dataloader.DataLoader): The training loader to train the model on
        optimizer (torch.optim.Adam): The optimizer to use to train the model
        epochs (int): The number of epochs to train the model for
    Returns:
        list[Any]: The losses of the model during training
    """
    # seed all random number generators
    seed_all()

    # get the device to train the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set the model to training mode
    model.train()

    # create a progress bar for the number of epochs
    pbar = tqdm(range(epochs))
    avg_mse = 1000.0
    losses = []

    # loop over the number of epochs
    for epoch in pbar:
        pbar.set_description(f"Epoch: {epoch}/{epochs} MSE: {avg_mse}")
        sum_loss = 0.0
        n_graphs = 0
        # loop over the training data in batches
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.mse_loss(out.flatten(), data.y)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * data.num_graphs
            n_graphs += data.num_graphs
        avg_mse = sum_loss / n_graphs
        losses.append(avg_mse)
    return losses


@torch.no_grad()
def test_attentivefp_pyg(model, test_loader) -> list[np.ndarray]:
    """
    Function to test the AttentiveFP model on a test set. The function takes a model and a test loader and returns the predictions of the model on the test set.
    Args:
        model (redxregressors.graph_models.AttentiveFP): The model to test
        test_loader (torch_geometric.loader.dataloader.DataLoader): The test loader to test the model on
    Returns:
        list[np.ndarray]: The predictions of the model on the test set
    """

    # predictions will be stored in a list
    predictions = []

    # get the device to train the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set the model to evaluation mode
    model.eval()
    model.to(device)

    # loop over the test data in batches
    for batch in test_loader:
        batch.to(device)
        predictions.append(
            model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            .detach()
            .numpy()
        )

    return predictions


def csv_to_attentivefp_dataset(
    path: str = os.getcwd(),
    csv_file: str = "data.csv",
    smiles_column: str = "smiles",
    target_column: Union[List[str], Tuple[str]] = ("target"),
    overwrite_existing: bool = False,
) -> dc.data.Dataset:
    """
    Function to convert a csv file to a PyTorch Geometric dataset using the AttentiveFP featurization. The function takes a path to the csv file, the name of the csv file,
    the name of the smiles column and the name of the target column. The function returns a PyTorch Geometric dataset.
    Args:
        path (str): The path to the csv file
        csv_file (str): The name of the csv file
        smiles_column (str): The name of the smiles column
        target_column (Union[List[str], Tuple[str]]): The name of the target column
        overwrite_existing (bool): Whether to overwrite the existing dataset
    Returns:
        dc.data.Dataset: The PyTorch Geometric dataset
    """
    seed_all()
    root = Path(path)
    csv_file = root.joinpath(csv_file)
    dataset = datasets.RedxCSVGraphMolDataSet(
        csv_file=str(csv_file.absolute()),
        smiles_column=smiles_column,
        property_columns=target_column,
        pre_transform=ml_featurization.GetAttentiveFPFeatures(),
        overwrite_existing=overwrite_existing,
    ).shuffle()
    return dataset


def gcn_train(
    model: torch.nn.Module,
    train_set: torch_geometric.loader.dataloader.DataLoader,
    optimizer: torch.optim.Adam,
    epochs: int = 100,
    model_type: str = "regression",
) -> list:
    """
    Function to train a graph convolutional neural network model. The function takes a model, a training set, an optimizer and the number of epochs to train the model for. The function returns the trained model.
    This function has been heavily comments to help users understand how to train a model using PyTorch Geometric as this is the first graph model to be built at Redx.
    Args:
        model (redxregressors.graph_models.GCN): The model to train
        train_set (Union[List[torch_geometric.data.data.Data], torch_geometric.data.DataLoader]): The training data to train the model on. This can be a list of data objects or a DataLoader object.
        optimizer (torch.optim.adam.Adam): The optimizer to use to train the model
        epochs (int): The number of epochs to train the model for
        model_type (str): The type of model to train. Either regression or classification
    Returns:
        list: The losses of the model during training
    """

    seed_all()

    # Define the model to training mode
    model.train()

    # if CUDA GPU is avaliable use that otherwise use CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # use CUDA if available

    if model_type == "regression":
        # define mean squared error loss for regression
        loss_fx = torch.nn.MSELoss()
        loss_type = "MSE"

    elif model_type == "classification":
        # define cross entropy loss for classification
        loss_fx = torch.nn.CrossEntropyLoss()
        loss_type = "cross entropy"

    aggregated_loss = 0

    # store the training losses
    losses = []

    log.info("Starting training of the model")
    log.debug(f"Type of the train set{type(train_set)}")
    # Assumes as a dataloader object has been passed that the trianing data is in batches this could be batches of 1
    if isinstance(train_set, torch_geometric.loader.dataloader.DataLoader):
        log.info(
            f"Training on the whole data set in batches of data sized {train_set.batch_size} using the DataLoader object"
        )
        # create a progress bar for the number of epochs
        average_loss = 1000.0
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            aggregated_loss = 0.0
            number_of_examples = 0
            pbar.set_description(
                f"Epoch: {epoch:g}, Average {loss_type} loss: {average_loss:.4f}"
            )
            for data in tqdm(train_set):
                # send data to device
                data = data.to(device)
                # zero the gradients for the next step
                optimizer.zero_grad()
                # evaluate the model
                out = model(data)
                log.debug(f"Output shape {out.size()} {out}")
                log.debug(f"Data.y shape {data.y.size()} {data.y}")
                # compute the loss
                if out.shape[1] == 1:
                    loss = loss_fx(out.flatten(), data.y)
                else:
                    loss = loss_fx(out, data.y)
                # add the loss to the aggregate loss accounting for batch size
                number_of_examples += data.num_graphs
                aggregated_loss += float(loss) * data.num_graphs
                # compute the gradients by backpropagation
                loss.backward()
                # optimize the model
                optimizer.step()

            average_loss = aggregated_loss / number_of_examples
            losses.append(average_loss)

    else:
        log.error(
            f"ERROR - Training data must be of type DataLoader not {type(train_set)}"
        )
        raise ValueError

    return losses


@torch.no_grad()
def gcn_test(
    model: torch.nn.Module,
    test_set: Union[
        List[torch_geometric.data.data.Data], torch_geometric.data.DataLoader
    ],
) -> List[np.ndarray]:
    """
    Function to train a graph convolutional neural network model. The function takes a model, a training set, an optimizer and the number of epochs to train the model for. The function returns the trained model.
    This function has been heavily comments to help users understand how to train a model using PyTorch Geometric as this is the first graph model to be built at Redx.
    Args:
        model (redxregressors.graph_models.GCN): The model to train
        test_set (Union[List[torch_geometric.data.data.Data], torch_geometric.data.DataLoader]): The training data to train the model on. This can be a list of data objects or a DataLoader object.
    Returns:
        List[np.ndarray]: The trained model predictions
    """
    # Define the model to evaluation mode
    model.eval()
    torch.use_deterministic_algorithms(True)
    # if CUDA GPU is avaliable use that otherwise use CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # use CUDA if available

    predictions = []

    for data in tqdm(test_set):
        data = data.to(device)
        out = model(data)
        predictions.append(out.detach().numpy())

    return predictions


@torch.no_grad()
def gcn_get_representations(
    model: torch.nn.Module,
    data_set: Union[
        List[torch_geometric.data.data.Data], torch_geometric.data.DataLoader
    ],
) -> List[np.ndarray]:
    """
    Function to train a graph convolutional neural network model. The function takes a model, a training set, an optimizer and the number of epochs to train the model for. The function returns the trained model.
    This function has been heavily comments to help users understand how to train a model using PyTorch Geometric as this is the first graph model to be built at Redx.
    Args:
        model (redxregressors.graph_models.GCN): The model to train
        test_set (Union[List[torch_geometric.data.data.Data], torch_geometric.data.DataLoader]): The training data to train the model on. This can be a list of data objects or a DataLoader object.
    Returns:
        List[np.ndarray]: The trained model predictions
    """
    # Define the model to evaluation mode
    model.eval()

    # if CUDA GPU is avaliable use that otherwise use CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # use CUDA if available

    representations = []

    for data in tqdm(data_set):
        data = data.to(device)
        out = model.get_representations(data)
        representations.append(out)

    return representations


class GCN(torch.nn.Module):
    """
    A general class for graph convolutional neural network regressors
    """

    def __init__(
        self,
        n_input_node_feature: int = 13,
        n_ouputs: int = 1,
        n_convolutions: int = 2,
        gcn_hidden_dims: List[int] = [24, 24],
        n_linear: int = 2,
        linear_hidden_dims: List[int] = [12],
        activations: list[Callable] = [F.relu, F.relu, F.relu],
        pool_type: str = "mean",
        dropout: Optional[float] = 0.25,
    ) -> None:
        """
        Defines a general class for Graph convolutional Neural Networks (GCN). The class allows for N convolutional layers with independent activation functions and M linear layers with independent activation functions.
        The class also allows for different pooling types to be used. The class is designed to be flexible and allow for a wide range of models to be built. The class is designed to be used with PyTorch Geometric data objects.
        Note that dropout rates are applied after each layer using a consistent fraction.
        Args:
            n_input_node_feature (int): The number of input node features
            n_outputs (int): The number of outputs
            n_convolutions (int): The number of graph convolutional layers
            gcn_hidden_dims (List[int]): The hidden dimensions of the graph convolutional layers
            n_linear (int): The number of linear layers
            linear_hidden_dims (List[int]): The hidden dimensions of the linear layers
            activations (list[Callable]): The activation functions to use in the model
            pool_type (str): The type of pooling to use. Options are "mean", "sum" and "max"
            dropout (Optional[float]): The dropout rate to use in the model
        """
        super().__init__()

        seed_all()

        if len(gcn_hidden_dims) != n_convolutions:
            log.error(
                f"The number of gcn hidden dimensions {gcn_hidden_dims} must be equal to the number of convolutions {n_convolutions}"
            )
            raise ValueError

        if len(linear_hidden_dims) != n_linear - 1:
            log.error(
                f"The number of linear hidden dimensions {linear_hidden_dims} must be equal to the number of linear layers {n_linear}"
            )
            raise ValueError

        if len(activations) != n_convolutions + n_linear - 1:
            log.error(
                "The number of activations must be equal to the number of convolutions minus 1 as the last layer is the output layer"
            )
            raise ValueError

        self.activations = activations
        self.n_convolutions = n_convolutions
        self.gcn_hidden_dims = gcn_hidden_dims
        self.n_linear = n_linear
        self.linear_hidden_dims = linear_hidden_dims
        self.n_input_node_feature = n_input_node_feature
        self.n_outputs = n_ouputs
        self.pool_type = pool_type
        self.dropout = dropout

        self.conv0 = GCNConv(self.n_input_node_feature, gcn_hidden_dims[0])
        for ith in range(1, n_convolutions):
            log.debug(f"The ith graph convolution layer {ith}")
            setattr(
                self,
                f"conv{ith}",
                GCNConv(gcn_hidden_dims[ith - 1], gcn_hidden_dims[ith]),
            )

        if self.pool_type == "mean":
            self.global_pool = aggr.MeanAggregation()
        elif self.pool_type == "sum":
            self.global_pool = aggr.SumAggregation()
        elif self.pool_type == "max":
            self.global_pool = aggr.MaxAggregation()
        else:
            log.error(f"Pooling type {self.pool_type} is not supported")
            raise ValueError

        self.linear0 = Linear(gcn_hidden_dims[-1], linear_hidden_dims[0])
        for ith in range(1, n_linear):
            log.debug(f"The ith linear layer {ith}")
            if ith < self.n_linear - 1:
                setattr(
                    self,
                    f"linear{ith}",
                    Linear(linear_hidden_dims[ith - 1], linear_hidden_dims[ith]),
                )
            else:
                setattr(
                    self,
                    f"linear{ith}",
                    Linear(linear_hidden_dims[ith - 1], self.n_outputs),
                )

    def forward(self, data) -> Any:
        """
        Forward pass of the model. This is the main function that is called when the model is used to make predictions. The data object is passed in and the model is used to make predictions on this data object.
        Args:
            data (torch_geometric.data.Data): The data object to make predictions on
        Returns:
            Any: The output of the model
        """

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        offset_activations = 0

        # GCN layers to extract node features
        for ith in range(self.n_convolutions):
            x = self.activations[ith](
                getattr(self, f"conv{ith}")(x, edge_index, edge_attr)
            )
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)
            offset_activations += 1

        # pooling layer to collect node features into a single vector graph feature
        x = self.global_pool(x, data.batch).relu_()

        # Linear layers to make predictions
        for ith in range(self.n_linear):
            log.debug(
                f"Current activation offset {offset_activations} and current layer {ith} sum {offset_activations + ith}"
            )
            if ith < self.n_linear - 1:
                x = self.activations[offset_activations + ith](
                    getattr(self, f"linear{ith}")(x)
                )
                if self.dropout is not None:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = getattr(self, f"linear{ith}")(x)

        return x

    def get_representations(self, data) -> np.ndarray:
        """
        Function to get the representations of a graph. The function generates the node representations using the GCN layers the napplied pooling to produce a single graph representation.
        Args:
            data (torch_geometric.data.Data): The data object to get the representations of
        Returns:
            List[np.ndarray]: The representations of the graphs from the model
        """

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # GCN layers to extract node features
        for ith in range(self.n_convolutions):
            x = self.activations[ith](
                getattr(self, f"conv{ith}")(x, edge_index, edge_attr)
            )
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # pooling layer to collect node features into a single vector graph feature
        x = self.global_pool(x, data.batch).relu_()

        return x.detach().numpy()


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
