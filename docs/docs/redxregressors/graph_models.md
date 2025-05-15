# Module redxregressors.graph_models

Module of graph deep learning regressors

??? example "View Source"
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

## Variables

```python3
log
```

## Functions


### csv_to_attentivefp_dataset

```python3
def csv_to_attentivefp_dataset(
    path: str = '/Users/j.mcdonagh/Documents/Projects/Software/For_Open_Sourcing/Redx_regressors_os/docs',
    csv_file: str = 'data.csv',
    smiles_column: str = 'smiles',
    target_column: Union[List[str], Tuple[str]] = 'target',
    overwrite_existing: bool = False
) -> deepchem.data.datasets.Dataset
```

Function to convert a csv file to a PyTorch Geometric dataset using the AttentiveFP featurization. The function takes a path to the csv file, the name of the csv file,

the name of the smiles column and the name of the target column. The function returns a PyTorch Geometric dataset.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| path | str | The path to the csv file | None |
| csv_file | str | The name of the csv file | None |
| smiles_column | str | The name of the smiles column | None |
| target_column | Union[List[str], Tuple[str]] | The name of the target column | None |
| overwrite_existing | bool | Whether to overwrite the existing dataset | None |

**Returns:**

| Type | Description |
|---|---|
| dc.data.Dataset | The PyTorch Geometric dataset |

??? example "View Source"
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


### gcn_get_representations

```python3
def gcn_get_representations(
    model: torch.nn.modules.module.Module,
    data_set: Union[List[torch_geometric.data.data.Data], torch_geometric.deprecation.DataLoader]
) -> List[numpy.ndarray]
```

Function to train a graph convolutional neural network model. The function takes a model, a training set, an optimizer and the number of epochs to train the model for. The function returns the trained model.

This function has been heavily comments to help users understand how to train a model using PyTorch Geometric as this is the first graph model to be built at Redx.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model | redxregressors.graph_models.GCN | The model to train | None |
| test_set | Union[List[torch_geometric.data.data.Data], torch_geometric.data.DataLoader] | The training data to train the model on. This can be a list of data objects or a DataLoader object. | None |

**Returns:**

| Type | Description |
|---|---|
| List[np.ndarray] | The trained model predictions |

??? example "View Source"
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


### gcn_test

```python3
def gcn_test(
    model: torch.nn.modules.module.Module,
    test_set: Union[List[torch_geometric.data.data.Data], torch_geometric.deprecation.DataLoader]
) -> List[numpy.ndarray]
```

Function to train a graph convolutional neural network model. The function takes a model, a training set, an optimizer and the number of epochs to train the model for. The function returns the trained model.

This function has been heavily comments to help users understand how to train a model using PyTorch Geometric as this is the first graph model to be built at Redx.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model | redxregressors.graph_models.GCN | The model to train | None |
| test_set | Union[List[torch_geometric.data.data.Data], torch_geometric.data.DataLoader] | The training data to train the model on. This can be a list of data objects or a DataLoader object. | None |

**Returns:**

| Type | Description |
|---|---|
| List[np.ndarray] | The trained model predictions |

??? example "View Source"
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


### gcn_train

```python3
def gcn_train(
    model: torch.nn.modules.module.Module,
    train_set: torch_geometric.loader.dataloader.DataLoader,
    optimizer: torch.optim.adam.Adam,
    epochs: int = 100,
    model_type: str = 'regression'
) -> list
```

Function to train a graph convolutional neural network model. The function takes a model, a training set, an optimizer and the number of epochs to train the model for. The function returns the trained model.

This function has been heavily comments to help users understand how to train a model using PyTorch Geometric as this is the first graph model to be built at Redx.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model | redxregressors.graph_models.GCN | The model to train | None |
| train_set | Union[List[torch_geometric.data.data.Data], torch_geometric.data.DataLoader] | The training data to train the model on. This can be a list of data objects or a DataLoader object. | None |
| optimizer | torch.optim.adam.Adam | The optimizer to use to train the model | None |
| epochs | int | The number of epochs to train the model for | None |
| model_type | str | The type of model to train. Either regression or classification | None |

**Returns:**

| Type | Description |
|---|---|
| list | The losses of the model during training |

??? example "View Source"
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


### test_attentivefp_pyg

```python3
def test_attentivefp_pyg(
    model,
    test_loader
) -> list[numpy.ndarray]
```

Function to test the AttentiveFP model on a test set. The function takes a model and a test loader and returns the predictions of the model on the test set.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model | redxregressors.graph_models.AttentiveFP | The model to test | None |
| test_loader | torch_geometric.loader.dataloader.DataLoader | The test loader to test the model on | None |

**Returns:**

| Type | Description |
|---|---|
| list[np.ndarray] | The predictions of the model on the test set |

??? example "View Source"
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


### train_attentivefp_pyg

```python3
def train_attentivefp_pyg(
    model,
    train_loader,
    optimizer,
    epochs: int = 100
) -> list[typing.Any]
```

Function to train the AttentiveFP model. The function takes a model, a training loader, an optimizer and the number of epochs to train the model for. The function returns the losses of the model during training.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model | redxregressors.graph_models.AttentiveFP | The model to train | None |
| train_loader | torch_geometric.loader.dataloader.DataLoader | The training loader to train the model on | None |
| optimizer | torch.optim.Adam | The optimizer to use to train the model | None |
| epochs | int | The number of epochs to train the model for | None |

**Returns:**

| Type | Description |
|---|---|
| list[Any] | The losses of the model during training |

??? example "View Source"
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

## Classes

### GCN

```python3
class GCN(
    n_input_node_feature: int = 13,
    n_ouputs: int = 1,
    n_convolutions: int = 2,
    gcn_hidden_dims: List[int] = [24, 24],
    n_linear: int = 2,
    linear_hidden_dims: List[int] = [12],
    activations: list[typing.Callable] = [<function relu at 0x12289d6c0>, <function relu at 0x12289d6c0>, <function relu at 0x12289d6c0>],
    pool_type: str = 'mean',
    dropout: Optional[float] = 0.25
)
```

A general class for graph convolutional neural network regressors

#### Ancestors (in MRO)

* torch.nn.modules.module.Module

#### Class variables

```python3
T_destination
```

```python3
call_super_init
```

```python3
dump_patches
```

#### Methods


#### add_module

```python3
def add_module(
    self,
    name: str,
    module: Optional[ForwardRef('Module')]
) -> None
```

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| name | str | name of the child module. The child module can be<br>accessed from this module using the given name | None |
| module | Module | child module to be added to the module. | None |

??? example "View Source"
            def add_module(self, name: str, module: Optional["Module"]) -> None:

                r"""Add a child module to the current module.

                The module can be accessed as an attribute using the given name.

                Args:

                    name (str): name of the child module. The child module can be

                        accessed from this module using the given name

                    module (Module): child module to be added to the module.

                """

                if not isinstance(module, Module) and module is not None:

                    raise TypeError(f"{torch.typename(module)} is not a Module subclass")

                elif not isinstance(name, str):

                    raise TypeError(

                        f"module name should be a string. Got {torch.typename(name)}"

                    )

                elif hasattr(self, name) and name not in self._modules:

                    raise KeyError(f"attribute '{name}' already exists")

                elif "." in name:

                    raise KeyError(f'module name can\'t contain ".", got: {name}')

                elif name == "":

                    raise KeyError('module name can\'t be empty string ""')

                for hook in _global_module_registration_hooks.values():

                    output = hook(self, name, module)

                    if output is not None:

                        module = output

                self._modules[name] = module


#### apply

```python3
def apply(
    self: ~T,
    fn: Callable[[ForwardRef('Module')], NoneType]
) -> ~T
```

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| fn ( | None | class:`Module` -> None): function to be applied to each submodule | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def apply(self: T, fn: Callable[["Module"], None]) -> T:

                r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

                Typical use includes initializing the parameters of a model

                (see also :ref:`nn-init-doc`).

                Args:

                    fn (:class:`Module` -> None): function to be applied to each submodule

                Returns:

                    Module: self

                Example::

                    >>> @torch.no_grad()

                    >>> def init_weights(m):

                    >>>     print(m)

                    >>>     if type(m) == nn.Linear:

                    >>>         m.weight.fill_(1.0)

                    >>>         print(m.weight)

                    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

                    >>> net.apply(init_weights)

                    Linear(in_features=2, out_features=2, bias=True)

                    Parameter containing:

                    tensor([[1., 1.],

                            [1., 1.]], requires_grad=True)

                    Linear(in_features=2, out_features=2, bias=True)

                    Parameter containing:

                    tensor([[1., 1.],

                            [1., 1.]], requires_grad=True)

                    Sequential(

                      (0): Linear(in_features=2, out_features=2, bias=True)

                      (1): Linear(in_features=2, out_features=2, bias=True)

                    )

                """

                for module in self.children():

                    module.apply(fn)

                fn(self)

                return self


#### bfloat16

```python3
def bfloat16(
    self: ~T
) -> ~T
```

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def bfloat16(self: T) -> T:

                r"""Casts all floating point parameters and buffers to ``bfloat16`` datatype.

                .. note::

                    This method modifies the module in-place.

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)


#### buffers

```python3
def buffers(
    self,
    recurse: bool = True
) -> collections.abc.Iterator[torch.Tensor]
```

Return an iterator over module buffers.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| recurse | bool | if True, then yields buffers of this module<br>and all submodules. Otherwise, yields only buffers that<br>are direct members of this module. | None |

**Yields:**

| Type | Description |
|---|---|
| torch.Tensor | module buffer |

??? example "View Source"
            def buffers(self, recurse: bool = True) -> Iterator[Tensor]:

                r"""Return an iterator over module buffers.

                Args:

                    recurse (bool): if True, then yields buffers of this module

                        and all submodules. Otherwise, yields only buffers that

                        are direct members of this module.

                Yields:

                    torch.Tensor: module buffer

                Example::

                    >>> # xdoctest: +SKIP("undefined vars")

                    >>> for buf in model.buffers():

                    >>>     print(type(buf), buf.size())

                    <class 'torch.Tensor'> (20L,)

                    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

                """

                for _, buf in self.named_buffers(recurse=recurse):

                    yield buf


#### children

```python3
def children(
    self
) -> collections.abc.Iterator['Module']
```

Return an iterator over immediate children modules.

**Yields:**

| Type | Description |
|---|---|
| Module | a child module |

??? example "View Source"
            def children(self) -> Iterator["Module"]:

                r"""Return an iterator over immediate children modules.

                Yields:

                    Module: a child module

                """

                for _name, module in self.named_children():

                    yield module


#### compile

```python3
def compile(
    self,
    *args,
    **kwargs
)
```

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

??? example "View Source"
            def compile(self, *args, **kwargs):

                """

                Compile this Module's forward using :func:`torch.compile`.

                This Module's `__call__` method is compiled and all arguments are passed as-is

                to :func:`torch.compile`.

                See :func:`torch.compile` for details on the arguments for this function.

                """

                self._compiled_call_impl = torch.compile(self._call_impl, *args, **kwargs)


#### cpu

```python3
def cpu(
    self: ~T
) -> ~T
```

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def cpu(self: T) -> T:

                r"""Move all model parameters and buffers to the CPU.

                .. note::

                    This method modifies the module in-place.

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.cpu())


#### cuda

```python3
def cuda(
    self: ~T,
    device: Union[int, torch.device, NoneType] = None
) -> ~T
```

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| device | int | if specified, all parameters will be<br>copied to that device | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:

                r"""Move all model parameters and buffers to the GPU.

                This also makes associated parameters and buffers different objects. So

                it should be called before constructing the optimizer if the module will

                live on GPU while being optimized.

                .. note::

                    This method modifies the module in-place.

                Args:

                    device (int, optional): if specified, all parameters will be

                        copied to that device

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.cuda(device))


#### double

```python3
def double(
    self: ~T
) -> ~T
```

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def double(self: T) -> T:

                r"""Casts all floating point parameters and buffers to ``double`` datatype.

                .. note::

                    This method modifies the module in-place.

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.double() if t.is_floating_point() else t)


#### eval

```python3
def eval(
    self: ~T
) -> ~T
```

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def eval(self: T) -> T:

                r"""Set the module in evaluation mode.

                This has an effect only on certain modules. See the documentation of

                particular modules for details of their behaviors in training/evaluation

                mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,

                etc.

                This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

                See :ref:`locally-disable-grad-doc` for a comparison between

                `.eval()` and several similar mechanisms that may be confused with it.

                Returns:

                    Module: self

                """

                return self.train(False)


#### extra_repr

```python3
def extra_repr(
    self
) -> str
```

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

??? example "View Source"
            def extra_repr(self) -> str:

                r"""Return the extra representation of the module.

                To print customized extra information, you should re-implement

                this method in your own modules. Both single-line and multi-line

                strings are acceptable.

                """

                return ""


#### float

```python3
def float(
    self: ~T
) -> ~T
```

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def float(self: T) -> T:

                r"""Casts all floating point parameters and buffers to ``float`` datatype.

                .. note::

                    This method modifies the module in-place.

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.float() if t.is_floating_point() else t)


#### forward

```python3
def forward(
    self,
    data
) -> Any
```

Forward pass of the model. This is the main function that is called when the model is used to make predictions. The data object is passed in and the model is used to make predictions on this data object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data | torch_geometric.data.Data | The data object to make predictions on | None |

**Returns:**

| Type | Description |
|---|---|
| Any | The output of the model |

??? example "View Source"
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


#### get_buffer

```python3
def get_buffer(
    self,
    target: str
) -> 'Tensor'
```

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| target | None | The fully-qualified string name of the buffer<br>to look for. (See ``get_submodule`` for how to specify a<br>fully-qualified string.) | None |

**Returns:**

| Type | Description |
|---|---|
| torch.Tensor | The buffer referenced by ``target`` |

**Raises:**

| Type | Description |
|---|---|
| AttributeError | If the target string references an invalid<br>path or resolves to something that is not a<br>buffer |

??? example "View Source"
            def get_buffer(self, target: str) -> "Tensor":

                """Return the buffer given by ``target`` if it exists, otherwise throw an error.

                See the docstring for ``get_submodule`` for a more detailed

                explanation of this method's functionality as well as how to

                correctly specify ``target``.

                Args:

                    target: The fully-qualified string name of the buffer

                        to look for. (See ``get_submodule`` for how to specify a

                        fully-qualified string.)

                Returns:

                    torch.Tensor: The buffer referenced by ``target``

                Raises:

                    AttributeError: If the target string references an invalid

                        path or resolves to something that is not a

                        buffer

                """

                module_path, _, buffer_name = target.rpartition(".")

                mod: torch.nn.Module = self.get_submodule(module_path)

                if not hasattr(mod, buffer_name):

                    raise AttributeError(

                        mod._get_name() + " has no attribute `" + buffer_name + "`"

                    )

                buffer: torch.Tensor = getattr(mod, buffer_name)

                if buffer_name not in mod._buffers:

                    raise AttributeError("`" + buffer_name + "` is not a buffer")

                return buffer


#### get_extra_state

```python3
def get_extra_state(
    self
) -> Any
```

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

**Returns:**

| Type | Description |
|---|---|
| object | Any extra state to store in the module's state_dict |

??? example "View Source"
            def get_extra_state(self) -> Any:

                """Return any extra state to include in the module's state_dict.

                Implement this and a corresponding :func:`set_extra_state` for your module

                if you need to store extra state. This function is called when building the

                module's `state_dict()`.

                Note that extra state should be picklable to ensure working serialization

                of the state_dict. We only provide backwards compatibility guarantees

                for serializing Tensors; other objects may break backwards compatibility if

                their serialized pickled form changes.

                Returns:

                    object: Any extra state to store in the module's state_dict

                """

                raise RuntimeError(

                    "Reached a code path in Module.get_extra_state() that should never be called. "

                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "

                    "to report this bug."

                )


#### get_parameter

```python3
def get_parameter(
    self,
    target: str
) -> 'Parameter'
```

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| target | None | The fully-qualified string name of the Parameter<br>to look for. (See ``get_submodule`` for how to specify a<br>fully-qualified string.) | None |

**Returns:**

| Type | Description |
|---|---|
| torch.nn.Parameter | The Parameter referenced by ``target`` |

**Raises:**

| Type | Description |
|---|---|
| AttributeError | If the target string references an invalid<br>path or resolves to something that is not an<br>``nn.Parameter`` |

??? example "View Source"
            def get_parameter(self, target: str) -> "Parameter":

                """Return the parameter given by ``target`` if it exists, otherwise throw an error.

                See the docstring for ``get_submodule`` for a more detailed

                explanation of this method's functionality as well as how to

                correctly specify ``target``.

                Args:

                    target: The fully-qualified string name of the Parameter

                        to look for. (See ``get_submodule`` for how to specify a

                        fully-qualified string.)

                Returns:

                    torch.nn.Parameter: The Parameter referenced by ``target``

                Raises:

                    AttributeError: If the target string references an invalid

                        path or resolves to something that is not an

                        ``nn.Parameter``

                """

                module_path, _, param_name = target.rpartition(".")

                mod: torch.nn.Module = self.get_submodule(module_path)

                if not hasattr(mod, param_name):

                    raise AttributeError(

                        mod._get_name() + " has no attribute `" + param_name + "`"

                    )

                param: torch.nn.Parameter = getattr(mod, param_name)

                if not isinstance(param, torch.nn.Parameter):

                    raise AttributeError("`" + param_name + "` is not an nn.Parameter")

                return param


#### get_representations

```python3
def get_representations(
    self,
    data
) -> numpy.ndarray
```

Function to get the representations of a graph. The function generates the node representations using the GCN layers the napplied pooling to produce a single graph representation.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data | torch_geometric.data.Data | The data object to get the representations of | None |

**Returns:**

| Type | Description |
|---|---|
| List[np.ndarray] | The representations of the graphs from the model |

??? example "View Source"
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


#### get_submodule

```python3
def get_submodule(
    self,
    target: str
) -> 'Module'
```

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| target | None | The fully-qualified string name of the submodule<br>to look for. (See above example for how to specify a<br>fully-qualified string.) | None |

**Returns:**

| Type | Description |
|---|---|
| torch.nn.Module | The submodule referenced by ``target`` |

**Raises:**

| Type | Description |
|---|---|
| AttributeError | If at any point along the path resulting from<br>the target string the (sub)path resolves to a non-existent<br>attribute name or an object that is not an instance of ``nn.Module``. |

??? example "View Source"
            def get_submodule(self, target: str) -> "Module":

                """Return the submodule given by ``target`` if it exists, otherwise throw an error.

                For example, let's say you have an ``nn.Module`` ``A`` that

                looks like this:

                .. code-block:: text

                    A(

                        (net_b): Module(

                            (net_c): Module(

                                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))

                            )

                            (linear): Linear(in_features=100, out_features=200, bias=True)

                        )

                    )

                (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested

                submodule ``net_b``, which itself has two submodules ``net_c``

                and ``linear``. ``net_c`` then has a submodule ``conv``.)

                To check whether or not we have the ``linear`` submodule, we

                would call ``get_submodule("net_b.linear")``. To check whether

                we have the ``conv`` submodule, we would call

                ``get_submodule("net_b.net_c.conv")``.

                The runtime of ``get_submodule`` is bounded by the degree

                of module nesting in ``target``. A query against

                ``named_modules`` achieves the same result, but it is O(N) in

                the number of transitive modules. So, for a simple check to see

                if some submodule exists, ``get_submodule`` should always be

                used.

                Args:

                    target: The fully-qualified string name of the submodule

                        to look for. (See above example for how to specify a

                        fully-qualified string.)

                Returns:

                    torch.nn.Module: The submodule referenced by ``target``

                Raises:

                    AttributeError: If at any point along the path resulting from

                        the target string the (sub)path resolves to a non-existent

                        attribute name or an object that is not an instance of ``nn.Module``.

                """

                if target == "":

                    return self

                atoms: list[str] = target.split(".")

                mod: torch.nn.Module = self

                for item in atoms:

                    if not hasattr(mod, item):

                        raise AttributeError(

                            mod._get_name() + " has no attribute `" + item + "`"

                        )

                    mod = getattr(mod, item)

                    if not isinstance(mod, torch.nn.Module):

                        raise AttributeError("`" + item + "` is not an nn.Module")

                return mod


#### half

```python3
def half(
    self: ~T
) -> ~T
```

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def half(self: T) -> T:

                r"""Casts all floating point parameters and buffers to ``half`` datatype.

                .. note::

                    This method modifies the module in-place.

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.half() if t.is_floating_point() else t)


#### ipu

```python3
def ipu(
    self: ~T,
    device: Union[int, torch.device, NoneType] = None
) -> ~T
```

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| device | int | if specified, all parameters will be<br>copied to that device | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:

                r"""Move all model parameters and buffers to the IPU.

                This also makes associated parameters and buffers different objects. So

                it should be called before constructing the optimizer if the module will

                live on IPU while being optimized.

                .. note::

                    This method modifies the module in-place.

                Arguments:

                    device (int, optional): if specified, all parameters will be

                        copied to that device

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.ipu(device))


#### load_state_dict

```python3
def load_state_dict(
    self,
    state_dict: collections.abc.Mapping[str, typing.Any],
    strict: bool = True,
    assign: bool = False
)
```

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| state_dict | dict | a dict containing parameters and<br>persistent buffers. | None |
| strict | bool | whether to strictly enforce that the keys<br>in :attr:`state_dict` match the keys returned by this module's<br>:meth:`~torch.nn.Module.state_dict` function. Default: ``True`` | None |
| assign | bool | When set to ``False``, the properties of the tensors<br>in the current module are preserved whereas setting it to ``True`` preserves<br>properties of the Tensors in the state dict. The only<br>exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s<br>for which the value from the module is preserved.<br>Default: ``False`` | None |

**Returns:**

| Type | Description |
|---|---|
| None | ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:<br>* **missing_keys** is a list of str containing any keys that are expected<br>    by this module but missing from the provided ``state_dict``.<br>* **unexpected_keys** is a list of str containing the keys that are not<br>    expected by this module but present in the provided ``state_dict``. |

??? example "View Source"
            def load_state_dict(

                self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False

            ):

                r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

                If :attr:`strict` is ``True``, then

                the keys of :attr:`state_dict` must exactly match the keys returned

                by this module's :meth:`~torch.nn.Module.state_dict` function.

                .. warning::

                    If :attr:`assign` is ``True`` the optimizer must be created after

                    the call to :attr:`load_state_dict` unless

                    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

                Args:

                    state_dict (dict): a dict containing parameters and

                        persistent buffers.

                    strict (bool, optional): whether to strictly enforce that the keys

                        in :attr:`state_dict` match the keys returned by this module's

                        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

                    assign (bool, optional): When set to ``False``, the properties of the tensors

                        in the current module are preserved whereas setting it to ``True`` preserves

                        properties of the Tensors in the state dict. The only

                        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s

                        for which the value from the module is preserved.

                        Default: ``False``

                Returns:

                    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:

                        * **missing_keys** is a list of str containing any keys that are expected

                            by this module but missing from the provided ``state_dict``.

                        * **unexpected_keys** is a list of str containing the keys that are not

                            expected by this module but present in the provided ``state_dict``.

                Note:

                    If a parameter or buffer is registered as ``None`` and its corresponding key

                    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a

                    ``RuntimeError``.

                """

                if not isinstance(state_dict, Mapping):

                    raise TypeError(

                        f"Expected state_dict to be dict-like, got {type(state_dict)}."

                    )

                missing_keys: list[str] = []

                unexpected_keys: list[str] = []

                error_msgs: list[str] = []

                # copy state_dict so _load_from_state_dict can modify it

                metadata = getattr(state_dict, "_metadata", None)

                state_dict = OrderedDict(state_dict)

                if metadata is not None:

                    # mypy isn't aware that "_metadata" exists in state_dict

                    state_dict._metadata = metadata  # type: ignore[attr-defined]

                def load(module, local_state_dict, prefix=""):

                    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

                    if assign:

                        local_metadata["assign_to_params_buffers"] = assign

                    module._load_from_state_dict(

                        local_state_dict,

                        prefix,

                        local_metadata,

                        True,

                        missing_keys,

                        unexpected_keys,

                        error_msgs,

                    )

                    for name, child in module._modules.items():

                        if child is not None:

                            child_prefix = prefix + name + "."

                            child_state_dict = {

                                k: v

                                for k, v in local_state_dict.items()

                                if k.startswith(child_prefix)

                            }

                            load(child, child_state_dict, child_prefix)  # noqa: F821

                    # Note that the hook can modify missing_keys and unexpected_keys.

                    incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)

                    for hook in module._load_state_dict_post_hooks.values():

                        out = hook(module, incompatible_keys)

                        assert out is None, (

                            "Hooks registered with ``register_load_state_dict_post_hook`` are not"

                            "expected to return new values, if incompatible_keys need to be modified,"

                            "it should be done inplace."

                        )

                load(self, state_dict)

                del load

                if strict:

                    if len(unexpected_keys) > 0:

                        error_msgs.insert(

                            0,

                            "Unexpected key(s) in state_dict: {}. ".format(

                                ", ".join(f'"{k}"' for k in unexpected_keys)

                            ),

                        )

                    if len(missing_keys) > 0:

                        error_msgs.insert(

                            0,

                            "Missing key(s) in state_dict: {}. ".format(

                                ", ".join(f'"{k}"' for k in missing_keys)

                            ),

                        )

                if len(error_msgs) > 0:

                    raise RuntimeError(

                        "Error(s) in loading state_dict for {}:\n\t{}".format(

                            self.__class__.__name__, "\n\t".join(error_msgs)

                        )

                    )

                return _IncompatibleKeys(missing_keys, unexpected_keys)


#### modules

```python3
def modules(
    self
) -> collections.abc.Iterator['Module']
```

Return an iterator over all modules in the network.

**Yields:**

| Type | Description |
|---|---|
| Module | a module in the network |

??? example "View Source"
            def modules(self) -> Iterator["Module"]:

                r"""Return an iterator over all modules in the network.

                Yields:

                    Module: a module in the network

                Note:

                    Duplicate modules are returned only once. In the following

                    example, ``l`` will be returned only once.

                Example::

                    >>> l = nn.Linear(2, 2)

                    >>> net = nn.Sequential(l, l)

                    >>> for idx, m in enumerate(net.modules()):

                    ...     print(idx, '->', m)

                    0 -> Sequential(

                      (0): Linear(in_features=2, out_features=2, bias=True)

                      (1): Linear(in_features=2, out_features=2, bias=True)

                    )

                    1 -> Linear(in_features=2, out_features=2, bias=True)

                """

                for _, module in self.named_modules():

                    yield module


#### mtia

```python3
def mtia(
    self: ~T,
    device: Union[int, torch.device, NoneType] = None
) -> ~T
```

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| device | int | if specified, all parameters will be<br>copied to that device | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def mtia(self: T, device: Optional[Union[int, device]] = None) -> T:

                r"""Move all model parameters and buffers to the MTIA.

                This also makes associated parameters and buffers different objects. So

                it should be called before constructing the optimizer if the module will

                live on MTIA while being optimized.

                .. note::

                    This method modifies the module in-place.

                Arguments:

                    device (int, optional): if specified, all parameters will be

                        copied to that device

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.mtia(device))


#### named_buffers

```python3
def named_buffers(
    self,
    prefix: str = '',
    recurse: bool = True,
    remove_duplicate: bool = True
) -> collections.abc.Iterator[tuple[str, torch.Tensor]]
```

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| prefix | str | prefix to prepend to all buffer names. | None |
| recurse | bool | if True, then yields buffers of this module<br>and all submodules. Otherwise, yields only buffers that<br>are direct members of this module. Defaults to True. | None |
| remove_duplicate | bool | whether to remove the duplicated buffers in the result. Defaults to True. | True |

**Yields:**

| Type | Description |
|---|---|
| None | (str, torch.Tensor): Tuple containing the name and buffer |

??? example "View Source"
            def named_buffers(

                self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True

            ) -> Iterator[tuple[str, Tensor]]:

                r"""Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

                Args:

                    prefix (str): prefix to prepend to all buffer names.

                    recurse (bool, optional): if True, then yields buffers of this module

                        and all submodules. Otherwise, yields only buffers that

                        are direct members of this module. Defaults to True.

                    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

                Yields:

                    (str, torch.Tensor): Tuple containing the name and buffer

                Example::

                    >>> # xdoctest: +SKIP("undefined vars")

                    >>> for name, buf in self.named_buffers():

                    >>>     if name in ['running_var']:

                    >>>         print(buf.size())

                """

                gen = self._named_members(

                    lambda module: module._buffers.items(),

                    prefix=prefix,

                    recurse=recurse,

                    remove_duplicate=remove_duplicate,

                )

                yield from gen


#### named_children

```python3
def named_children(
    self
) -> collections.abc.Iterator[tuple[str, 'Module']]
```

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

**Yields:**

| Type | Description |
|---|---|
| None | (str, Module): Tuple containing a name and child module |

??? example "View Source"
            def named_children(self) -> Iterator[tuple[str, "Module"]]:

                r"""Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

                Yields:

                    (str, Module): Tuple containing a name and child module

                Example::

                    >>> # xdoctest: +SKIP("undefined vars")

                    >>> for name, module in model.named_children():

                    >>>     if name in ['conv4', 'conv5']:

                    >>>         print(module)

                """

                memo = set()

                for name, module in self._modules.items():

                    if module is not None and module not in memo:

                        memo.add(module)

                        yield name, module


#### named_modules

```python3
def named_modules(
    self,
    memo: Optional[set['Module']] = None,
    prefix: str = '',
    remove_duplicate: bool = True
)
```

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| memo | None | a memo to store the set of modules already added to the result | None |
| prefix | None | a prefix that will be added to the name of the module | None |
| remove_duplicate | None | whether to remove the duplicated module instances in the result<br>or not | None |

**Yields:**

| Type | Description |
|---|---|
| None | (str, Module): Tuple of name and module |

??? example "View Source"
            def named_modules(

                self,

                memo: Optional[set["Module"]] = None,

                prefix: str = "",

                remove_duplicate: bool = True,

            ):

                r"""Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

                Args:

                    memo: a memo to store the set of modules already added to the result

                    prefix: a prefix that will be added to the name of the module

                    remove_duplicate: whether to remove the duplicated module instances in the result

                        or not

                Yields:

                    (str, Module): Tuple of name and module

                Note:

                    Duplicate modules are returned only once. In the following

                    example, ``l`` will be returned only once.

                Example::

                    >>> l = nn.Linear(2, 2)

                    >>> net = nn.Sequential(l, l)

                    >>> for idx, m in enumerate(net.named_modules()):

                    ...     print(idx, '->', m)

                    0 -> ('', Sequential(

                      (0): Linear(in_features=2, out_features=2, bias=True)

                      (1): Linear(in_features=2, out_features=2, bias=True)

                    ))

                    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

                """

                if memo is None:

                    memo = set()

                if self not in memo:

                    if remove_duplicate:

                        memo.add(self)

                    yield prefix, self

                    for name, module in self._modules.items():

                        if module is None:

                            continue

                        submodule_prefix = prefix + ("." if prefix else "") + name

                        yield from module.named_modules(

                            memo, submodule_prefix, remove_duplicate

                        )


#### named_parameters

```python3
def named_parameters(
    self,
    prefix: str = '',
    recurse: bool = True,
    remove_duplicate: bool = True
) -> collections.abc.Iterator[tuple[str, torch.nn.parameter.Parameter]]
```

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| prefix | str | prefix to prepend to all parameter names. | None |
| recurse | bool | if True, then yields parameters of this module<br>and all submodules. Otherwise, yields only parameters that<br>are direct members of this module. | None |
| remove_duplicate | bool | whether to remove the duplicated<br>parameters in the result. Defaults to True. | None |

**Yields:**

| Type | Description |
|---|---|
| None | (str, Parameter): Tuple containing the name and parameter |

??? example "View Source"
            def named_parameters(

                self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True

            ) -> Iterator[tuple[str, Parameter]]:

                r"""Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

                Args:

                    prefix (str): prefix to prepend to all parameter names.

                    recurse (bool): if True, then yields parameters of this module

                        and all submodules. Otherwise, yields only parameters that

                        are direct members of this module.

                    remove_duplicate (bool, optional): whether to remove the duplicated

                        parameters in the result. Defaults to True.

                Yields:

                    (str, Parameter): Tuple containing the name and parameter

                Example::

                    >>> # xdoctest: +SKIP("undefined vars")

                    >>> for name, param in self.named_parameters():

                    >>>     if name in ['bias']:

                    >>>         print(param.size())

                """

                gen = self._named_members(

                    lambda module: module._parameters.items(),

                    prefix=prefix,

                    recurse=recurse,

                    remove_duplicate=remove_duplicate,

                )

                yield from gen


#### parameters

```python3
def parameters(
    self,
    recurse: bool = True
) -> collections.abc.Iterator[torch.nn.parameter.Parameter]
```

Return an iterator over module parameters.

This is typically passed to an optimizer.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| recurse | bool | if True, then yields parameters of this module<br>and all submodules. Otherwise, yields only parameters that<br>are direct members of this module. | None |

**Yields:**

| Type | Description |
|---|---|
| Parameter | module parameter |

??? example "View Source"
            def parameters(self, recurse: bool = True) -> Iterator[Parameter]:

                r"""Return an iterator over module parameters.

                This is typically passed to an optimizer.

                Args:

                    recurse (bool): if True, then yields parameters of this module

                        and all submodules. Otherwise, yields only parameters that

                        are direct members of this module.

                Yields:

                    Parameter: module parameter

                Example::

                    >>> # xdoctest: +SKIP("undefined vars")

                    >>> for param in model.parameters():

                    >>>     print(type(param), param.size())

                    <class 'torch.Tensor'> (20L,)

                    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

                """

                for _name, param in self.named_parameters(recurse=recurse):

                    yield param


#### register_backward_hook

```python3
def register_backward_hook(
    self,
    hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]]
) -> torch.utils.hooks.RemovableHandle
```

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

**Returns:**

| Type | Description |
|---|---|
| None | :class:`torch.utils.hooks.RemovableHandle`:<br>a handle that can be used to remove the added hook by calling<br>``handle.remove()`` |

??? example "View Source"
            def register_backward_hook(

                self, hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]]

            ) -> RemovableHandle:

                r"""Register a backward hook on the module.

                This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and

                the behavior of this function will change in future versions.

                Returns:

                    :class:`torch.utils.hooks.RemovableHandle`:

                        a handle that can be used to remove the added hook by calling

                        ``handle.remove()``

                """

                if self._is_full_backward_hook is True:

                    raise RuntimeError(

                        "Cannot use both regular backward hooks and full backward hooks on a "

                        "single Module. Please use only one of them."

                    )

                self._is_full_backward_hook = False

                handle = RemovableHandle(self._backward_hooks)

                self._backward_hooks[handle.id] = hook

                return handle


#### register_buffer

```python3
def register_buffer(
    self,
    name: str,
    tensor: Optional[torch.Tensor],
    persistent: bool = True
) -> None
```

Add a buffer to the module.

This is typically used to register a buffer that should not to be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| name | str | name of the buffer. The buffer can be accessed<br>from this module using the given name | None |
| tensor | Tensor or None | buffer to be registered. If ``None``, then operations<br>that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,<br>the buffer is **not** included in the module's :attr:`state_dict`. | None |
| persistent | bool | whether the buffer is part of this module's<br>:attr:`state_dict`. | None |

??? example "View Source"
            def register_buffer(

                self, name: str, tensor: Optional[Tensor], persistent: bool = True

            ) -> None:

                r"""Add a buffer to the module.

                This is typically used to register a buffer that should not to be

                considered a model parameter. For example, BatchNorm's ``running_mean``

                is not a parameter, but is part of the module's state. Buffers, by

                default, are persistent and will be saved alongside parameters. This

                behavior can be changed by setting :attr:`persistent` to ``False``. The

                only difference between a persistent buffer and a non-persistent buffer

                is that the latter will not be a part of this module's

                :attr:`state_dict`.

                Buffers can be accessed as attributes using given names.

                Args:

                    name (str): name of the buffer. The buffer can be accessed

                        from this module using the given name

                    tensor (Tensor or None): buffer to be registered. If ``None``, then operations

                        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,

                        the buffer is **not** included in the module's :attr:`state_dict`.

                    persistent (bool): whether the buffer is part of this module's

                        :attr:`state_dict`.

                Example::

                    >>> # xdoctest: +SKIP("undefined vars")

                    >>> self.register_buffer('running_mean', torch.zeros(num_features))

                """

                if persistent is False and isinstance(self, torch.jit.ScriptModule):

                    raise RuntimeError("ScriptModule does not support non-persistent buffers")

                if "_buffers" not in self.__dict__:

                    raise AttributeError("cannot assign buffer before Module.__init__() call")

                elif not isinstance(name, str):

                    raise TypeError(

                        f"buffer name should be a string. Got {torch.typename(name)}"

                    )

                elif "." in name:

                    raise KeyError('buffer name can\'t contain "."')

                elif name == "":

                    raise KeyError('buffer name can\'t be empty string ""')

                elif hasattr(self, name) and name not in self._buffers:

                    raise KeyError(f"attribute '{name}' already exists")

                elif tensor is not None and not isinstance(tensor, torch.Tensor):

                    raise TypeError(

                        f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "

                        "(torch Tensor or None required)"

                    )

                else:

                    for hook in _global_buffer_registration_hooks.values():

                        output = hook(self, name, tensor)

                        if output is not None:

                            tensor = output

                    self._buffers[name] = tensor

                    if persistent:

                        self._non_persistent_buffers_set.discard(name)

                    else:

                        self._non_persistent_buffers_set.add(name)


#### register_forward_hook

```python3
def register_forward_hook(
    self,
    hook: Union[Callable[[~T, tuple[Any, ...], Any], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]]],
    *,
    prepend: bool = False,
    with_kwargs: bool = False,
    always_call: bool = False
) -> torch.utils.hooks.RemovableHandle
```

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| hook | Callable | The user defined hook to be registered. | None |
| prepend | bool | If ``True``, the provided ``hook`` will be fired<br>before all existing ``forward`` hooks on this<br>:class:`torch.nn.Module`. Otherwise, the provided<br>``hook`` will be fired after all existing ``forward`` hooks on<br>this :class:`torch.nn.Module`. Note that global<br>``forward`` hooks registered with<br>:func:`register_module_forward_hook` will fire before all hooks<br>registered by this method.<br>Default: ``False`` | None |
| with_kwargs | bool | If ``True``, the ``hook`` will be passed the<br>kwargs given to the forward function.<br>Default: ``False`` | None |
| always_call | bool | If ``True`` the ``hook`` will be run regardless of<br>whether an exception is raised while calling the Module.<br>Default: ``False`` | None |

**Returns:**

| Type | Description |
|---|---|
| None | :class:`torch.utils.hooks.RemovableHandle`:<br>a handle that can be used to remove the added hook by calling<br>``handle.remove()`` |

??? example "View Source"
            def register_forward_hook(

                self,

                hook: Union[

                    Callable[[T, tuple[Any, ...], Any], Optional[Any]],

                    Callable[[T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]],

                ],

                *,

                prepend: bool = False,

                with_kwargs: bool = False,

                always_call: bool = False,

            ) -> RemovableHandle:

                r"""Register a forward hook on the module.

                The hook will be called every time after :func:`forward` has computed an output.

                If ``with_kwargs`` is ``False`` or not specified, the input contains only

                the positional arguments given to the module. Keyword arguments won't be

                passed to the hooks and only to the ``forward``. The hook can modify the

                output. It can modify the input inplace but it will not have effect on

                forward since this is called after :func:`forward` is called. The hook

                should have the following signature::

                    hook(module, args, output) -> None or modified output

                If ``with_kwargs`` is ``True``, the forward hook will be passed the

                ``kwargs`` given to the forward function and be expected to return the

                output possibly modified. The hook should have the following signature::

                    hook(module, args, kwargs, output) -> None or modified output

                Args:

                    hook (Callable): The user defined hook to be registered.

                    prepend (bool): If ``True``, the provided ``hook`` will be fired

                        before all existing ``forward`` hooks on this

                        :class:`torch.nn.Module`. Otherwise, the provided

                        ``hook`` will be fired after all existing ``forward`` hooks on

                        this :class:`torch.nn.Module`. Note that global

                        ``forward`` hooks registered with

                        :func:`register_module_forward_hook` will fire before all hooks

                        registered by this method.

                        Default: ``False``

                    with_kwargs (bool): If ``True``, the ``hook`` will be passed the

                        kwargs given to the forward function.

                        Default: ``False``

                    always_call (bool): If ``True`` the ``hook`` will be run regardless of

                        whether an exception is raised while calling the Module.

                        Default: ``False``

                Returns:

                    :class:`torch.utils.hooks.RemovableHandle`:

                        a handle that can be used to remove the added hook by calling

                        ``handle.remove()``

                """

                handle = RemovableHandle(

                    self._forward_hooks,

                    extra_dict=[

                        self._forward_hooks_with_kwargs,

                        self._forward_hooks_always_called,

                    ],

                )

                self._forward_hooks[handle.id] = hook

                if with_kwargs:

                    self._forward_hooks_with_kwargs[handle.id] = True

                if always_call:

                    self._forward_hooks_always_called[handle.id] = True

                if prepend:

                    self._forward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]

                return handle


#### register_forward_pre_hook

```python3
def register_forward_pre_hook(
    self,
    hook: Union[Callable[[~T, tuple[Any, ...]], Optional[Any]], Callable[[~T, tuple[Any, ...], dict[str, Any]], Optional[tuple[Any, dict[str, Any]]]]],
    *,
    prepend: bool = False,
    with_kwargs: bool = False
) -> torch.utils.hooks.RemovableHandle
```

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.

If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| hook | Callable | The user defined hook to be registered. | None |
| prepend | bool | If true, the provided ``hook`` will be fired before<br>all existing ``forward_pre`` hooks on this<br>:class:`torch.nn.Module`. Otherwise, the provided<br>``hook`` will be fired after all existing ``forward_pre`` hooks<br>on this :class:`torch.nn.Module`. Note that global<br>``forward_pre`` hooks registered with<br>:func:`register_module_forward_pre_hook` will fire before all<br>hooks registered by this method.<br>Default: ``False`` | None |
| with_kwargs | bool | If true, the ``hook`` will be passed the kwargs<br>given to the forward function.<br>Default: ``False`` | None |

**Returns:**

| Type | Description |
|---|---|
| None | :class:`torch.utils.hooks.RemovableHandle`:<br>a handle that can be used to remove the added hook by calling<br>``handle.remove()`` |

??? example "View Source"
            def register_forward_pre_hook(

                self,

                hook: Union[

                    Callable[[T, tuple[Any, ...]], Optional[Any]],

                    Callable[

                        [T, tuple[Any, ...], dict[str, Any]],

                        Optional[tuple[Any, dict[str, Any]]],

                    ],

                ],

                *,

                prepend: bool = False,

                with_kwargs: bool = False,

            ) -> RemovableHandle:

                r"""Register a forward pre-hook on the module.

                The hook will be called every time before :func:`forward` is invoked.



                If ``with_kwargs`` is false or not specified, the input contains only

                the positional arguments given to the module. Keyword arguments won't be

                passed to the hooks and only to the ``forward``. The hook can modify the

                input. User can either return a tuple or a single modified value in the

                hook. We will wrap the value into a tuple if a single value is returned

                (unless that value is already a tuple). The hook should have the

                following signature::

                    hook(module, args) -> None or modified input

                If ``with_kwargs`` is true, the forward pre-hook will be passed the

                kwargs given to the forward function. And if the hook modifies the

                input, both the args and kwargs should be returned. The hook should have

                the following signature::

                    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

                Args:

                    hook (Callable): The user defined hook to be registered.

                    prepend (bool): If true, the provided ``hook`` will be fired before

                        all existing ``forward_pre`` hooks on this

                        :class:`torch.nn.Module`. Otherwise, the provided

                        ``hook`` will be fired after all existing ``forward_pre`` hooks

                        on this :class:`torch.nn.Module`. Note that global

                        ``forward_pre`` hooks registered with

                        :func:`register_module_forward_pre_hook` will fire before all

                        hooks registered by this method.

                        Default: ``False``

                    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs

                        given to the forward function.

                        Default: ``False``

                Returns:

                    :class:`torch.utils.hooks.RemovableHandle`:

                        a handle that can be used to remove the added hook by calling

                        ``handle.remove()``

                """

                handle = RemovableHandle(

                    self._forward_pre_hooks, extra_dict=self._forward_pre_hooks_with_kwargs

                )

                self._forward_pre_hooks[handle.id] = hook

                if with_kwargs:

                    self._forward_pre_hooks_with_kwargs[handle.id] = True

                if prepend:

                    self._forward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]

                return handle


#### register_full_backward_hook

```python3
def register_full_backward_hook(
    self,
    hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor], Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]],
    prepend: bool = False
) -> torch.utils.hooks.RemovableHandle
```

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module
are computed, i.e. the hook will execute if and only if the gradients with
respect to module outputs are computed. The hook should have the following
signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| hook | Callable | The user-defined hook to be registered. | None |
| prepend | bool | If true, the provided ``hook`` will be fired before<br>all existing ``backward`` hooks on this<br>:class:`torch.nn.Module`. Otherwise, the provided<br>``hook`` will be fired after all existing ``backward`` hooks on<br>this :class:`torch.nn.Module`. Note that global<br>``backward`` hooks registered with<br>:func:`register_module_full_backward_hook` will fire before<br>all hooks registered by this method. | None |

**Returns:**

| Type | Description |
|---|---|
| None | :class:`torch.utils.hooks.RemovableHandle`:<br>a handle that can be used to remove the added hook by calling<br>``handle.remove()`` |

??? example "View Source"
            def register_full_backward_hook(

                self,

                hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],

                prepend: bool = False,

            ) -> RemovableHandle:

                r"""Register a backward hook on the module.

                The hook will be called every time the gradients with respect to a module

                are computed, i.e. the hook will execute if and only if the gradients with

                respect to module outputs are computed. The hook should have the following

                signature::

                    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

                The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients

                with respect to the inputs and outputs respectively. The hook should

                not modify its arguments, but it can optionally return a new gradient with

                respect to the input that will be used in place of :attr:`grad_input` in

                subsequent computations. :attr:`grad_input` will only correspond to the inputs given

                as positional arguments and all kwarg arguments are ignored. Entries

                in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor

                arguments.

                For technical reasons, when this hook is applied to a Module, its forward function will

                receive a view of each Tensor passed to the Module. Similarly the caller will receive a view

                of each Tensor returned by the Module's forward function.

                .. warning ::

                    Modifying inputs or outputs inplace is not allowed when using backward hooks and

                    will raise an error.

                Args:

                    hook (Callable): The user-defined hook to be registered.

                    prepend (bool): If true, the provided ``hook`` will be fired before

                        all existing ``backward`` hooks on this

                        :class:`torch.nn.Module`. Otherwise, the provided

                        ``hook`` will be fired after all existing ``backward`` hooks on

                        this :class:`torch.nn.Module`. Note that global

                        ``backward`` hooks registered with

                        :func:`register_module_full_backward_hook` will fire before

                        all hooks registered by this method.

                Returns:

                    :class:`torch.utils.hooks.RemovableHandle`:

                        a handle that can be used to remove the added hook by calling

                        ``handle.remove()``

                """

                if self._is_full_backward_hook is False:

                    raise RuntimeError(

                        "Cannot use both regular backward hooks and full backward hooks on a "

                        "single Module. Please use only one of them."

                    )

                self._is_full_backward_hook = True

                handle = RemovableHandle(self._backward_hooks)

                self._backward_hooks[handle.id] = hook

                if prepend:

                    self._backward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]

                return handle


#### register_full_backward_pre_hook

```python3
def register_full_backward_pre_hook(
    self,
    hook: Callable[[ForwardRef('Module'), Union[tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, tuple[torch.Tensor, ...], torch.Tensor]],
    prepend: bool = False
) -> torch.utils.hooks.RemovableHandle
```

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| hook | Callable | The user-defined hook to be registered. | None |
| prepend | bool | If true, the provided ``hook`` will be fired before<br>all existing ``backward_pre`` hooks on this<br>:class:`torch.nn.Module`. Otherwise, the provided<br>``hook`` will be fired after all existing ``backward_pre`` hooks<br>on this :class:`torch.nn.Module`. Note that global<br>``backward_pre`` hooks registered with<br>:func:`register_module_full_backward_pre_hook` will fire before<br>all hooks registered by this method. | None |

**Returns:**

| Type | Description |
|---|---|
| None | :class:`torch.utils.hooks.RemovableHandle`:<br>a handle that can be used to remove the added hook by calling<br>``handle.remove()`` |

??? example "View Source"
            def register_full_backward_pre_hook(

                self,

                hook: Callable[["Module", _grad_t], Union[None, _grad_t]],

                prepend: bool = False,

            ) -> RemovableHandle:

                r"""Register a backward pre-hook on the module.

                The hook will be called every time the gradients for the module are computed.

                The hook should have the following signature::

                    hook(module, grad_output) -> tuple[Tensor] or None

                The :attr:`grad_output` is a tuple. The hook should

                not modify its arguments, but it can optionally return a new gradient with

                respect to the output that will be used in place of :attr:`grad_output` in

                subsequent computations. Entries in :attr:`grad_output` will be ``None`` for

                all non-Tensor arguments.

                For technical reasons, when this hook is applied to a Module, its forward function will

                receive a view of each Tensor passed to the Module. Similarly the caller will receive a view

                of each Tensor returned by the Module's forward function.

                .. warning ::

                    Modifying inputs inplace is not allowed when using backward hooks and

                    will raise an error.

                Args:

                    hook (Callable): The user-defined hook to be registered.

                    prepend (bool): If true, the provided ``hook`` will be fired before

                        all existing ``backward_pre`` hooks on this

                        :class:`torch.nn.Module`. Otherwise, the provided

                        ``hook`` will be fired after all existing ``backward_pre`` hooks

                        on this :class:`torch.nn.Module`. Note that global

                        ``backward_pre`` hooks registered with

                        :func:`register_module_full_backward_pre_hook` will fire before

                        all hooks registered by this method.

                Returns:

                    :class:`torch.utils.hooks.RemovableHandle`:

                        a handle that can be used to remove the added hook by calling

                        ``handle.remove()``

                """

                handle = RemovableHandle(self._backward_pre_hooks)

                self._backward_pre_hooks[handle.id] = hook

                if prepend:

                    self._backward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]

                return handle


#### register_load_state_dict_post_hook

```python3
def register_load_state_dict_post_hook(
    self,
    hook
)
```

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

**Returns:**

| Type | Description |
|---|---|
| None | :class:`torch.utils.hooks.RemovableHandle`:<br>a handle that can be used to remove the added hook by calling<br>``handle.remove()`` |

??? example "View Source"
            def register_load_state_dict_post_hook(self, hook):

                r"""Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

                It should have the following signature::

                    hook(module, incompatible_keys) -> None

                The ``module`` argument is the current module that this hook is registered

                on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting

                of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``

                is a ``list`` of ``str`` containing the missing keys and

                ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

                The given incompatible_keys can be modified inplace if needed.

                Note that the checks performed when calling :func:`load_state_dict` with

                ``strict=True`` are affected by modifications the hook makes to

                ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either

                set of keys will result in an error being thrown when ``strict=True``, and

                clearing out both missing and unexpected keys will avoid an error.

                Returns:

                    :class:`torch.utils.hooks.RemovableHandle`:

                        a handle that can be used to remove the added hook by calling

                        ``handle.remove()``

                """

                handle = RemovableHandle(self._load_state_dict_post_hooks)

                self._load_state_dict_post_hooks[handle.id] = hook

                return handle


#### register_load_state_dict_pre_hook

```python3
def register_load_state_dict_pre_hook(
    self,
    hook
)
```

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| hook | Callable | Callable hook that will be invoked before<br>loading the state dict. | None |

??? example "View Source"
            def register_load_state_dict_pre_hook(self, hook):

                r"""Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

                It should have the following signature::

                    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

                Arguments:

                    hook (Callable): Callable hook that will be invoked before

                        loading the state dict.

                """

                return self._register_load_state_dict_pre_hook(hook, with_module=True)


#### register_module

```python3
def register_module(
    self,
    name: str,
    module: Optional[ForwardRef('Module')]
) -> None
```

Alias for :func:`add_module`.

??? example "View Source"
            def register_module(self, name: str, module: Optional["Module"]) -> None:

                r"""Alias for :func:`add_module`."""

                self.add_module(name, module)


#### register_parameter

```python3
def register_parameter(
    self,
    name: str,
    param: Optional[torch.nn.parameter.Parameter]
) -> None
```

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| name | str | name of the parameter. The parameter can be accessed<br>from this module using the given name | None |
| param | Parameter or None | parameter to be added to the module. If<br>``None``, then operations that run on parameters, such as :attr:`cuda`,<br>are ignored. If ``None``, the parameter is **not** included in the<br>module's :attr:`state_dict`. | None |

??? example "View Source"
            def register_parameter(self, name: str, param: Optional[Parameter]) -> None:

                r"""Add a parameter to the module.

                The parameter can be accessed as an attribute using given name.

                Args:

                    name (str): name of the parameter. The parameter can be accessed

                        from this module using the given name

                    param (Parameter or None): parameter to be added to the module. If

                        ``None``, then operations that run on parameters, such as :attr:`cuda`,

                        are ignored. If ``None``, the parameter is **not** included in the

                        module's :attr:`state_dict`.

                """

                if "_parameters" not in self.__dict__:

                    raise AttributeError(

                        "cannot assign parameter before Module.__init__() call"

                    )

                elif not isinstance(name, str):

                    raise TypeError(

                        f"parameter name should be a string. Got {torch.typename(name)}"

                    )

                elif "." in name:

                    raise KeyError('parameter name can\'t contain "."')

                elif name == "":

                    raise KeyError('parameter name can\'t be empty string ""')

                elif hasattr(self, name) and name not in self._parameters:

                    raise KeyError(f"attribute '{name}' already exists")

                if param is None:

                    self._parameters[name] = None

                elif not isinstance(param, Parameter):

                    raise TypeError(

                        f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "

                        "(torch.nn.Parameter or None required)"

                    )

                elif param.grad_fn:

                    raise ValueError(

                        f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "

                        f"parameters must be created explicitly. To express '{name}' "

                        "as a function of another Tensor, compute the value in "

                        "the forward() method."

                    )

                else:

                    for hook in _global_parameter_registration_hooks.values():

                        output = hook(self, name, param)

                        if output is not None:

                            param = output

                    self._parameters[name] = param


#### register_state_dict_post_hook

```python3
def register_state_dict_post_hook(
    self,
    hook
)
```

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

??? example "View Source"
            def register_state_dict_post_hook(self, hook):

                r"""Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

                It should have the following signature::

                    hook(module, state_dict, prefix, local_metadata) -> None

                The registered hooks can modify the ``state_dict`` inplace.

                """

                # In _register_state_dict_hook there was a bug described in

                # https://github.com/pytorch/pytorch/issues/117437 where the return value

                # was only respected for the root module but not child submodules.

                # We fix this in this public version by only allowing inplace modifications on

                # the state_dict by the hook. However, since hooks registered via both these

                # APIs will be added to `_state_dict_hooks` and the type of `_state_dict_hooks`

                # cannot be changed due to many dependencies on it, we mark a hook

                # as being registered via the public API by setting `_from_public_api` on it.

                # In the implementation of `state_dict`, if the callable does not have this

                # flag, the old behavior of respecting the return value will be preserved

                # for the root module, otherwise, we ensure that the hook returns None.

                hook._from_public_api = True

                handle = RemovableHandle(self._state_dict_hooks)

                self._state_dict_hooks[handle.id] = hook

                return handle


#### register_state_dict_pre_hook

```python3
def register_state_dict_pre_hook(
    self,
    hook
)
```

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

??? example "View Source"
            def register_state_dict_pre_hook(self, hook):

                r"""Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

                It should have the following signature::

                    hook(module, prefix, keep_vars) -> None

                The registered hooks can be used to perform pre-processing before the ``state_dict``

                call is made.

                """

                handle = RemovableHandle(self._state_dict_pre_hooks)

                self._state_dict_pre_hooks[handle.id] = hook

                return handle


#### requires_grad_

```python3
def requires_grad_(
    self: ~T,
    requires_grad: bool = True
) -> ~T
```

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| requires_grad | bool | whether autograd should record operations on<br>parameters in this module. Default: ``True``. | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def requires_grad_(self: T, requires_grad: bool = True) -> T:

                r"""Change if autograd should record operations on parameters in this module.

                This method sets the parameters' :attr:`requires_grad` attributes

                in-place.

                This method is helpful for freezing part of the module for finetuning

                or training parts of a model individually (e.g., GAN training).

                See :ref:`locally-disable-grad-doc` for a comparison between

                `.requires_grad_()` and several similar mechanisms that may be confused with it.

                Args:

                    requires_grad (bool): whether autograd should record operations on

                                          parameters in this module. Default: ``True``.

                Returns:

                    Module: self

                """

                for p in self.parameters():

                    p.requires_grad_(requires_grad)

                return self


#### set_extra_state

```python3
def set_extra_state(
    self,
    state: Any
) -> None
```

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding

??? example "View Source"
            def set_extra_state(self, state: Any) -> None:

                """Set extra state contained in the loaded `state_dict`.

                This function is called from :func:`load_state_dict` to handle any extra state

                found within the `state_dict`. Implement this function and a corresponding

                :func:`get_extra_state` for your module if you need to store extra state within its

                `state_dict`.

                Args:

                    state (dict): Extra state from the `state_dict`

                """

                raise RuntimeError(

                    "Reached a code path in Module.set_extra_state() that should never be called. "

                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "

                    "to report this bug."

                )


#### set_submodule

```python3
def set_submodule(
    self,
    target: str,
    module: 'Module',
    strict: bool = False
) -> None
```

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| target | None | The fully-qualified string name of the submodule<br>to look for. (See above example for how to specify a<br>fully-qualified string.) | None |
| module | None | The module to set the submodule to. | None |
| strict | None | If ``False``, the method will replace an existing submodule<br>or create a new submodule if the parent module exists. If ``True``,<br>the method will only attempt to replace an existing submodule and throw an error<br>if the submodule doesn't already exist. | None |

**Raises:**

| Type | Description |
|---|---|
| ValueError | If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``. |
| AttributeError | If at any point along the path resulting from<br>the ``target`` string the (sub)path resolves to a non-existent<br>attribute name or an object that is not an instance of ``nn.Module``. |

??? example "View Source"
            def set_submodule(

                self, target: str, module: "Module", strict: bool = False

            ) -> None:

                """

                Set the submodule given by ``target`` if it exists, otherwise throw an error.

                .. note::

                    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule

                    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,

                    the method will only attempt to replace an existing submodule and throw an error if

                    the submodule does not exist.

                For example, let's say you have an ``nn.Module`` ``A`` that

                looks like this:

                .. code-block:: text

                    A(

                        (net_b): Module(

                            (net_c): Module(

                                (conv): Conv2d(3, 3, 3)

                            )

                            (linear): Linear(3, 3)

                        )

                    )

                (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested

                submodule ``net_b``, which itself has two submodules ``net_c``

                and ``linear``. ``net_c`` then has a submodule ``conv``.)

                To override the ``Conv2d`` with a new submodule ``Linear``, you

                could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``

                where ``strict`` could be ``True`` or ``False``

                To add a new submodule ``Conv2d`` to the existing ``net_b`` module,

                you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

                In the above if you set ``strict=True`` and call

                ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError

                will be raised because ``net_b`` does not have a submodule named ``conv``.

                Args:

                    target: The fully-qualified string name of the submodule

                        to look for. (See above example for how to specify a

                        fully-qualified string.)

                    module: The module to set the submodule to.

                    strict: If ``False``, the method will replace an existing submodule

                        or create a new submodule if the parent module exists. If ``True``,

                        the method will only attempt to replace an existing submodule and throw an error

                        if the submodule doesn't already exist.

                Raises:

                    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.

                    AttributeError: If at any point along the path resulting from

                        the ``target`` string the (sub)path resolves to a non-existent

                        attribute name or an object that is not an instance of ``nn.Module``.

                """

                if target == "":

                    raise ValueError("Cannot set the submodule without a target name!")

                atoms: list[str] = target.split(".")

                if not isinstance(module, torch.nn.Module):

                    raise ValueError(

                        "`" + "module" + f"` is not an nn.Module, found {type(module)}"

                    )

                if len(atoms) == 1:

                    parent: torch.nn.Module = self

                else:

                    parent_key = ".".join(atoms[:-1])

                    parent = self.get_submodule(parent_key)

                if strict and not hasattr(parent, atoms[-1]):

                    raise AttributeError(

                        parent._get_name() + " has no attribute `" + atoms[-1] + "`"

                    )

                if hasattr(parent, atoms[-1]):

                    mod = getattr(parent, atoms[-1])

                    if not isinstance(mod, torch.nn.Module):

                        raise AttributeError("`" + atoms[-1] + "` is not an nn.Module")

                setattr(parent, atoms[-1], module)


#### share_memory

```python3
def share_memory(
    self: ~T
) -> ~T
```

See :meth:`torch.Tensor.share_memory_`.

??? example "View Source"
            def share_memory(self: T) -> T:

                r"""See :meth:`torch.Tensor.share_memory_`."""

                return self._apply(lambda t: t.share_memory_())


#### state_dict

```python3
def state_dict(
    self,
    *args,
    destination=None,
    prefix='',
    keep_vars=False
)
```

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| destination | dict | If provided, the state of module will<br>be updated into the dict and the same object is returned.<br>Otherwise, an ``OrderedDict`` will be created and returned.<br>Default: ``None``. | None |
| prefix | str | a prefix added to parameter and buffer<br>names to compose the keys in state_dict. Default: ``''``. | None |
| keep_vars | bool | by default the :class:`~torch.Tensor` s<br>returned in the state dict are detached from autograd. If it's<br>set to ``True``, detaching will not be performed.<br>Default: ``False``. | None |

**Returns:**

| Type | Description |
|---|---|
| dict | a dictionary containing a whole state of the module |

??? example "View Source"
            def state_dict(self, *args, destination=None, prefix="", keep_vars=False):

                r"""Return a dictionary containing references to the whole state of the module.

                Both parameters and persistent buffers (e.g. running averages) are

                included. Keys are corresponding parameter and buffer names.

                Parameters and buffers set to ``None`` are not included.

                .. note::

                    The returned object is a shallow copy. It contains references

                    to the module's parameters and buffers.

                .. warning::

                    Currently ``state_dict()`` also accepts positional arguments for

                    ``destination``, ``prefix`` and ``keep_vars`` in order. However,

                    this is being deprecated and keyword arguments will be enforced in

                    future releases.

                .. warning::

                    Please avoid the use of argument ``destination`` as it is not

                    designed for end-users.

                Args:

                    destination (dict, optional): If provided, the state of module will

                        be updated into the dict and the same object is returned.

                        Otherwise, an ``OrderedDict`` will be created and returned.

                        Default: ``None``.

                    prefix (str, optional): a prefix added to parameter and buffer

                        names to compose the keys in state_dict. Default: ``''``.

                    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s

                        returned in the state dict are detached from autograd. If it's

                        set to ``True``, detaching will not be performed.

                        Default: ``False``.

                Returns:

                    dict:

                        a dictionary containing a whole state of the module

                Example::

                    >>> # xdoctest: +SKIP("undefined vars")

                    >>> module.state_dict().keys()

                    ['bias', 'weight']

                """

                # TODO: Remove `args` and the parsing logic when BC allows.

                if len(args) > 0:

                    # DeprecationWarning is ignored by default

                    warnings.warn(

                        "Positional args are being deprecated, use kwargs instead. Refer to "

                        "https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.state_dict"

                        " for details.",

                        FutureWarning,

                        stacklevel=2,

                    )

                    if destination is None:

                        destination = args[0]

                    if len(args) > 1 and prefix == "":

                        prefix = args[1]

                    if len(args) > 2 and keep_vars is False:

                        keep_vars = args[2]

                if destination is None:

                    destination = OrderedDict()

                    destination._metadata = OrderedDict()

                local_metadata = dict(version=self._version)

                if hasattr(destination, "_metadata"):

                    destination._metadata[prefix[:-1]] = local_metadata

                for hook in self._state_dict_pre_hooks.values():

                    hook(self, prefix, keep_vars)

                self._save_to_state_dict(destination, prefix, keep_vars)

                for name, module in self._modules.items():

                    if module is not None:

                        module.state_dict(

                            destination=destination,

                            prefix=prefix + name + ".",

                            keep_vars=keep_vars,

                        )

                for hook in self._state_dict_hooks.values():

                    hook_result = hook(self, destination, prefix, local_metadata)

                    if not getattr(hook, "_from_public_api", False):

                        if hook_result is not None:

                            destination = hook_result

                    else:

                        if hook_result is not None:

                            raise RuntimeError("state_dict post-hook must return None")

                return destination


#### to

```python3
def to(
    self,
    *args,
    **kwargs
)
```

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| device ( | None | class:`torch.device`): the desired device of the parameters<br>and buffers in this module | None |
| dtype ( | None | class:`torch.dtype`): the desired floating point or complex dtype of<br>the parameters and buffers in this module | None |
| tensor | torch.Tensor | Tensor whose dtype and device are the desired<br>dtype and device for all parameters and buffers in this module | None |
| memory_format ( | None | class:`torch.memory_format`): the desired memory<br>format for 4D parameters and buffers in this module (keyword<br>only argument) | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def to(self, *args, **kwargs):

                r"""Move and/or cast the parameters and buffers.

                This can be called as

                .. function:: to(device=None, dtype=None, non_blocking=False)

                   :noindex:

                .. function:: to(dtype, non_blocking=False)

                   :noindex:

                .. function:: to(tensor, non_blocking=False)

                   :noindex:

                .. function:: to(memory_format=torch.channels_last)

                   :noindex:

                Its signature is similar to :meth:`torch.Tensor.to`, but only accepts

                floating point or complex :attr:`dtype`\ s. In addition, this method will

                only cast the floating point or complex parameters and buffers to :attr:`dtype`

                (if given). The integral parameters and buffers will be moved

                :attr:`device`, if that is given, but with dtypes unchanged. When

                :attr:`non_blocking` is set, it tries to convert/move asynchronously

                with respect to the host if possible, e.g., moving CPU Tensors with

                pinned memory to CUDA devices.

                See below for examples.

                .. note::

                    This method modifies the module in-place.

                Args:

                    device (:class:`torch.device`): the desired device of the parameters

                        and buffers in this module

                    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of

                        the parameters and buffers in this module

                    tensor (torch.Tensor): Tensor whose dtype and device are the desired

                        dtype and device for all parameters and buffers in this module

                    memory_format (:class:`torch.memory_format`): the desired memory

                        format for 4D parameters and buffers in this module (keyword

                        only argument)

                Returns:

                    Module: self

                Examples::

                    >>> # xdoctest: +IGNORE_WANT("non-deterministic")

                    >>> linear = nn.Linear(2, 2)

                    >>> linear.weight

                    Parameter containing:

                    tensor([[ 0.1913, -0.3420],

                            [-0.5113, -0.2325]])

                    >>> linear.to(torch.double)

                    Linear(in_features=2, out_features=2, bias=True)

                    >>> linear.weight

                    Parameter containing:

                    tensor([[ 0.1913, -0.3420],

                            [-0.5113, -0.2325]], dtype=torch.float64)

                    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)

                    >>> gpu1 = torch.device("cuda:1")

                    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)

                    Linear(in_features=2, out_features=2, bias=True)

                    >>> linear.weight

                    Parameter containing:

                    tensor([[ 0.1914, -0.3420],

                            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')

                    >>> cpu = torch.device("cpu")

                    >>> linear.to(cpu)

                    Linear(in_features=2, out_features=2, bias=True)

                    >>> linear.weight

                    Parameter containing:

                    tensor([[ 0.1914, -0.3420],

                            [-0.5112, -0.2324]], dtype=torch.float16)

                    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)

                    >>> linear.weight

                    Parameter containing:

                    tensor([[ 0.3741+0.j,  0.2382+0.j],

                            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)

                    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))

                    tensor([[0.6122+0.j, 0.1150+0.j],

                            [0.6122+0.j, 0.1150+0.j],

                            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

                """

                device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(

                    *args, **kwargs

                )

                if dtype is not None:

                    if not (dtype.is_floating_point or dtype.is_complex):

                        raise TypeError(

                            "nn.Module.to only accepts floating point or complex "

                            f"dtypes, but got desired dtype={dtype}"

                        )

                    if dtype.is_complex:

                        warnings.warn(

                            "Complex modules are a new feature under active development whose design may change, "

                            "and some modules might not work as expected when using complex tensors as parameters or buffers. "

                            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "

                            "if a complex module does not work as expected."

                        )

                def convert(t):

                    try:

                        if convert_to_format is not None and t.dim() in (4, 5):

                            return t.to(

                                device,

                                dtype if t.is_floating_point() or t.is_complex() else None,

                                non_blocking,

                                memory_format=convert_to_format,

                            )

                        return t.to(

                            device,

                            dtype if t.is_floating_point() or t.is_complex() else None,

                            non_blocking,

                        )

                    except NotImplementedError as e:

                        if str(e) == "Cannot copy out of meta tensor; no data!":

                            raise NotImplementedError(

                                f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "

                                f"when moving module from meta to a different device."

                            ) from None

                        else:

                            raise

                return self._apply(convert)


#### to_empty

```python3
def to_empty(
    self: ~T,
    *,
    device: Union[int, str, torch.device, NoneType],
    recurse: bool = True
) -> ~T
```

Move the parameters and buffers to the specified device without copying storage.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| device ( | None | class:`torch.device`): The desired device of the parameters<br>and buffers in this module. | None |
| recurse | bool | Whether parameters and buffers of submodules should<br>be recursively moved to the specified device. | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def to_empty(

                self: T, *, device: Optional[DeviceLikeType], recurse: bool = True

            ) -> T:

                r"""Move the parameters and buffers to the specified device without copying storage.

                Args:

                    device (:class:`torch.device`): The desired device of the parameters

                        and buffers in this module.

                    recurse (bool): Whether parameters and buffers of submodules should

                        be recursively moved to the specified device.

                Returns:

                    Module: self

                """

                return self._apply(

                    lambda t: torch.empty_like(t, device=device), recurse=recurse

                )


#### train

```python3
def train(
    self: ~T,
    mode: bool = True
) -> ~T
```

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mode | bool | whether to set training mode (``True``) or evaluation<br>mode (``False``). Default: ``True``. | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def train(self: T, mode: bool = True) -> T:

                r"""Set the module in training mode.

                This has an effect only on certain modules. See the documentation of

                particular modules for details of their behaviors in training/evaluation

                mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,

                etc.

                Args:

                    mode (bool): whether to set training mode (``True``) or evaluation

                                 mode (``False``). Default: ``True``.

                Returns:

                    Module: self

                """

                if not isinstance(mode, bool):

                    raise ValueError("training mode is expected to be boolean")

                self.training = mode

                for module in self.children():

                    module.train(mode)

                return self


#### type

```python3
def type(
    self: ~T,
    dst_type: Union[torch.dtype, str]
) -> ~T
```

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| dst_type | type or string | the desired type | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def type(self: T, dst_type: Union[dtype, str]) -> T:

                r"""Casts all parameters and buffers to :attr:`dst_type`.

                .. note::

                    This method modifies the module in-place.

                Args:

                    dst_type (type or string): the desired type

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.type(dst_type))


#### xpu

```python3
def xpu(
    self: ~T,
    device: Union[int, torch.device, NoneType] = None
) -> ~T
```

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| device | int | if specified, all parameters will be<br>copied to that device | None |

**Returns:**

| Type | Description |
|---|---|
| Module | self |

??? example "View Source"
            def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:

                r"""Move all model parameters and buffers to the XPU.

                This also makes associated parameters and buffers different objects. So

                it should be called before constructing optimizer if the module will

                live on XPU while being optimized.

                .. note::

                    This method modifies the module in-place.

                Arguments:

                    device (int, optional): if specified, all parameters will be

                        copied to that device

                Returns:

                    Module: self

                """

                return self._apply(lambda t: t.xpu(device))


#### zero_grad

```python3
def zero_grad(
    self,
    set_to_none: bool = True
) -> None
```

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| set_to_none | bool | instead of setting to zero, set the grads to None.<br>See :meth:`torch.optim.Optimizer.zero_grad` for details. | None |

??? example "View Source"
            def zero_grad(self, set_to_none: bool = True) -> None:

                r"""Reset gradients of all model parameters.

                See similar function under :class:`torch.optim.Optimizer` for more context.

                Args:

                    set_to_none (bool): instead of setting to zero, set the grads to None.

                        See :meth:`torch.optim.Optimizer.zero_grad` for details.

                """

                if getattr(self, "_is_replica", False):

                    warnings.warn(

                        "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "

                        "The parameters are copied (in a differentiable manner) from the original module. "

                        "This means they are not leaf nodes in autograd and so don't accumulate gradients. "

                        "If you need gradients in your forward method, consider using autograd.grad instead."

                    )

                for p in self.parameters():

                    if p.grad is not None:

                        if set_to_none:

                            p.grad = None

                        else:

                            if p.grad.grad_fn is not None:

                                p.grad.detach_()

                            else:

                                p.grad.requires_grad_(False)

                            p.grad.zero_()
