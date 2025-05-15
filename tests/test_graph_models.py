#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit tests for graph models
"""

import pytest
import torch
import numpy as np
import pandas as pd
import logging
from torch_geometric.loader import DataLoader
from redxregressors import datasets, graph_models, ml_featurization
from redxregressors.ml_featurization import GCN_featurize
from torch_geometric.nn.models import AttentiveFP
from redxregressors.graph_models import train_attentivefp_pyg, gcn_train, GCN

log = logging.getLogger(__name__)


@pytest.fixture
def sample_data_attentive_fp():
    raw_dataset = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "c1ccccc1",
                "C#N",
                "C#C",
                "C1CCCCC1CCN",
                "c1ccccc1C(=O)O",
                "c1ccccc1Br",
                "c1ccccc1I",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "names": [
                "ethanol",
                "ethanamine",
                "propane",
                "benzene",
                "hydrogen cyanide",
                "acetylene",
                "cyclohexylamine",
                "benzoic acid",
                "bromobenzene",
                "iodobenzene",
            ],
        }
    )

    dataset = datasets.RedxCSVGraphMolDataSet(
        csv_file=raw_dataset,
        smiles_column="smiles",
        property_columns=["target"],
        pre_transform=ml_featurization.GetAttentiveFPFeatures(),
        overwrite_existing=True,
    ).shuffle()

    train, test, validation = (
        datasets.split_redx_csv_data_set_into_train_test_validation(
            dataset, train_frac=0.8, test_frac=0.1, batch_size=2
        )
    )

    return train, test, validation


def test_get_features_attentivefp():
    raw_dataset = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "c1ccccc1",
                "C#N",
                "C#C",
                "C1CCCCC1CCN",
                "c1ccccc1C(=O)O",
                "c1ccccc1Br",
                "c1ccccc1I",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "names": [
                "ethanol",
                "ethanamine",
                "propane",
                "benzene",
                "hydrogen cyanide",
                "acetylene",
                "cyclohexylamine",
                "benzoic acid",
                "bromobenzene",
                "iodobenzene",
            ],
        }
    )

    dataset = datasets.RedxCSVGraphMolDataSet(
        csv_file=raw_dataset,
        smiles_column="smiles",
        property_columns=["target"],
        pre_transform=ml_featurization.GetAttentiveFPFeatures(),
        overwrite_existing=True,
    )

    assert isinstance(dataset, datasets.RedxCSVGraphMolDataSet)
    assert len(dataset) == 10
    expected_1 = np.array(
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    expected_2 = np.array(
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    expected_3 = np.array(
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    assert np.allclose(dataset[0].x[0].numpy(), expected_1)
    log.error(dataset[0].x[0].numpy())
    assert np.allclose(dataset[0].x[1].numpy(), expected_2)
    log.error(dataset[0].x[1].numpy())
    assert np.allclose(dataset[1].x[0].numpy(), expected_3)
    log.error(dataset[1].x[0].numpy())


def test_train_test_validation_splitting():
    raw_dataset = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "c1ccccc1",
                "C#N",
                "C#C",
                "C1CCCCC1CCN",
                "c1ccccc1C(=O)O",
                "c1ccccc1Br",
                "c1ccccc1I",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "names": [
                "ethanol",
                "ethanamine",
                "propane",
                "benzene",
                "hydrogen cyanide",
                "acetylene",
                "cyclohexylamine",
                "benzoic acid",
                "bromobenzene",
                "iodobenzene",
            ],
        }
    )

    dataset = datasets.RedxCSVGraphMolDataSet(
        csv_file=raw_dataset,
        smiles_column="smiles",
        property_columns=["target"],
        pre_transform=ml_featurization.GetAttentiveFPFeatures(),
        overwrite_existing=True,
    ).shuffle()

    train, test, validation = (
        datasets.split_redx_csv_data_set_into_train_test_validation(
            dataset, train_frac=0.8, test_frac=0.1, batch_size=1
        )
    )

    assert len(train) == 8
    assert len(test) == 1
    assert len(validation) == 1


def test_train_attentivefp_pyg(sample_data_attentive_fp):
    train, test, _ = sample_data_attentive_fp

    device = "cpu"
    model = AttentiveFP(
        in_channels=39,
        hidden_channels=200,
        out_channels=1,
        edge_dim=10,
        num_layers=2,
        num_timesteps=2,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5, weight_decay=10**-5)

    n_epochs = 5
    losses = train_attentivefp_pyg(model, train, optimizer=optimizer, epochs=n_epochs)

    assert isinstance(losses, list)
    assert len(losses) == n_epochs
    assert all(isinstance(loss, float) for loss in losses)

    predictions = graph_models.test_attentivefp_pyg(model, test)
    assert isinstance(predictions, list)

    predictions2 = graph_models.test_attentivefp_pyg(model, test)
    assert isinstance(predictions2, list)

    assert np.allclose(predictions, predictions2)


##### GCN #####


@pytest.fixture
def sample_data_gcn():
    raw_dataset = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "c1ccccc1",
                "C#N",
                "C#C",
                "C1CCCCC1CCN",
                "c1ccccc1C(=O)O",
                "c1ccccc1Br",
                "c1ccccc1I",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "names": [
                "ethanol",
                "ethanamine",
                "propane",
                "benzene",
                "hydrogen cyanide",
                "acetylene",
                "cyclohexylamine",
                "benzoic acid",
                "bromobenzene",
                "iodobenzene",
            ],
        }
    )
    gcn_featurize = GCN_featurize(
        raw_dataset["smiles"].values, raw_dataset["target"].values
    )

    train, test, validation = gcn_featurize()

    return train, test, validation


def test_get_features_gcn():
    raw_dataset = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "c1ccccc1",
                "C#N",
                "C#C",
                "C1CCCCC1CCN",
                "c1ccccc1C(=O)O",
                "c1ccccc1Br",
                "c1ccccc1I",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "names": [
                "ethanol",
                "ethanamine",
                "propane",
                "benzene",
                "hydrogen cyanide",
                "acetylene",
                "cyclohexylamine",
                "benzoic acid",
                "bromobenzene",
                "iodobenzene",
            ],
        }
    )
    gcn_featurize = GCN_featurize(
        raw_dataset["smiles"].values, raw_dataset["target"].values, train_fraction=None
    )
    dataset = gcn_featurize()

    assert isinstance(dataset, DataLoader)
    assert len(dataset) == 10
    expected_1 = np.array(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    expected_2 = np.array(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    expected_3 = np.array(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    assert np.allclose(dataset.dataset[0].x[0].numpy(), expected_1)
    log.critical(dataset.dataset[0].x[0].numpy())
    assert np.allclose(dataset.dataset[0].x[1].numpy(), expected_2)
    log.critical(dataset.dataset[0].x[1].numpy())
    assert np.allclose(dataset.dataset[1].x[0].numpy(), expected_3)
    log.critical(dataset.dataset[1].x[0].numpy())


def test_train_test_validation_gcn_splitting():
    raw_dataset = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "c1ccccc1",
                "C#N",
                "C#C",
                "C1CCCCC1CCN",
                "c1ccccc1C(=O)O",
                "c1ccccc1Br",
                "c1ccccc1I",
            ],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "names": [
                "ethanol",
                "ethanamine",
                "propane",
                "benzene",
                "hydrogen cyanide",
                "acetylene",
                "cyclohexylamine",
                "benzoic acid",
                "bromobenzene",
                "iodobenzene",
            ],
        }
    )
    gcn_featurize = GCN_featurize(
        raw_dataset["smiles"].values,
        raw_dataset["target"].values,
        train_fraction=0.8,
        test_fraction=0.1,
        batch_size=1,
    )

    train, test, validation = gcn_featurize()

    assert len(train) == 8
    assert len(test) == 1
    assert len(validation) == 1


def test_train_gcn_pyg(sample_data_gcn):
    train, test, _ = sample_data_gcn

    model = GCN(n_ouputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    n_epochs = 5
    losses = gcn_train(model, train, optimizer=optimizer, epochs=n_epochs)

    assert isinstance(losses, list)
    assert len(losses) == n_epochs
    assert all(isinstance(loss, float) for loss in losses)

    predictions = graph_models.gcn_test(model, test)
    assert isinstance(predictions, list)

    predictions2 = graph_models.gcn_test(model, test)
    assert isinstance(predictions2, list)

    assert np.allclose(predictions, predictions2)


if __name__ == "__main__":
    pytest.main()
