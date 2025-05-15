import pytest
import pandas as pd
import numpy as np
import deepchem as dc
import logging
from redxregressors.deep_net_models import (
    seed_all,
    build_in_memory_loader,
    fit_mtr_pytorch_model,
    fit_mtr_pytorch_model_per_task,
    evaluate_mtr_pytorch_model,
    train_multitask_regressor,
    train_progressive_multitask_regressor,
)

log = logging.getLogger(__name__)


@pytest.fixture
def sample_data():
    data_df = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "CCS",
                "CCCl",
                "CCBr",
                "CCI",
                "CCF",
                "CCCO",
                "CCCN",
                "CCCO",
                "CCCN",
                "CCCC",
                "CCCS",
                "CCCCl",
                "CCCBr",
                "CCCI",
                "CCCF",
                "CCCCO",
                "CCCCN",
            ],
            "task1": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
            ],
            "task2": [
                1.5,
                2.5,
                3.5,
                4.5,
                5.5,
                6.5,
                7.5,
                8.5,
                9.5,
                10.5,
                11.5,
                12.5,
                13.5,
                14.5,
                15.5,
                16.5,
                17.5,
                18.5,
                19.5,
                20.5,
            ],
        }
    )
    tasks = ["task1", "task2"]
    return data_df, tasks


# def test_seed_all():
#     seed_all()
#     assert np.random.get_state()[1][0] == random_seed
#     assert random.getstate()[1][0] == random_seed
#     assert torch.initial_seed() == random_seed


def test_build_in_memory_loader(sample_data):
    data_df, tasks = sample_data
    featurizer = dc.feat.CircularFingerprint(size=1024)
    ids_column = "smiles"
    ids = data_df[ids_column].values
    smiles = data_df["smiles"].values
    targets = data_df[tasks].values
    weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)
    splitter = dc.splits.RandomSplitter()

    train_dataset, valid_dataset, test_dataset, splitter = build_in_memory_loader(
        tasks=tasks,
        featurizer=featurizer,
        ids_column=ids_column,
        ids=ids,
        smiles=smiles,
        targets=targets,
        weights=weights,
        splitter=splitter,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
    )

    assert train_dataset is not None
    assert valid_dataset is not None
    assert test_dataset is not None
    assert splitter is not None
    assert train_dataset.X.shape == (16, 1024)
    assert valid_dataset.X.shape == (2, 1024)
    assert test_dataset.X.shape == (2, 1024)


def test_fit_mtr_pytorch_model(sample_data):
    data_df, tasks = sample_data
    featurizer = dc.feat.CircularFingerprint(size=100)
    ids_column = "smiles"
    ids = data_df[ids_column].values
    smiles = data_df["smiles"].values
    targets = data_df[tasks].values
    weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)
    splitter = dc.splits.RandomSplitter()

    seed_all()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dc.data.NumpyDataset(
            X=featurizer.featurize(smiles), y=targets, w=weights, ids=ids
        )
    )

    model = dc.models.MultitaskRegressor(
        n_tasks=len(tasks),
        n_features=100,
        layer_sizes=[10, 10, 10],
        learning_rate=0.0001,
        batch_size=4,
        deterministic=True,
    )

    trained_model = fit_mtr_pytorch_model(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=4,
        unique_string="test",
    )

    assert trained_model is not None
    predictions = trained_model.predict_on_batch(test_dataset.X)
    predictions_reversed = trained_model.predict_on_batch(
        np.array([test_dataset.X[1], test_dataset.X[0]])
    )
    assert np.allclose(predictions[0], predictions_reversed[1])
    assert np.allclose(predictions[1], predictions_reversed[0])


def test_fit_mtr_pytorch_model_per_task(sample_data):
    data_df, tasks = sample_data
    featurizer = dc.feat.CircularFingerprint(size=100)
    ids_column = "smiles"
    ids = data_df[ids_column].values
    smiles = data_df["smiles"].values
    targets = data_df[tasks].values
    weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)
    splitter = dc.splits.RandomSplitter()

    seed_all()

    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dc.data.NumpyDataset(
            X=featurizer.featurize(smiles), y=targets, w=weights, ids=ids
        )
    )

    model = dc.models.torch_models.ProgressiveMultitaskModel(
        n_tasks=len(tasks),
        n_features=100,
        layer_sizes=[10, 10, 10],
        learning_rate=0.0001,
        batch_size=50,
        deterministic=True,
    )

    trained_model = fit_mtr_pytorch_model_per_task(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=10,
        unique_string="test",
    )

    assert trained_model is not None
    predictions = trained_model.predict_on_batch(test_dataset.X)
    predictions_reversed = trained_model.predict_on_batch(
        np.array([test_dataset.X[1], test_dataset.X[0]])
    )
    # log.critical(test_dataset.X)
    # log.critical(predictions)
    # log.critical(predictions_reversed)
    assert np.allclose(predictions[0], predictions_reversed[1])
    assert np.allclose(predictions[1], predictions_reversed[0])


def test_evaluate_mtr_pytorch_model(sample_data):
    data_df, tasks = sample_data
    featurizer = dc.feat.CircularFingerprint(size=1024)
    ids_column = "smiles"
    ids = data_df[ids_column].values
    smiles = data_df["smiles"].values
    targets = data_df[tasks].values
    weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)
    splitter = dc.splits.RandomSplitter()
    seed_all()
    _, _, test_dataset = splitter.train_valid_test_split(
        dc.data.NumpyDataset(
            X=featurizer.featurize(smiles), y=targets, w=weights, ids=ids
        )
    )

    model = dc.models.MultitaskRegressor(
        n_tasks=len(tasks),
        n_features=1024,
        layer_sizes=[1000, 1000],
        learning_rate=0.0001,
        batch_size=50,
    )

    test_set_df = evaluate_mtr_pytorch_model(
        model=model, test_dataset=test_dataset, tasks=tasks, unique_string="test"
    )

    log.critical(f"Evaluation : {test_set_df}")
    # assert True is False
    assert isinstance(test_set_df, pd.DataFrame)
    assert test_set_df.shape[0] == 5
    assert test_set_df.shape[1] == 3
    assert test_set_df.columns.tolist() == ["mean over tasks", "task1", "task2"]
    assert test_set_df.index.tolist() == ["RMS", "R2", "Pearson R2", "MAE", "MAPE"]
    # These is is a very brittle tests as the values are not fixed. In these 0.95, 12.0 and 1.0 are the expected values and we check that we find it within 0.005
    assert round(test_set_df["task2"]["MAPE"], 2) - 0.95 <= 0.005
    assert round(test_set_df["task1"]["MAE"], 2) - 12.0 <= 0.005
    assert round(test_set_df["mean over tasks"]["Pearson R2"], 2) - 1.0 <= 0.005


def test_train_multitask_regressor(sample_data):
    data_df, tasks = sample_data

    model, train_dataset, valid_dataset, test_dataset, test_set_df = (
        train_multitask_regressor(data_df=data_df, tasks=tasks, epochs=4)
    )

    assert model is not None
    assert train_dataset is not None
    assert valid_dataset is not None
    assert test_dataset is not None
    assert isinstance(test_set_df, pd.DataFrame)


def test_train_progressive_multitask_regressor(sample_data):
    data_df, tasks = sample_data

    model, train_dataset, valid_dataset, test_dataset, test_set_df = (
        train_progressive_multitask_regressor(data_df=data_df, tasks=tasks, epochs=4)
    )

    assert model is not None
    assert train_dataset is not None
    assert valid_dataset is not None
    assert test_dataset is not None
    assert isinstance(test_set_df, pd.DataFrame)


if __name__ == "__main__":
    pytest.main()
