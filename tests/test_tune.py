import logging
import shutil
import pytest
import numpy as np
from unittest.mock import Mock
from redxregressors import tune
from redxregressors.tune import (
    build_kfold_objective,
    JoinKfoldData,
    build_train_test_objective,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import optuna
import matplotlib.pyplot as plt

# Mock utilities and log
tune.utilities = Mock()
tune.utilities.random_seed = 42
tune.log = Mock()


log = logging.getLogger(__name__)


# Mock optuna trial
class MockTrial:
    def suggest_float(self, name, low, high):
        return (low + high) / 2

    def suggest_int(self, name, low, high):
        return (low + high) // 2

    def suggest_categorical(self, name, choices):
        return choices[0]


@pytest.fixture
def trial():
    return MockTrial()


def test_build_param_grid(trial):
    parameters = {
        "param1": ["float", 0.0, 1.0],
        "param2": ["int", 1, 10],
        "param3": ["catagorical", ["a", "b", "c"]],
    }
    expected_output = {"param1": 0.5, "param2": 5, "param3": "a"}
    assert tune.build_param_grid(parameters, trial) == expected_output


def test_objective_defined_train_test_single_rmse(trial):
    regressor = Mock()
    regressor.return_value = regressor
    regressor.fit = Mock()
    regressor.predict = Mock(return_value=np.array([1, 2, 3]))

    X_train = np.array([[1], [2], [3]])
    X_test = np.array([[1], [2], [3]])
    y_train = np.array([1, 2, 3])
    y_test = np.array([1, 2, 3])

    parameters = {
        "param1": ["float", 0.0, 1.0],
        "param2": ["int", 1, 10],
        "param3": ["catagorical", ["a", "b", "c"]],
    }

    expected_output = 0.0
    assert (
        tune.objective_defined_train_test_single_rmse(
            trial, regressor, parameters, X_train, X_test, y_train, y_test
        )
        == expected_output
    )


def test_layer_size_to_network(trial):
    param_grid = {"n_layers": 3}
    expected_output = {"hidden_layer_sizes": (55, 55, 55)}
    assert (
        tune.layer_size_to_network(trial, param_grid, min_nodes=50, max_nodes=60)
        == expected_output
    )


def test_objective_random_cv_multi_rmse__r2__diff_train_test(trial):
    """
    We make a mock regressor that returns a fixed value of 1 when predict is called. As we use the train and test set with the predict method we
    need to use the same number of values for each set hence need an even number of inputs and even number of folds which is exactly divisable
    """
    regressorfx = Mock()
    regressorfx.return_value = regressorfx
    regressorfx.fit = Mock()
    regressorfx.predict = Mock(return_value=np.array([1, 2]))

    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 4])

    parameters = {
        "param1": ["float", 0.0, 1.0],
        "param2": ["int", 1, 10],
        "param3": ["catagorical", ["a", "b", "c"]],
    }

    expected_output = (1.1441228056353687, -0.5, 0.0)
    o = tune.objective_random_cv_multi_rmse__r2__diff_train_test(
        trial, regressorfx, parameters, X, y, k_fold=2
    )
    assert (
        np.isclose(o[0], expected_output[0], rtol=1e-5)
        and np.isclose(o[1], expected_output[1], rtol=1e-5)
        and np.isclose(o[2], expected_output[2], rtol=1e-5)
    )


def test_objective_predefined_cv_multi_rmse__r2__diff_train_test(trial):
    regressorfx = Mock()
    regressorfx.return_value = regressorfx
    regressorfx.fit = Mock()
    regressorfx.predict = Mock(return_value=np.array([1, 2, 3]))

    train = Mock()
    train.X = np.array([[1], [2], [3]])
    train.y = np.array([1, 2, 3])

    test = Mock()
    test.X = np.array([[1], [2], [3]])
    test.y = np.array([1, 2, 3])

    predfined_kfolds = [(train, test)]

    parameters = {
        "param1": ["float", 0.0, 1.0],
        "param2": ["int", 1, 10],
        "param3": ["catagorical", ["a", "b", "c"]],
    }

    expected_output = (0.0, 1.0, 0.0)
    assert (
        tune.objective_predefined_cv_multi_rmse__r2__diff_train_test(
            trial, regressorfx, parameters, predfined_kfolds
        )
        == expected_output
    )


##### KFold Objective Tests #####


@pytest.fixture
def sample_data():
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    return X, y


@pytest.fixture
def sample_kf():
    kf = KFold(n_splits=5)
    return [
        (train_idx, test_idx)
        for train_idx, test_idx in kf.split(np.random.rand(100, 10))
    ]


@pytest.fixture
def sample_trial():
    trial = Mock(spec=optuna.trial.Trial)
    trial.suggest_float = Mock(return_value=0.5)
    trial.suggest_int = Mock(return_value=5)
    trial.suggest_categorical = Mock(return_value="a")
    return trial


def test_get_all_data_as_single_fold(sample_data):
    X, y = sample_data
    kf = KFold(n_splits=5)
    cv_data = [
        (
            JoinKfoldData(X[train_idx], y[train_idx]),
            JoinKfoldData(X[test_idx], y[test_idx]),
        )
        for train_idx, test_idx in kf.split(X, y)
    ]

    obj = build_kfold_objective(name="test", k=5)
    obj.cv_data = cv_data

    X_combined, y_combined = obj.get_all_data_as_single_fold()

    assert X_combined.shape == (100, 10)
    assert y_combined.shape == (100,)


def test_abs_train_test_diff_objective():
    obj = build_kfold_objective(name="test", k=5)
    train_scores = np.array([0.8, 0.85, 0.9])
    test_scores = np.array([0.75, 0.8, 0.85])

    diff = obj.abs_train_test_diff_objective(train_scores, test_scores)

    assert diff == pytest.approx(0.05)


def test_get_data(sample_data):
    X, y = sample_data
    obj = build_kfold_objective(name="test", k=5)

    obj.get_data(X, y)

    assert len(obj.cv_data) == 5
    assert len(obj.cv_ids) == 5


def test_set_objectives():
    obj = build_kfold_objective(name="test", k=5)
    objectives = [r2_score, mean_absolute_error]
    directions = ["minimize", "maximize"]

    result = obj.set_objectives(
        objectives, directions, add_train_test_diff_objective=True
    )

    assert len(obj.objectives) == 2
    assert len(obj.directions) == 2
    assert result[-1] == "minimize"


def test_parity_plot():
    y_test = np.random.rand(100)
    y_pred = np.random.rand(100)
    xymin = 0
    xymax = 1

    fig = build_kfold_objective.parity_plot(y_test, y_pred, xymin, xymax)

    assert isinstance(fig, plt.Figure)


def test_plot_residuals():
    y_test = np.random.rand(100)
    y_pred = np.random.rand(100)

    fig = build_kfold_objective.plot_residuals(y_test, y_pred)

    assert isinstance(fig, plt.Figure)


def test_plot_prediction_error():
    y_test = np.random.rand(100)
    y_pred = np.random.rand(100)

    fig = build_kfold_objective.plot_prediction_error(y_test, y_pred)

    assert isinstance(fig, plt.Figure)


def test_plot_qq():
    y_test = np.random.rand(100)
    y_pred = np.random.rand(100)

    fig = build_kfold_objective.plot_qq(y_test, y_pred)

    assert isinstance(fig, plt.Figure)


def test_objectivefx_kf(sample_data, sample_parameters_tt_rf, sample_trial_tt):
    X, y = sample_data
    obj = build_kfold_objective(name="test", k=5)
    obj.get_data(X=X, y=y)
    obj.set_objectives([mean_squared_error, r2_score], ["minimize", "maximize"])

    pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))]
    )

    scores = obj.objectivefx(
        trial=sample_trial_tt,
        regressorfx=pipe,
        parameters=sample_parameters_tt_rf,
        name="kfold_run",
        without_mlflow=True,
    )

    shutil.rmtree("Kfold-training", ignore_errors=True)
    assert isinstance(scores, list)
    assert len(scores) == 2


##### Train Test Objective Tests #####


@pytest.fixture
def sample_data_tt():
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    return X, y


@pytest.fixture
def sample_parameters_tt():
    return {
        "param1": ["float", 0.0, 1.0],
        "param2": ["int", 1, 10],
        "param3": ["catagorical", ["a", "b", "c"]],
    }


@pytest.fixture
def sample_parameters_tt_rf():
    return {
        "model__n_estimators": ["int", 10, 20],
        "model__max_depth": ["int", 2, 4],
        "model__max_features": ["float", 0.5, 0.75],
    }


@pytest.fixture
def sample_trial_tt():
    study = optuna.create_study(direction="minimize")
    return study.ask()


def test_get_all_data_as_single_set(sample_data_tt):
    X, y = sample_data_tt
    obj = build_train_test_objective(name="test")
    obj.get_data(X, y, train_frac=0.8)
    X_combined, y_combined = obj.get_all_data_as_single_set()

    assert X_combined.shape == (100, 10)
    assert y_combined.shape == (100,)


def test_abs_train_test_diff_objective_tt():
    obj = build_train_test_objective(name="test")
    train_scores = np.array([0.8, 0.85, 0.9])
    test_scores = np.array([0.75, 0.8, 0.85])

    diff = obj.abs_train_test_diff_objective(train_scores, test_scores)

    assert diff == pytest.approx(0.05)


def test_build_param_grid_tt(sample_parameters_tt, sample_trial_tt):
    obj = build_train_test_objective(name="test")
    param_grid = obj.build_param_grid(sample_parameters_tt, sample_trial_tt)

    assert "param1" in param_grid
    assert "param2" in param_grid
    assert "param3" in param_grid


def test_get_data_tt(sample_data_tt):
    X, y = sample_data_tt
    obj = build_train_test_objective(name="test")

    obj.get_data(X, y, train_frac=0.8)

    assert obj.train.X.shape[0] == 80
    assert obj.test.X.shape[0] == 20


def test_set_objectives_tt():
    obj = build_train_test_objective(name="test")
    objectives = [mean_squared_error, r2_score]
    directions = ["minimize", "maximize"]

    result = obj.set_objectives(
        objectives, directions, add_train_test_diff_objective=True
    )

    assert len(obj.objectives) == 2
    assert len(obj.directions) == 2
    assert result[-1] == "minimize"


def test_parity_plot_tt(sample_data_tt):
    X, y = sample_data_tt
    obj = build_train_test_objective(name="test")
    obj.get_data(X, y, train_frac=0.8)

    y_test = obj.test.y
    y_pred = np.random.rand(len(y_test))
    xymin = 0
    xymax = 1

    fig = obj.parity_plot(y_test, y_pred, xymin, xymax)

    assert isinstance(fig, plt.Figure)


def test_plot_residuals_tt(sample_data_tt):
    X, y = sample_data_tt
    obj = build_train_test_objective(name="test")
    obj.get_data(X, y, train_frac=0.8)

    y_test = obj.test.y
    y_pred = np.random.rand(len(y_test))

    fig = obj.plot_residuals(y_test, y_pred)

    assert isinstance(fig, plt.Figure)


def test_plot_prediction_error_tt(sample_data_tt):
    X, y = sample_data_tt
    obj = build_train_test_objective(name="test")
    obj.get_data(X, y, train_frac=0.8)

    y_test = obj.test.y
    y_pred = np.random.rand(len(y_test))

    fig = obj.plot_prediction_error(y_test, y_pred)

    assert isinstance(fig, plt.Figure)


def test_plot_qq_tt(sample_data_tt):
    X, y = sample_data_tt
    obj = build_train_test_objective(name="test")
    obj.get_data(X, y, train_frac=0.8)

    y_test = obj.test.y
    y_pred = np.random.rand(len(y_test))

    fig = obj.plot_qq(y_test, y_pred)

    assert isinstance(fig, plt.Figure)


def test_objectivefx_tt(sample_data_tt, sample_parameters_tt_rf, sample_trial_tt):
    X, y = sample_data_tt
    obj = build_train_test_objective(name="test")
    obj.get_data(X, y, train_frac=0.8)
    obj.set_objectives([mean_squared_error, r2_score], ["minimize", "maximize"])

    pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))]
    )

    scores = obj.objectivefx(
        trial=sample_trial_tt,
        regressorfx=pipe,
        parameters=sample_parameters_tt_rf,
        name="tt_run",
        without_mlflow=True,
    )
    shutil.rmtree("Train-test-training", ignore_errors=True)

    assert isinstance(scores, list)
    assert len(scores) == 2


if __name__ == "__main__":
    pytest.main()
