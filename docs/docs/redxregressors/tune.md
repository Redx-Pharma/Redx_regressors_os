# Module redxregressors.tune

Module for tuning hyper-parameters for a model using Optuna

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module for tuning hyper-parameters for a model using Optuna

        """

        import logging

        from pathlib import Path

        import deepchem as dc

        import json

        import deepchem.data

        import optuna

        import mlflow

        from typing import Union, Callable, Tuple, Optional, Any, List

        from datetime import datetime

        import pandas as pd

        import numpy as np

        from sklearn.model_selection import KFold, train_test_split

        from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

        from sklearn.pipeline import Pipeline

        import seaborn as sns

        import matplotlib.pyplot as plt

        import scipy

        from dataclasses import dataclass

        from deepchem import data

        from copy import deepcopy

        from redxregressors import deep_net_models



        from redxregressors import (

            classical_ml_models,

            ml_featurization,

            utilities,

            ml_flow_funcs,

            applicability_domain,

        )

        log = logging.getLogger(__name__)

        optuna.logging.set_verbosity(optuna.logging.WARNING)



        @dataclass

        class JoinKfoldData:

            X: np.ndarray

            y: np.ndarray



        class build_train_test_objective(object):

            def __init__(

                self,

                name: str,

                train_frac: Optional[float] = None,

                n_train: Optional[int] = None,

                **kwargs,

            ) -> None:

                self.name = name

                self.train_frac = train_frac

                self.n_train = n_train

                self.train = []

                self.test = []

                self.train_test_ids = []

                self.kwargs = kwargs

            def get_all_data_as_single_set(

                self,

                kf: Optional[

                    List[

                        Tuple[

                            Union[np.ndarray, data.datasets.DiskDataset],

                            Union[np.ndarray, data.datasets.DiskDataset],

                        ]

                    ]

                ] = None,

            ) -> Tuple[np.ndarray, np.ndarray]:

                """

                Function to get all the data as a single set. It combines te train and test set into a single set. This is useful for training a model on all the data.

                The returned data is a tuple of the features and target values as numpy ndarrays.

                Returns:

                    Tuple[np.ndarray, np.ndarray]: The features and target values

                """

                X = np.concatenate([ent.X for ent in [self.train, self.test]])

                y = np.concatenate([ent.y for ent in [self.train, self.test]])

                return X, y

            def abs_train_test_diff_objective(

                self, train_scores: np.ndarray, test_scores: np.ndarray

            ) -> float:

                """

                Function to calculate the difference between the training and test set scores. This is used as a restraining metric to prevent overfitting.

                Note this is the difference between the mean scores for all other objectives on all folds of the k fold.

                Args:

                    train_scores (np.ndarray): The training set scores

                    test_scores (np.ndarray): The test set scores

                Returns:

                    float: The absolute difference between the training and test set scores

                """

                return np.abs(np.mean(train_scores) - np.mean(test_scores))

            def build_param_grid(

                self, parameters, trial: optuna.trial.Trial, param_grid: Optional[dict] = None

            ) -> dict[Any, Any]:

                """

                Function to build the parameter grid for the optimization

                Args:

                    parameters (dict): The parameters to optimize over

                    trial (optuna.trial.Trial): The optuna trial object

                    param_grid (Optional[dict], optional): The parameter grid to update. Defaults to None.

                Returns:

                    dict[Any, Any]: The parameter grid

                """

                if param_grid is None:

                    param_grid = {}

                for k, v in parameters.items():

                    if v[0].lower().strip() == "float":

                        param_grid[k] = trial.suggest_float(k, v[1], v[2])

                    elif v[0].lower().strip() == "int":

                        param_grid[k] = trial.suggest_int(k, v[1], v[2])

                    if v[0].lower().strip() == "catagorical":

                        param_grid[k] = trial.suggest_categorical(k, v[1])

                return param_grid

            def get_data(

                self,

                X: Optional[Union[np.ndarray, pd.DataFrame]] = None,

                y: Optional[Union[np.ndarray, pd.Series]] = None,

                train_test_predefined: Optional[

                    Tuple[

                        Union[np.ndarray, data.datasets.DiskDataset],

                        Union[np.ndarray, data.datasets.DiskDataset],

                    ]

                ] = None,

                train_frac: Optional[float] = None,

                n_train: Optional[int] = None,

            ) -> None:

                """

                Data should be passed in as either lists or tuples containing numpy arrays or pandas dataframes/Series objects to X and y

                which will be randomly split into k folds or a predefined kfold object can be passed in as kf. The predefined kf should be

                either a deepchem kfold object or list of tuples of JoinKfoldData classes with X and y being numpy arrays. The data is then

                stored in the cv_data and cv_ids attributes of the class object.

                Args:

                    X (Optional[Union[np.ndarray, pd.DataFrame]]): The features

                    y (Optional[Union[np.ndarray, pd.Series]]): The target values

                    kf (Optional[List[Tuple[Union[np.ndarray, data.datasets.DiskDataset], Union[np.ndarray, data.datasets.DiskDataset]]]): The predefined kfold object

                """

                # deal with the case where the data is passed in as numpy arrays or pandas dataframes

                if X is not None and y is not None:

                    if isinstance(X, pd.DataFrame):

                        X = X.values

                    if isinstance(y, pd.Series):

                        y = y.values

                    # if the train_frac is not set then set it to the default value of 0.8

                    if self.train_frac is None and train_frac is None:

                        # if the n_train is not set then set it to the default value of None and use 0.8 fraction

                        if self.n_train is None and n_train is None:

                            log.warning(

                                "Both the class level train_frac variable and function level train_frac argument are set to None, the train_frac will be set to the default value of 0.8"

                            )

                            self.train_frac = 0.8

                        elif n_train is not None:

                            self.n_train = n_train

                    elif train_frac is not None:

                        self.train_frac = train_frac

                    # if the train_frac is set then split the data into a training and test set

                    if self.train_frac is not None:

                        X_train, X_test, y_train, y_test = train_test_split(

                            X,

                            y,

                            train_size=self.train_frac,

                            random_state=utilities.random_seed,

                            shuffle=True,

                        )

                        self.train = JoinKfoldData(X=X_train, y=y_train)

                        self.test = JoinKfoldData(X=X_test, y=y_test)

                    # if the n_train is set then split the data into a training and test set

                    else:

                        X_train, X_test, y_train, y_test = train_test_split(

                            X,

                            y,

                            train_size=self.n_train,

                            random_state=utilities.random_seed,

                            shuffle=True,

                        )

                        self.train = JoinKfoldData(X=X_train, y=y_train)

                        self.test = JoinKfoldData(X=X_test, y=y_test)

                # deal with the case where the data is passed in as a predefined train test split

                elif train_test_predefined is not None:

                    try:

                        # if the train_test_predefined is a deepchem object then set the train and test attributes

                        if "deepchem" in str(type(train_test_predefined[0])):

                            self.train = JoinKfoldData(

                                X=train_test_predefined[0].X, y=train_test_predefined[0].y

                            )

                            self.test = JoinKfoldData(

                                X=train_test_predefined[1].X, y=train_test_predefined[1].y

                            )

                            self.train_test_ids.append(

                                (train_test_predefined[0].ids, train_test_predefined[1].ids)

                            )

                        # if the train_test_predefined is a list of tuples of numpy arrays then set the train and test attributes

                        else:

                            # if the train_test_predefined is a JoinKfoldData object then set the train and test attributes

                            if isinstance(train_test_predefined[0], JoinKfoldData):

                                self.train_test_ids.append(

                                    (

                                        [

                                            f"train_row_{jth}"

                                            for jth in range(len(train_test_predefined[0].X))

                                        ],

                                        [

                                            f"test_row_{jth}"

                                            for jth in range(len(train_test_predefined[1].X))

                                        ],

                                    )

                                )

                            # if the train_test_predefined is a list of tuples of numpy arrays then set the train and test attributes

                            else:

                                log.warning(

                                    "The train_test_predefined do not have a deepchem or JoinKfoldData object format will try to format"

                                )

                                self.train = JoinKfoldData(

                                    X=train_test_predefined[0][0], y=train_test_predefined[0][1]

                                )

                                self.test = (

                                    JoinKfoldData(

                                        X=train_test_predefined[1][0],

                                        y=train_test_predefined[1][1],

                                    ),

                                )

                                self.train_test_ids.append(

                                    (

                                        [

                                            f"train_row_{jth}"

                                            for jth in range(len(train_test_predefined[0][0]))

                                        ],

                                        [

                                            f"test_row_{jth}"

                                            for jth in range(len(train_test_predefined[0][1]))

                                        ],

                                    )

                                )

                    # if the train_test_predefined is not in the correct format then raise an error

                    except IndexError:

                        log.error(

                            "The train_test_predefined object is not in the correct format, it should be either a deepchem data object or a list of tuples of numpy arrays"

                        )

            def set_objectives(

                self,

                objectives: Optional[List[Callable]] = None,

                directions: Optional[List[str]] = None,

                add_train_test_diff_objective: bool = False,

            ) -> List[str]:

                """

                Set the objectives and directions for the optimization. These should be metrics with an interface of objective = func(y_true, y_pred) and direction to be one of ["minimize", "maximize"].

                If add_train_test_diff_objective is set to True the difference between the training and test set scores will be added as an objective to minimize. This provides a restraining metric to prevent overfitting.

                Args:

                    objectives (Optional[List[Callable]]): A list of objective functions to optimize

                    directions (Optional[List[str]]): A list of directions to optimize the objectives in

                    add_train_test_diff_objective (bool): Whether to add a restraining metric to prevent overfitting

                Returns:

                    List[str]: A list of directions to optimize the objectives

                """

                if objectives is None:

                    self.objectives = [root_mean_squared_error]

                    self.directions = ["minimize"]

                else:

                    self.objectives = objectives

                    self.directions = directions

                self.objective_values = np.zeros(len(self.objectives))

                self.add_train_test_diff_objective = add_train_test_diff_objective

                log.debug(self.objective_values)

                if add_train_test_diff_objective is True:

                    log.debug(f"After adding {self.objective_values}")

                    return np.array(self.directions + ["minimize"])

                else:

                    self.objective_values = np.array(self.objective_values)

                    return np.array(self.directions)

            # Make the plotting functions static methods so they can be used without an instance of the class

            @staticmethod

            def parity_plot(

                y_test,

                y_pred,

                xymin,

                xymax,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the parity plot of the test set predictions

                Args:

                    y_test (JoinKfoldData): The test set data

                    y_pred (np.ndarray): The predicted values

                    xymin (float): The minimum value for the x and y axis

                    xymax (float): The maximum value for the x and y axis

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The parity plot

                """

                # Least squares regression line

                m, c = np.polyfit(y_test, y_pred, deg=1)

                xseq = np.linspace(xymin, xymax, num=100)

                # plot the parity plot figure

                with plt.style.context(style=style):

                    fig = plt.figure(figsize=size)

                    ticks = np.arange(xymin, xymax + 1, 1.0)

                    plt.scatter(

                        y_test.ravel(),

                        y_pred.ravel(),

                        label="Prefect Prediction",

                        c="#89a0b0",

                        alpha=0.25,

                    )

                    plt.plot([xymin, xymax], [xymin, xymax], "k--", label="x = y")

                    plt.plot(

                        xseq, m * xseq + c, "m-.", lw=1.5, label="Least Squares Regression Line"

                    )  # y = mx + c

                    plt.scatter(

                        y_test,

                        y_pred.ravel(),

                        label=f"Model predictions RMSE: {root_mean_squared_error(y_test, y_pred):.2f} R2 Coefficent of determination {r2_score(y_test, y_pred):.2f}",

                    )

                    plt.grid()

                    plt.legend()

                    plt.xlabel("Experimental", fontsize=max(title_fontsize - 2, 10))

                    plt.ylabel("Prediction", fontsize=max(title_fontsize - 2, 10))

                    plt.title("Test Set Experimental Vs. Prediction", fontsize=title_fontsize)

                    ax = plt.gca()

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 2, 7)

                    )

                    ax.set_yticks(ticks)

                    ax.set_xticks(ticks)

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

            @staticmethod

            def plot_residuals(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the residuals of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.figure: The residuals plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    sns.residplot(

                        x=y_pred,

                        y=y_test - y_pred,

                        lowess=False,

                        ax=ax,

                        line_kws={"color": "orange", "lw": 1.5},

                    )

                    ax.axhline(y=0, color="black")

                    ax.set_title("Residual Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Prediction", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Residuals", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

            @staticmethod

            def plot_prediction_error(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the prediction error plot of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    style (str): The style of the plot

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The prediction error plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    ax.scatter(y_pred, y_test - y_pred)

                    ax.axhline(y=0, color="orange", linestyle="-.")

                    ax.set_title("Prediction Error Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Predictions", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Errors", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

            @staticmethod

            def plot_qq(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the QQ plot of the residuals of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    style (str): The style of the plot

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The QQ plot

                """

                log.critical(f"LOOK: {fname}")

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    scipy.stats.probplot(y_test - y_pred, dist="norm", plot=ax)

                    ax.set_title("QQ Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Theoretical Quantiles", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Ordered Values", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                        return None

                plt.close(fig)

                return fig

            def objectivefx(

                self,

                trial: optuna.trial.Trial,

                regressorfx: Union[Pipeline, Callable],

                parameters: dict,

                update_param_grid_callback: Optional[Callable] = None,

                name: Optional[str] = "kfold_study",

                experiment_id: Optional[int] = None,

                experiment_description: Optional[str] = None,

                without_mlflow: bool = False,

                **kwargs,

            ) -> Tuple[

                Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

            ]:

                """

                Function to do a k fold cross valuation over the data and return the mean scores for the objectives. This is built for use with optuna and the objective function should be a function that takes a trial object and returns the scores for the objectives.

                It should be used like:

                ```python

                cls = build_kfold_objective("test", 5)

                cls.get_data(X, y)

                directions = cls.set_objectives([root_mean_squared_error, r2_score], ["minimize", "maximize"])

                experiment_id = ml_flow_funcs.setup_for_mlflow("kold_study_4", utilities.mlflow_local_uri)

                pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))])

                study = optuna.create_study(directions=directions, study_name="test", storage=f'sqlite:///test.db', load_if_exists=True)

                func = lambda trial: cls.objectivefx(

                    trial,

                    pipe,

                    priors,

                    name="kfold_run",

                    experiment_id=experiment_id

                )

                study.optimize(

                    func,

                    n_trials=40,

                )

                ```

                Args:

                    trial (optuna.trial.Trial): The optuna trial object

                    regressorfx (Any): The regressor function to optimize

                    parameters (dict): The parameter grid to optimize over

                    update_param_grid_callback (Optional[Callable], optional): A callback function to update the parameter grid. Defaults to None.

                    name (Optional[str], optional): The name of the study. Defaults to "kfold_study".

                    experiment_id (Optional[int], optional): The experiment id for mlflow. Defaults to None.

                    experiment_description (Optional[str], optional): The experiment description for mlflow. Defaults to None.

                Returns:

                    Tuple[Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float]: The scores for the objectives

                """

                if without_mlflow is True:

                    log.critical("Training without MLFlow")

                    scores = self._objectivefx_without_mlflow(

                        trial,

                        regressorfx,

                        parameters,

                        update_param_grid_callback,

                        name,

                        **kwargs,

                    )

                else:

                    log.critical("Training with MLFlow")

                    scores = self._objectivefx_with_mlflow(

                        trial,

                        regressorfx,

                        parameters,

                        update_param_grid_callback,

                        name,

                        experiment_id,

                        experiment_description,

                        **kwargs,

                    )

                return scores

            def _objectivefx_without_mlflow(

                self,

                trial: optuna.trial.Trial,

                regressorfx: Union[Pipeline, Callable],

                parameters: dict,

                update_param_grid_callback: Optional[Callable] = None,

                name: Optional[str] = "tt_study",

                experiment_description: Optional[str] = None,

                **kwargs,

            ) -> Tuple[

                Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

            ]:

                """

                Function to do a k fold cross valuation over the data and return the mean scores for the objectives. This is built for use with optuna and the objective function should be a function that takes a trial object and returns the scores for the objectives.

                It should be used like:

                ```python

                cls = build_kfold_objective("test", 5)

                cls.get_data(X, y)

                directions = cls.set_objectives([root_mean_squared_error, r2_score], ["minimize", "maximize"])

                experiment_id = ml_flow_funcs.setup_for_mlflow("kold_study_4", utilities.mlflow_local_uri)

                pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))])

                study = optuna.create_study(directions=directions, study_name="test", storage=f'sqlite:///test.db', load_if_exists=True)

                func = lambda trial: cls.objectivefx(

                    trial,

                    pipe,

                    priors,

                    name="kfold_run",

                    experiment_id=experiment_id

                )

                study.optimize(

                    func,

                    n_trials=40,

                )

                ```

                Args:

                    trial (optuna.trial.Trial): The optuna trial object

                    regressorfx (Any): The regressor function to optimize

                    parameters (dict): The parameter grid to optimize over

                    update_param_grid_callback (Optional[Callable], optional): A callback function to update the parameter grid. Defaults to None.

                    name (Optional[str], optional): The name of the study. Defaults to "kfold_study".

                    experiment_description (Optional[str], optional): The experiment description for mlflow. Defaults to None.

                Returns:

                    Tuple[Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float]: The scores for the objectives

                """

                ptrain = Path(f"Train-test-training/train-{name}_{trial.number}")

                ptrain.mkdir(parents=True, exist_ok=True)

                ptest = Path(f"Train-test-training/test-{name}_{trial.number}")

                ptest.mkdir(parents=True, exist_ok=True)

                # get the parameter grid

                param_grid = build_param_grid(parameters, trial)

                log.debug(param_grid)

                with open(ptrain.joinpath("param_grid.json"), "w") as jout:

                    json.dump(param_grid, jout)

                # update the parameter grid if a callback is provided

                if update_param_grid_callback is not None:

                    param_grid = update_param_grid_callback(

                        trial, deepcopy(param_grid), **kwargs

                    )

                # set variables for the training scores to be saved per call

                train_scores = np.zeros_like(self.objective_values)

                test_scores = np.zeros_like(self.objective_values)

                log.debug(train_scores)

                log.debug(test_scores)

                # run the model trianing

                # set the regressor with the parameters

                if isinstance(regressorfx, Pipeline):

                    regressor = regressorfx.set_params(**param_grid)

                else:

                    regressor = regressorfx(**param_grid)

                # fit the model

                if len(self.train.y.shape) > 1:

                    regressor.fit(self.train.X, self.train.y.ravel())

                else:

                    regressor.fit(self.train.X, self.train.y)

                # get the training set predcition scores

                for jth, obj in enumerate(self.objectives):

                    log.debug(

                        f"Training set objective [{jth}]: {obj.__name__} Train scores shape: {train_scores.shape}"

                    )

                    train_scores[jth] = obj(self.train.y, regressor.predict(self.train.X))

                    log.debug(f"Training scores data frame {pd.DataFrame(train_scores)}")

                    log.debug(f"Training scores means {train_scores.mean(axis=0)}")

                    t_df = pd.DataFrame(

                        [train_scores],

                        columns=[metric.__name__ for metric in self.objectives],

                        index=["train"],

                    )

                    t_df.to_csv(ptrain.joinpath("train_scores.csv"))

                # get the test set predictions

                y_pred = regressor.predict(self.test.X)

                for jth, obj in enumerate(self.objectives):

                    test_scores[jth] = obj(self.test.y, y_pred)

                # plot the parity plot of the predictions

                xymin = np.floor(min(self.test.y.ravel().tolist() + y_pred.tolist()))

                xymax = np.ceil(max(self.test.y.ravel().tolist() + y_pred.tolist()))

                log.debug(f"min {xymin}, max {xymax}")

                _ = self.parity_plot(

                    self.test.y.ravel(),

                    y_pred,

                    xymin,

                    xymax,

                    fname=str(

                        ptest.joinpath(

                            "internal_test_set_parity_plot_train_test.png"

                        ).absolute()

                    ),

                )

                _ = self.plot_residuals(

                    self.test.y.ravel(),

                    y_pred,

                    fname=str(

                        ptest.joinpath(

                            "internal_test_set_residual_plot_train_test.png"

                        ).absolute()

                    ),

                )

                _ = self.plot_prediction_error(

                    self.test.y.ravel(),

                    y_pred,

                    fname=str(

                        ptest.joinpath("internal_test_set_error_plot_train_test.png").absolute()

                    ),

                )

                log.error(ptest.joinpath("internal_test_set_qq_plot_train_test.png"))

                _ = self.plot_qq(

                    self.test.y.ravel(),

                    y_pred,

                    fname=str(

                        ptest.joinpath("internal_test_set_qq_plot_train_test.png").absolute()

                    ),

                )

                if isinstance(regressorfx, Pipeline):

                    description = (

                        "Experiment to train an scikit-learn pipeline defined as: "

                        + " ".join([f"{k}: {v}" for k, v in regressor.named_steps.items()])

                        + ". "

                    )

                else:

                    description = f"Experiment to train a model {regressorfx.__name__} model with parameters: {param_grid}. "

                if experiment_description is not None:

                    description += experiment_description

                with open(ptrain.joinpath("description.txt"), "w") as fout:

                    fout.write(description)

                # Log to MLflow the parameters and the objective values

                for xth, obj in enumerate(self.objectives):

                    log.debug(train_scores[xth])

                    log.debug(test_scores[xth])

                tt_df = pd.DataFrame(

                    [train_scores, test_scores],

                    columns=[metric.__name__ for metric in self.objectives],

                    index=["train", "test"],

                )

                tt_df.to_csv(ptest.joinpath("train_test_scores.csv"))

                # add the absolute difference between the training and test set scores as an objective if requested then return the scores

                if self.add_train_test_diff_objective is True:

                    train_val_abs_diff = self.abs_train_test_diff_objective(

                        train_scores, test_scores

                    )

                    return *test_scores.tolist(), train_val_abs_diff

                else:

                    return test_scores.tolist()

            def _objectivefx_with_mlflow(

                self,

                trial: optuna.trial.Trial,

                regressorfx: Union[Pipeline, Callable],

                parameters: dict,

                update_param_grid_callback: Optional[Callable] = None,

                name: Optional[str] = "tt_study",

                experiment_id: Optional[int] = None,

                experiment_description: Optional[str] = None,

                **kwargs,

            ) -> Tuple[

                Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

            ]:

                """

                Function to do a k fold cross valuation over the data and return the mean scores for the objectives. This is built for use with optuna and the objective function should be a function that takes a trial object and returns the scores for the objectives.

                It should be used like:

                ```python

                cls = build_kfold_objective("test", 5)

                cls.get_data(X, y)

                directions = cls.set_objectives([root_mean_squared_error, r2_score], ["minimize", "maximize"])

                experiment_id = ml_flow_funcs.setup_for_mlflow("kold_study_4", utilities.mlflow_local_uri)

                pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))])

                study = optuna.create_study(directions=directions, study_name="test", storage=f'sqlite:///test.db', load_if_exists=True)

                func = lambda trial: cls.objectivefx(

                    trial,

                    pipe,

                    priors,

                    name="kfold_run",

                    experiment_id=experiment_id

                )

                study.optimize(

                    func,

                    n_trials=40,

                )

                ```

                Args:

                    trial (optuna.trial.Trial): The optuna trial object

                    regressorfx (Any): The regressor function to optimize

                    parameters (dict): The parameter grid to optimize over

                    update_param_grid_callback (Optional[Callable], optional): A callback function to update the parameter grid. Defaults to None.

                    name (Optional[str], optional): The name of the study. Defaults to "kfold_study".

                    experiment_id (Optional[int], optional): The experiment id for mlflow. Defaults to None.

                    experiment_description (Optional[str], optional): The experiment description for mlflow. Defaults to None.

                Returns:

                    Tuple[Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float]: The scores for the objectives

                """

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"Train-{name}_{trial.number}",

                    nested=True,

                ):

                    # get the parameter grid

                    param_grid = build_param_grid(parameters, trial)

                    log.debug(param_grid)

                    # update the parameter grid if a callback is provided

                    if update_param_grid_callback is not None:

                        param_grid = update_param_grid_callback(

                            trial, deepcopy(param_grid), **kwargs

                        )

                    # set variables for the training scores to be saved per call

                    train_scores = np.zeros_like(self.objective_values)

                    test_scores = np.zeros_like(self.objective_values)

                    log.debug(train_scores)

                    log.debug(test_scores)

                    # run the model trianing

                    # set the regressor with the parameters

                    if isinstance(regressorfx, Pipeline):

                        regressor = regressorfx.set_params(**param_grid)

                    else:

                        regressor = regressorfx(**param_grid)

                    # fit the model

                    if len(self.train.y.shape) > 1:

                        regressor.fit(self.train.X, self.train.y.ravel())

                    else:

                        regressor.fit(self.train.X, self.train.y)

                    # get the training set predcition scores

                    for jth, obj in enumerate(self.objectives):

                        log.debug(

                            f"Training set objective [{jth}]: {obj.__name__} Train scores shape: {train_scores.shape}"

                        )

                        train_scores[jth] = obj(self.train.y, regressor.predict(self.train.X))

                        log.debug(f"Training scores data frame {pd.DataFrame(train_scores)}")

                        log.debug(f"Training scores means {train_scores.mean(axis=0)}")

                    # get the test set predictions

                    y_pred = regressor.predict(self.test.X)

                    for jth, obj in enumerate(self.objectives):

                        test_scores[jth] = obj(self.test.y, y_pred)

                    # plot the parity plot of the predictions

                    xymin = np.floor(min(self.test.y.ravel().tolist() + y_pred.tolist()))

                    xymax = np.ceil(max(self.test.y.ravel().tolist() + y_pred.tolist()))

                    log.debug(f"min {xymin}, max {xymax}")

                    parity_plt = self.parity_plot(

                        self.test.y.ravel(),

                        y_pred,

                        xymin,

                        xymax,

                    )

                    mlflow.log_figure(

                        parity_plt, "internal_test_set_parity_plot_train_test.png"

                    )

                    residual_plt = self.plot_residuals(self.test.y.ravel(), y_pred)

                    mlflow.log_figure(

                        residual_plt, "internal_test_set_residual_plot_train_test.png"

                    )

                    err_plt = self.plot_prediction_error(self.test.y.ravel(), y_pred)

                    mlflow.log_figure(err_plt, "internal_test_set_error_plot_train_test.png")

                    qq_plt = self.plot_qq(self.test.y.ravel(), y_pred)

                    mlflow.log_figure(qq_plt, "internal_test_set_qq_plot_train_test.png")

                    if isinstance(regressorfx, Pipeline):

                        description = (

                            "Experiment to train an scikit-learn pipeline defined as: "

                            + " ".join([f"{k}: {v}" for k, v in regressor.named_steps.items()])

                            + ". "

                        )

                    else:

                        description = f"Experiment to train a model {regressorfx.__name__} model with parameters: {param_grid}. "

                    if experiment_description is not None:

                        description += experiment_description

                    mlflow.log_text(description, "description.txt")

                    # Log to MLflow the parameters and the objective values

                    mlflow.log_params(param_grid)

                    for xth, obj in enumerate(self.objectives):

                        log.debug(train_scores[xth])

                        log.debug(test_scores[xth])

                        mlflow.log_metric(f"{obj.__name__}_train", train_scores[xth])

                        mlflow.log_metric(f"{obj.__name__}_test", test_scores[xth])

                    # add the absolute difference between the training and test set scores as an objective if requested then return the scores

                    if self.add_train_test_diff_objective is True:

                        train_val_abs_diff = self.abs_train_test_diff_objective(

                            train_scores, test_scores

                        )

                        return *test_scores.tolist(), train_val_abs_diff

                    else:

                        return test_scores.tolist()



        class build_kfold_objective(object):

            def __init__(self, name: str, k: int = 5, **kwargs) -> None:

                self.name = name

                self.k = k

                self.cv_data = []

                self.cv_ids = []

                self.kwargs = kwargs

            def get_all_data_as_single_fold(

                self,

                kf: Optional[

                    List[

                        Tuple[

                            Union[np.ndarray, data.datasets.DiskDataset],

                            Union[np.ndarray, data.datasets.DiskDataset],

                        ]

                    ]

                ] = None,

            ) -> Tuple[np.ndarray, np.ndarray]:

                """

                Function to get all the data as a single fold. It combines each test set from the k fold into a single set. This is useful for training a model on all the data.

                The returned data is a tuple of the features and target values as numpy.

                Returns:

                    Tuple[np.ndarray, np.ndarray]: The features and target values

                """

                if kf is None:

                    X = np.concatenate([test.X for _, test in self.cv_data])

                    y = np.concatenate([test.y for _, test in self.cv_data])

                else:

                    X = np.concatenate([test.X for _, test in kf])

                    y = np.concatenate([test.y for _, test in kf])

                return X, y

            def abs_train_test_diff_objective(

                self, train_scores: np.ndarray, test_scores: np.ndarray

            ) -> float:

                """

                Function to calculate the difference between the training and test set scores. This is used as a restraining metric to prevent overfitting.

                Note this is the difference between the mean scores for all other objectives on all folds of the k fold.

                Args:

                    train_scores (np.ndarray): The training set scores

                    test_scores (np.ndarray): The test set scores

                Returns:

                    float: The absolute difference between the training and test set scores

                """

                return np.abs(np.mean(train_scores) - np.mean(test_scores))

            def build_param_grid(

                self, parameters, trial: optuna.trial.Trial, param_grid: Optional[dict] = None

            ) -> dict[Any, Any]:

                """

                Function to build the parameter grid for the optimization

                Args:

                    parameters (dict): The parameters to optimize over

                    trial (optuna.trial.Trial): The optuna trial object

                    param_grid (Optional[dict], optional): The parameter grid to update. Defaults to None.

                Returns:

                    dict[Any, Any]: The parameter grid

                """

                if param_grid is None:

                    param_grid = {}

                for k, v in parameters.items():

                    if v[0].lower().strip() == "float":

                        param_grid[k] = trial.suggest_float(k, v[1], v[2])

                    elif v[0].lower().strip() == "int":

                        param_grid[k] = trial.suggest_int(k, v[1], v[2])

                    if v[0].lower().strip() == "catagorical":

                        param_grid[k] = trial.suggest_categorical(k, v[1])

                return param_grid

            def get_data(

                self,

                X: Optional[Union[np.ndarray, pd.DataFrame]] = None,

                y: Optional[Union[np.ndarray, pd.Series]] = None,

                kf: Optional[

                    List[

                        Tuple[

                            Union[np.ndarray, data.datasets.DiskDataset],

                            Union[np.ndarray, data.datasets.DiskDataset],

                        ]

                    ]

                ] = None,

            ) -> None:

                """

                Data should be passed in as either lists or tuples containing numpy arrays or pandas dataframes/Series objects to X and y

                which will be randomly split into k folds or a predefined kfold object can be passed in as kf. The predefined kf should be

                either a deepchem kfold object or list of tuples of JoinKfoldData classes with X and y being numpy arrays. The data is then

                stored in the cv_data and cv_ids attributes of the class object.

                Args:

                    X (Optional[Union[np.ndarray, pd.DataFrame]]): The features

                    y (Optional[Union[np.ndarray, pd.Series]]): The target values

                    kf (Optional[List[Tuple[Union[np.ndarray, data.datasets.DiskDataset], Union[np.ndarray, data.datasets.DiskDataset]]]): The predefined kfold object

                """

                if X is not None and y is not None:

                    if isinstance(X, pd.DataFrame):

                        X = X.values

                    if isinstance(y, pd.Series):

                        y = y.values

                    cv = KFold(n_splits=self.k, **self.kwargs)

                    for train_indx, test_indx in cv.split(X, y):

                        self.cv_data.append(

                            (

                                JoinKfoldData(X=X[train_indx], y=y[train_indx]),

                                JoinKfoldData(X=X[test_indx], y=y[test_indx]),

                            )

                        )

                        self.cv_ids.append((train_indx, test_indx))

                elif kf is not None:

                    try:

                        if "deepchem" in str(type(kf[0][0])):

                            for trainf, testf in kf:

                                self.cv_data.append(

                                    (

                                        JoinKfoldData(X=trainf.X, y=trainf.y),

                                        JoinKfoldData(X=testf.X, y=testf.y),

                                    )

                                )

                                self.cv_ids.append((trainf.ids, testf.ids))

                        else:

                            if isinstance(kf[0][0], JoinKfoldData):

                                self.cv_data = kf

                                for ith, (trainf, testf) in enumerate(kf):

                                    self.cv_ids.append(

                                        (

                                            [

                                                f"fold_{ith}_train_row_{jth}"

                                                for jth in range(len(trainf.X))

                                            ],

                                            [

                                                f"fold_{ith}_testrow_{jth}"

                                                for jth in range(len(testf.X))

                                            ],

                                        )

                                    )

                            else:

                                log.warning(

                                    "The kfolds do not have a deepchem or JoinKfoldData object format will try to format"

                                )

                                for ith, (trainf, testf) in enumerate(kf):

                                    self.cv_data.append(

                                        (

                                            JoinKfoldData(X=trainf[0], y=trainf[1]),

                                            JoinKfoldData(X=testf[0], y=testf[1]),

                                        )

                                    )

                                    self.cv_ids.append(

                                        (

                                            [

                                                f"fold_{ith}_train_row_{jth}"

                                                for jth in range(len(trainf[0]))

                                            ],

                                            [

                                                f"fold_{ith}_testrow_{jth}"

                                                for jth in range(len(testf[0]))

                                            ],

                                        )

                                    )

                    except IndexError:

                        log.error(

                            "The kfolds object is not in the correct format, it should be either a deepchem kfold data object or a list of tuples of numpy arrays"

                        )

            def set_objectives(

                self,

                objectives: Optional[List[Callable]] = None,

                directions: Optional[List[str]] = None,

                add_train_test_diff_objective: bool = False,

            ) -> List[str]:

                """

                Set the objectives and directions for the optimization. These should be metrics with an interface of objective = func(y_true, y_pred) and direction to be one of ["minimize", "maximize"].

                If add_train_test_diff_objective is set to True the difference between the training and test set scores will be added as an objective to minimize. This provides a restraining metric to prevent overfitting.

                Args:

                    objectives (Optional[List[Callable]]): A list of objective functions to optimize

                    directions (Optional[List[str]]): A list of directions to optimize the objectives in

                    add_train_test_diff_objective (bool): Whether to add a restraining metric to prevent overfitting

                Returns:

                    List[str]: A list of directions to optimize the objectives

                """

                if objectives is None:

                    self.objectives = [root_mean_squared_error]

                    self.directions = ["minimize"]

                else:

                    self.objectives = objectives

                    self.directions = directions

                # self.objective_values = [np.zeros(self.k) for _ in range(len(self.objectives))]

                self.objective_values = [np.zeros(len(self.objectives)) for _ in range(self.k)]

                self.add_train_test_diff_objective = add_train_test_diff_objective

                log.debug(self.objective_values)

                if add_train_test_diff_objective is True:

                    # log.debug("LOOK HERE: ADDED TRAIN TEST DIFF OBJECTIVE")

                    # log.debug(f"Before adding {self.objective_values}")

                    # self.objective_values = np.array(

                    #     [np.append(ent, [0.0]) for ent in self.objective_values]

                    # )

                    log.debug(f"After adding {self.objective_values}")

                    # self.objectives += [self.abs_train_test_diff_objective]

                    return np.array(self.directions + ["minimize"])

                else:

                    log.info("Not added train test diff objective as requested")

                    self.objective_values = np.array(self.objective_values)

                    return np.array(self.directions)

            # Make the plotting functions static methods so they can be used without an instance of the class

            @staticmethod

            def parity_plot(

                y_test,

                y_pred,

                xymin,

                xymax,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the parity plot of the test set predictions

                Args:

                    y_test (JoinKfoldData): The test set data

                    y_pred (np.ndarray): The predicted values

                    xymin (float): The minimum value for the x and y axis

                    xymax (float): The maximum value for the x and y axis

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The parity plot

                """

                # Least squares regression line

                m, c = np.polyfit(y_test, y_pred, deg=1)

                xseq = np.linspace(xymin, xymax, num=100)

                # plot the parity plot figure

                with plt.style.context(style=style):

                    fig = plt.figure(figsize=size)

                    ticks = np.arange(xymin, xymax + 1, 1.0)

                    plt.scatter(

                        y_test.ravel(),

                        y_pred.ravel(),

                        label="Prefect Prediction",

                        c="#89a0b0",

                        alpha=0.25,

                    )

                    plt.plot([xymin, xymax], [xymin, xymax], "k--", label="x = y")

                    plt.plot(

                        xseq, m * xseq + c, "m-.", lw=1.5, label="Least Squares Regression Line"

                    )  # y = mx + c

                    plt.scatter(

                        y_test,

                        y_pred.ravel(),

                        label=f"Model predictions RMSE: {root_mean_squared_error(y_test, y_pred):.2f} R2 Coefficent of determination {r2_score(y_test, y_pred):.2f}",

                    )

                    plt.grid()

                    plt.legend()

                    plt.xlabel("Experimental", fontsize=max(title_fontsize - 2, 10))

                    plt.ylabel("Prediction", fontsize=max(title_fontsize - 2, 10))

                    plt.title("Test Set Experimental Vs. Prediction", fontsize=title_fontsize)

                    ax = plt.gca()

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 2, 7)

                    )

                    ax.set_yticks(ticks)

                    ax.set_xticks(ticks)

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

            @staticmethod

            def plot_residuals(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the residuals of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.figure: The residuals plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    sns.residplot(

                        x=y_pred,

                        y=y_test - y_pred,

                        lowess=False,

                        ax=ax,

                        line_kws={"color": "orange", "lw": 1.5},

                    )

                    ax.axhline(y=0, color="black")

                    ax.set_title("Residual Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Prediction", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Residuals", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

            @staticmethod

            def plot_prediction_error(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the prediction error plot of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    style (str): The style of the plot

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The prediction error plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    ax.scatter(y_pred, y_test - y_pred)

                    ax.axhline(y=0, color="orange", linestyle="-.")

                    ax.set_title("Prediction Error Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Predictions", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Errors", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

            @staticmethod

            def plot_qq(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the QQ plot of the residuals of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    style (str): The style of the plot

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The QQ plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    scipy.stats.probplot(y_test - y_pred, dist="norm", plot=ax)

                    ax.set_title("QQ Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Theoretical Quantiles", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Ordered Values", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

            def objectivefx(

                self,

                trial: optuna.trial.Trial,

                regressorfx: Union[Pipeline, Callable],

                parameters: dict,

                update_param_grid_callback: Optional[Callable] = None,

                name: Optional[str] = "kfold_study",

                experiment_id: Optional[int] = None,

                experiment_description: Optional[str] = None,

                without_mlflow: bool = False,

                **kwargs,

            ) -> Tuple[

                Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

            ]:

                """

                Function to do a k fold cross valuation over the data and return the mean scores for the objectives. This is built for use with optuna and the objective function should be a function that takes a trial object and returns the scores for the objectives.

                It should be used like:

                ```python

                cls = build_kfold_objective("test", 5)

                cls.get_data(X, y)

                directions = cls.set_objectives([root_mean_squared_error, r2_score], ["minimize", "maximize"])

                experiment_id = ml_flow_funcs.setup_for_mlflow("kold_study_4", utilities.mlflow_local_uri)

                pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))])

                study = optuna.create_study(directions=directions, study_name="test", storage=f'sqlite:///test.db', load_if_exists=True)

                func = lambda trial: cls.objectivefx(

                    trial,

                    pipe,

                    priors,

                    name="kfold_run",

                    experiment_id=experiment_id

                )

                study.optimize(

                    func,

                    n_trials=40,

                )

                ```

                Args:

                    trial (optuna.trial.Trial): The optuna trial object

                    regressorfx (Any): The regressor function to optimize

                    parameters (dict): The parameter grid to optimize over

                    update_param_grid_callback (Optional[Callable], optional): A callback function to update the parameter grid. Defaults to None.

                    name (Optional[str], optional): The name of the study. Defaults to "kfold_study".

                    experiment_id (Optional[int], optional): The experiment id for mlflow. Defaults to None.

                    experiment_description (Optional[str], optional): The experiment description for mlflow. Defaults to None.

                    with_mlflow (bool): Whether to use mlflow or not

                Returns:

                    Tuple[Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float]: The scores for the objectives

                """

                if without_mlflow is False:

                    return self._objectivefx_with_mlflow(

                        trial,

                        regressorfx,

                        parameters,

                        update_param_grid_callback,

                        name,

                        experiment_id,

                        experiment_description,

                        **kwargs,

                    )

                else:

                    return self._objectivefx_without_mlflow(

                        trial,

                        regressorfx,

                        parameters,

                        update_param_grid_callback,

                        name,

                        experiment_description,

                        **kwargs,

                    )

            def _objectivefx_with_mlflow(

                self,

                trial: optuna.trial.Trial,

                regressorfx: Union[Pipeline, Callable],

                parameters: dict,

                update_param_grid_callback: Optional[Callable] = None,

                name: Optional[str] = "kfold_study",

                experiment_id: Optional[int] = None,

                experiment_description: Optional[str] = None,

                **kwargs,

            ) -> Tuple[

                Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

            ]:

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"Train-{name}_{trial.number}",

                    nested=True,

                ):

                    # get the parameter grid

                    param_grid = build_param_grid(parameters, trial)

                    log.debug(param_grid)

                    # update the parameter grid if a callback is provided

                    if update_param_grid_callback is not None:

                        param_grid = update_param_grid_callback(

                            trial, deepcopy(param_grid), **kwargs

                        )

                    # set variables for the training scores to be saved per call

                    train_scores = np.zeros_like(self.objective_values)

                    test_scores = np.zeros_like(self.objective_values)

                    log.debug(train_scores)

                    log.debug(test_scores)

                    # run the cross validation folds

                    for ith, (train, test) in enumerate(self.cv_data):

                        # set the regressor with the parameters

                        if isinstance(regressorfx, Pipeline):

                            regressor = regressorfx.set_params(**param_grid)

                        else:

                            regressor = regressorfx(**param_grid)

                        # fit the model

                        if len(train.y.shape) > 1:

                            regressor.fit(train.X, train.y.ravel())

                        else:

                            regressor.fit(train.X, train.y)

                        # get the training set predcition scores

                        for jth, obj in enumerate(self.objectives):

                            log.debug(

                                f"Training set objective [{ith}{jth}]: {obj.__name__} Train scores shape: {train_scores.shape}"

                            )

                            train_scores[ith, jth] = obj(train.y, regressor.predict(train.X))

                            log.debug(

                                f"Training scores data frame {pd.DataFrame(train_scores)}"

                            )

                            log.debug(f"Training scores means {train_scores.mean(axis=0)}")

                        # get the test set predictions

                        y_pred = regressor.predict(test.X)

                        for jth, obj in enumerate(self.objectives):

                            test_scores[ith, jth] = obj(test.y, y_pred)

                        # plot the parity plot of the predictions

                        xymin = np.floor(min(test.y.ravel().tolist() + y_pred.tolist()))

                        xymax = np.ceil(max(test.y.ravel().tolist() + y_pred.tolist()))

                        log.debug(f"min {xymin}, max {xymax}")

                        parity_plt = self.parity_plot(test.y.ravel(), y_pred, xymin, xymax)

                        mlflow.log_figure(

                            parity_plt, f"internal_test_set_parity_plot_fold_{ith}.png"

                        )

                        residual_plt = self.plot_residuals(test.y.ravel(), y_pred)

                        mlflow.log_figure(

                            residual_plt, f"internal_test_set_residual_plot_fold_{ith}.png"

                        )

                        err_plt = self.plot_prediction_error(test.y.ravel(), y_pred)

                        mlflow.log_figure(

                            err_plt, f"internal_test_set_error_plot_fold_{ith}.png"

                        )

                        qq_plt = self.plot_qq(test.y.ravel(), y_pred)

                        mlflow.log_figure(qq_plt, f"internal_test_set_qq_plot_fold_{ith}.png")

                    if isinstance(regressorfx, Pipeline):

                        description = (

                            "Experiment to train an scikit-learn pipeline defined as: "

                            + " ".join([f"{k}: {v}" for k, v in regressor.named_steps.items()])

                            + ". "

                        )

                    else:

                        description = f"Experiment to train a model {regressorfx.__name__} model with parameters: {param_grid}. "

                    if experiment_description is not None:

                        description += experiment_description

                    mlflow.log_text(description, "description.txt")

                    # get the mean scores for the cross validation

                    train_cv_scores = train_scores.mean(axis=0)

                    test_cv_scores = test_scores.mean(axis=0)

                    train_cv_scores_std = train_scores.std(axis=0)

                    test_cv_scores_std = test_scores.std(axis=0)

                    # Log to MLflow the parameters and the objective values

                    mlflow.log_params(param_grid)

                    for xth, obj in enumerate(self.objectives):

                        log.debug(train_cv_scores[xth])

                        log.debug(test_cv_scores[xth])

                        mlflow.log_metric(f"{obj.__name__}_train", train_cv_scores[xth])

                        mlflow.log_metric(f"{obj.__name__}_test", test_cv_scores[xth])

                        mlflow.log_metric(f"{obj.__name__}_std_train", train_cv_scores_std[xth])

                        mlflow.log_metric(f"{obj.__name__}_std_test", test_cv_scores_std[xth])

                    # add the absolute difference between the training and test set scores as an objective if requested then return the scores

                    if self.add_train_test_diff_objective is True:

                        train_val_abs_diff = self.abs_train_test_diff_objective(

                            train_scores, test_scores

                        )

                        return *test_cv_scores.tolist(), train_val_abs_diff

                    else:

                        return test_cv_scores.tolist()

            def _objectivefx_without_mlflow(

                self,

                trial: optuna.trial.Trial,

                regressorfx: Union[Pipeline, Callable],

                parameters: dict,

                update_param_grid_callback: Optional[Callable] = None,

                name: Optional[str] = "kfold_study",

                experiment_description: Optional[str] = None,

                **kwargs,

            ) -> Tuple[

                Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

            ]:

                ptrain = Path(f"Kfold-training/train-{name}_{trial.number}")

                ptrain.mkdir(parents=True, exist_ok=True)

                ptest = Path(f"Kfold-training/test-{name}_{trial.number}")

                ptest.mkdir(parents=True, exist_ok=True)

                # get the parameter grid

                param_grid = build_param_grid(parameters, trial)

                log.debug(param_grid)

                with open(ptrain.joinpath("param_grid.json"), "w") as jout:

                    json.dump(param_grid, jout)

                # update the parameter grid if a callback is provided

                if update_param_grid_callback is not None:

                    param_grid = update_param_grid_callback(

                        trial, deepcopy(param_grid), **kwargs

                    )

                # set variables for the training scores to be saved per call

                train_scores = np.zeros_like(self.objective_values)

                test_scores = np.zeros_like(self.objective_values)

                log.debug(train_scores)

                log.debug(test_scores)

                # run the cross validation folds

                for ith, (train, test) in enumerate(self.cv_data):

                    # set the regressor with the parameters

                    if isinstance(regressorfx, Pipeline):

                        regressor = regressorfx.set_params(**param_grid)

                    else:

                        regressor = regressorfx(**param_grid)

                    # fit the model

                    if len(train.y.shape) > 1:

                        regressor.fit(train.X, train.y.ravel())

                    else:

                        regressor.fit(train.X, train.y)

                    # get the training set predcition scores

                    for jth, obj in enumerate(self.objectives):

                        log.debug(

                            f"Training set objective [{ith}{jth}]: {obj.__name__} Train scores shape: {train_scores.shape}"

                        )

                        train_scores[ith, jth] = obj(train.y, regressor.predict(train.X))

                        log.debug(f"Training scores data frame {pd.DataFrame(train_scores)}")

                        log.debug(f"Training scores means {train_scores.mean(axis=0)}")

                    t_df = pd.DataFrame(

                        train_scores,

                        columns=[metric.__name__ for metric in self.objectives],

                        index=[f"train_{xth}" for xth in range(self.k)],

                    )

                    t_df.to_csv(ptrain.joinpath("train_scores.csv"))

                    # get the test set predictions

                    y_pred = regressor.predict(test.X)

                    for jth, obj in enumerate(self.objectives):

                        test_scores[ith, jth] = obj(test.y, y_pred)

                    test_df = pd.DataFrame(

                        test_scores,

                        columns=[metric.__name__ for metric in self.objectives],

                        index=[f"test_{xth}" for xth in range(self.k)],

                    )

                    test_df.to_csv(ptest.joinpath("test_scores.csv"))

                    # plot the parity plot of the predictions

                    xymin = np.floor(min(test.y.ravel().tolist() + y_pred.tolist()))

                    xymax = np.ceil(max(test.y.ravel().tolist() + y_pred.tolist()))

                    log.debug(f"min {xymin}, max {xymax}")

                    _ = self.parity_plot(

                        test.y.ravel(),

                        y_pred,

                        xymin,

                        xymax,

                        fname=str(

                            ptest.joinpath(

                                f"internal_test_set_parity_plot_fold_{ith}.png"

                            ).absolute()

                        ),

                    )

                    _ = self.plot_residuals(

                        test.y.ravel(),

                        y_pred,

                        fname=str(

                            ptest.joinpath(

                                f"internal_test_set_residual_plot_fold_{ith}.png"

                            ).absolute()

                        ),

                    )

                    _ = self.plot_prediction_error(

                        test.y.ravel(),

                        y_pred,

                        fname=str(

                            ptest.joinpath(

                                f"internal_test_set_error_plot_fold_{ith}.png"

                            ).absolute()

                        ),

                    )

                    _ = self.plot_qq(

                        test.y.ravel(),

                        y_pred,

                        fname=str(

                            ptest.joinpath(

                                f"internal_test_set_qq_plot_fold_{ith}.png"

                            ).absolute()

                        ),

                    )

                if isinstance(regressorfx, Pipeline):

                    description = (

                        "Experiment to train an scikit-learn pipeline defined as: "

                        + " ".join([f"{k}: {v}" for k, v in regressor.named_steps.items()])

                        + ". "

                    )

                else:

                    description = f"Experiment to train a model {regressorfx.__name__} model with parameters: {param_grid}. "

                if experiment_description is not None:

                    description += experiment_description

                with open(ptrain.joinpath("description.txt"), "w") as fout:

                    fout.write(description)

                # get the mean scores for the cross validation

                train_cv_scores = train_scores.mean(axis=0)

                test_cv_scores = test_scores.mean(axis=0)

                train_cv_scores_std = train_scores.std(axis=0)

                test_cv_scores_std = test_scores.std(axis=0)

                for xth, obj in enumerate(self.objectives):

                    log.debug(train_cv_scores[xth])

                    log.debug(test_cv_scores[xth])

                tt_df = pd.DataFrame(

                    [train_cv_scores, train_cv_scores_std, test_cv_scores, test_cv_scores_std],

                    columns=[metric.__name__ for metric in self.objectives],

                    index=["train mean", "train std dev", "test mean", "test std dev"],

                )

                tt_df.to_csv(ptest.joinpath("train_test_scores.csv"))

                # add the absolute difference between the training and test set scores as an objective if requested then return the scores

                if self.add_train_test_diff_objective is True:

                    train_val_abs_diff = self.abs_train_test_diff_objective(

                        train_scores, test_scores

                    )

                    return *test_cv_scores.tolist(), train_val_abs_diff

                else:

                    return test_cv_scores.tolist()



        class WrappedKNeighbors(object):

            def __init__(self, model) -> None:

                self.model = model

            def predict(self, X: np.ndarray, n_neighbors: int = 1, **kwargs) -> np.ndarray:

                distances, _ = self.model.kneighbors(X, n_neighbors=n_neighbors, **kwargs)

                return distances



        def optimize_dnn_arch_and_train(

            trial: optuna.trial.Trial,

            assay_targets: List[str],

            decreasing_neurons_only: bool = False,

            uncertainty: bool = True,

            epochs: int = 100,

            fit_transformers: Optional[List[dc.trans.Transformer]] = None,

            train_set: Optional[dc.data.data_loader.DataLoader] = None,

            valid_set: Optional[dc.data.data_loader.DataLoader] = None,

            test_set: Optional[dc.data.data_loader.DataLoader] = None,

            df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            name_column: Optional[str] = None,

            parameters: Optional[dict] = None,

            verbose: bool = False,

        ):

            """ """

            if parameters is None:

                parameters = {

                    "n_hidden_layers": ("int", 1, 4),

                    "n_neurons": ("int", 100, 1000),

                    "dropout_rate": ("float", 0.1, 1.0),

                    "activation": ("categorical", ["relu", "sigmoid", "tanh"]),

                    "learning_rate": ("float", 1e-6, 1e-2),

                    "batch_size": ("int", 4, 256),

                }

            # Ensures all the expected keys are present in the parameters dictionary

            else:

                default_parameters = {

                    "n_hidden_layers": ("int", 1, 4),

                    "n_neurons": ("int", 100, 1000),

                    "dropout_rate": ("float", 0.1, 1.0),

                    "activation": ("categorical", ["relu", "sigmoid", "tanh"]),

                    "learning_rate": ("float", 1e-6, 1e-2),

                    "batch_size": ("int", 4, 128),

                }

                parameters = {**default_parameters, **parameters}

            param_grid = {}

            param_grid["n_hidden_layers"] = trial.suggest_int(

                "n_hidden_layers",

                parameters["n_hidden_layers"][1],

                parameters["n_hidden_layers"][2],

            )

            param_grid["learning_rate"] = trial.suggest_float(

                "learning_rate", parameters["learning_rate"][1], parameters["learning_rate"][2]

            )

            param_grid["batch_size"] = trial.suggest_int(

                "batch_size", parameters["batch_size"][1], parameters["batch_size"][2]

            )

            layer_sizes = []

            activation_funcs = []

            drop_out = []

            for ith in range(param_grid["n_hidden_layers"]):

                if decreasing_neurons_only is True and ith > 0:

                    lower = max(int(layer_sizes / 10), parameters["n_neurons"][1])

                    upper = max(int(layer_sizes / 2), parameters["n_neurons"][1])

                else:

                    lower = parameters["n_neurons"][1]

                    upper = parameters["n_neurons"][2]

                layer_sizes.append(trial.suggest_int(f"n_neurons_{ith}", lower, upper))

                activation_funcs.append(

                    trial.suggest_categorical(f"activation_{ith}", parameters["activation"][1])

                )

                drop_out.append(

                    trial.suggest_float(

                        f"dropout_rate_{ith}",

                        parameters["dropout_rate"][1],

                        parameters["dropout_rate"][2],

                    )

                )

            param_grid["hidden_layer_sizes"] = layer_sizes

            param_grid["activation"] = activation_funcs

            param_grid["dropout_rate"] = drop_out

            if verbose is True:

                log.info(f"Parameter grid: {param_grid}")

            if train_set is None and test_set is None and df is not None:

                _, _, _, _, test_set_df = deep_net_models.train_multitask_regressor(

                    tasks=assay_targets,

                    data_df=df,

                    fit_transformers=fit_transformers,

                    smiles_column=smiles_column,

                    ids_column=name_column,

                    layer_sizes=param_grid["hidden_layer_sizes"],

                    epochs=epochs,

                    batch_size=param_grid["batch_size"],

                    learning_rate=param_grid["learning_rate"],

                    dropout_rate=param_grid["dropout_rate"],

                    activation_fns=param_grid["activation"],

                    uncertainty=uncertainty,

                )

                error = test_set_df.loc["RMS", "mean over tasks"]

                return error

            elif train_set is not None and test_set is not None and df is None:

                _, _, _, _, test_set_df = deep_net_models.train_multitask_regressor(

                    tasks=assay_targets,

                    train_dataset=train_set,

                    test_dataset=test_set,

                    valid_dataset=valid_set,

                    fit_transformers=fit_transformers,

                    layer_sizes=param_grid["hidden_layer_sizes"],

                    epochs=epochs,

                    batch_size=param_grid["batch_size"],

                    learning_rate=param_grid["learning_rate"],

                    dropout_rate=param_grid["dropout_rate"],

                    activation_fns=param_grid["activation"],

                    uncertainty=uncertainty,

                )

                error = test_set_df.loc["RMS", "mean over tasks"]

                return error

            else:

                log.critical("NO TRAINING IS HAPPENING")



        def run_train_test_study_with_mlflow(

            models: List[classical_ml_models.skmodel],

            cls: build_train_test_objective,

            priors_dict: Optional[dict],

            holdout_set: Union[

                JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset

            ],

            num_trials: int,

            directions: List[str],

            mlflow_experiment_name: str,

            experiment_id: Optional[int] = None,

            pipeline_list: Optional[List[Tuple[str, Callable]]] = None,

            local_abs_store_path: Optional[str] = None,

            model_key_in_pipeline: Optional[str] = "model",

            sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(

                seed=utilities.random_seed,

            ),

            training_smiles: Optional[List[str]] = None,

            experiment_description: Optional[str] = None,

        ) -> tuple[list[Any], int]:

            """

            Function to run a kfold study with mlflow logging the results and models for each trial

            Args:

                models (List[classical_ml_models.skmodel]): The models to run the kfold study on

                cls (build_train_test_objective): The class object to run the train test study

                pipe_or_model (Union[Pipeline, Callable]): The pipeline or model to optimize

                priors_dict (Optional[dict]): The priors to optimize over

                holdout_set (Union[JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset]): The holdout set to validate the models

                num_trials (int): The number of trials to run

                experiment_id (int): The mlflow experiment id

                directions (List[str]): The directions to optimize the objectives in

                mlflow_experiment_name (str): The mlflow experiment name

                local_abs_store_path (Optional[str], optional): The local absolute path to store the models. Defaults to None.

                model_key_in_pipeline (Optional[str], optional): The key to prepend to the model parameters in the pipeline. Defaults to "model__".

                sampler (optuna.samplers.BaseSampler, optional): The sampler to use for the optimization. Defaults to optuna.samplers.TPESampler(seed=utilities.random_seed).

            Returns:

                tuple[list[Any], int]: The studies and the experiment id

            """

            if local_abs_store_path is not None:

                store_path = Path(local_abs_store_path)

                if not store_path.exists():

                    store_path.mkdir(parents=True)

            experiment_id = ml_flow_funcs.setup_for_mlflow(

                mlflow_experiment_name, utilities.mlflow_local_uri

            )

            # if training_smiles is given automatically create an applicability domain based on Tanimoto distance/similarity model and log it to mlflow.

            # It is the same for all prediction models.

            if training_smiles is not None:

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name="Applicability_domain",

                    nested=True,

                ):

                    log.debug(f"Training smiles: {training_smiles}")

                    admodel = applicability_domain.get_tanimoto_ad_model(

                        training_smiles=training_smiles,

                        radius=2,

                        hash_length=1024,

                        algorithm="brute",

                    )

                    wadmodel = WrappedKNeighbors(admodel)

                    example_input = ml_featurization.get_ecfp(

                        smiles=["c1ccccc1", "CCCCC", "C(CCN)C(=O)O", "c1ccccc1NC(=O)C"],

                        radius=2,

                        hash_length=1024,

                        return_np=True,

                    )

                    mlflow.sklearn.log_model(

                        sk_model=wadmodel,

                        artifact_path="applicability_domain_model",

                        signature=mlflow.models.infer_signature(

                            example_input, wadmodel.predict(example_input)

                        ),

                        registered_model_name="applicability_domain_kneighbours_tanimoto_distance_model",

                    )

                    mlflow.log_metric("number_of_training_points", admodel.n_samples_fit_)

                    mlflow.log_metric("number_of_features", admodel.n_features_in_)

            studies = []

            for m in models:

                if not isinstance(m, classical_ml_models.skmodel):

                    log.error(f"Model {m} is not an instance of skmodel")

                    continue

                date = datetime.now().strftime("%Y-%m-%d")

                date_and_time = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M")

                if local_abs_store_path is not None:

                    model_store_path = store_path.joinpath(f"kfold_{m.name}_{date}")

                    if not store_path.exists():

                        model_store_path.mkdir(parents=True)

                if priors_dict is None:

                    priors = m.default_param_range_priors

                else:

                    priors = priors_dict

                if pipeline_list is not None:

                    pipe_or_model = Pipeline(pipeline_list + [(model_key_in_pipeline, m.model)])

                else:

                    pipe_or_model = m.model

                log.info(f"Model: {pipe_or_model}")

                if isinstance(pipe_or_model, Pipeline):

                    priors = utilities.prepend_dictionary_keys(

                        priors, prepend=f"{model_key_in_pipeline}__"

                    )

                log.info(f"Priors: {priors}")

                name = f"train_test_{m.name}_{str(sampler)}"

                if local_abs_store_path is not None:

                    study = optuna.create_study(

                        study_name=f"library_prod_{name}",

                        sampler=sampler,

                        storage=f"sqlite:///{model_store_path.joinpath(f'library_prod_{name}_{date_and_time}.db')}",

                        load_if_exists=True,

                        directions=directions,

                    )

                else:

                    study = optuna.create_study(

                        study_name=f"library_prod_{name}",

                        sampler=sampler,

                        storage=None,

                        load_if_exists=False,

                        directions=directions,

                    )

                log.info(

                    f"For model {m.name} we will run {num_trials} trials to optimize the hyperparameters"

                )

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"{m.name}_training_runs",

                    nested=True,

                ):

                    study.optimize(

                        lambda trial: cls.objectivefx(

                            trial,

                            pipe_or_model,

                            priors,

                            name=f"train_and_testing_{m.name}",

                            experiment_id=experiment_id,

                            experiment_description=experiment_description,

                        ),

                        n_trials=num_trials,

                        show_progress_bar=True,

                        gc_after_trial=False,

                    )

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"{m.name}_validation_runs",

                    nested=True,

                ):

                    # Account for the absolute difference between the training and test set scores if its an objective

                    n_standard_ojectives = len(cls.objectives)

                    if cls.add_train_test_diff_objective is True:

                        n_standard_ojectives += 1

                        target_names = [ent.__name__ for ent in cls.objectives] + [

                            "abs_train_test_diff"

                        ]

                    else:

                        target_names = [ent.__name__ for ent in cls.objectives]

                    if n_standard_ojectives <= 3:

                        ax_po = optuna.visualization.plot_pareto_front(

                            study, target_names=target_names

                        )

                        log.info(type(ax_po))

                        ax_po.write_image(f"pareto_front_{m.name}.png")

                        mlflow.log_artifact(f"pareto_front_{m.name}.png")

                    all_data_as_single_train_set_x, all_data_as_single_train_set_y = (

                        cls.get_all_data_as_single_set()

                    )

                    validation_ojective_values = []

                    validation_objective_indexes = []

                    for bs in study.best_trials:

                        log.info(f"Best trial: {bs.number} {bs.values}")

                        mod = pipe_or_model.set_params(**bs.params)

                        log.info(all_data_as_single_train_set_y.shape)

                        mod.fit(

                            all_data_as_single_train_set_x,

                            all_data_as_single_train_set_y.ravel(),

                        )

                        mlflow.sklearn.log_model(

                            sk_model=mod,

                            artifact_path=f"model_{bs.number}",

                            signature=mlflow.models.infer_signature(

                                all_data_as_single_train_set_x,

                                mod.predict(all_data_as_single_train_set_x),

                            ),

                            input_example=all_data_as_single_train_set_x,

                            registered_model_name=f"train_test_{m.name}_model_{bs.number}",

                        )

                        validation_set_pred = mod.predict(holdout_set.X)

                        validation_ojective_values.append(

                            [obj(holdout_set.y, validation_set_pred) for obj in cls.objectives]

                        )

                        validation_objective_indexes.append(f"{m.name}_{bs.number}")

                        log.info(

                            f"Validation set objective values for {validation_objective_indexes[-1]}: {validation_ojective_values[-1]}"

                        )

                        # plot the parity plot of the predictions

                        xymin = np.floor(

                            min(holdout_set.y.ravel().tolist() + validation_set_pred.tolist())

                        )

                        xymax = np.ceil(

                            max(holdout_set.y.ravel().tolist() + validation_set_pred.tolist())

                        )

                        log.info(f"min {xymin}, max {xymax}")

                        size = (10, 10)

                        title_fontsize = 27

                        parity_plt = build_kfold_objective.parity_plot(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            xymin,

                            xymax,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            parity_plt,

                            "external_validation_set_parity_plot_train_test_validation.png",

                        )

                        residual_plt = build_kfold_objective.plot_residuals(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            residual_plt,

                            "external_validation_set_residual_plot_train_test_validation.png",

                        )

                        err_plt = build_kfold_objective.plot_prediction_error(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            err_plt,

                            "external_validation_set_error_plot_train_test_validation.png",

                        )

                        qq_plt = build_kfold_objective.plot_qq(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            qq_plt, "external_validation_set_qq_plot_train_test_validation.png"

                        )

                    validation_metric_tab_df = pd.DataFrame(

                        validation_ojective_values,

                        columns=[ent.__name__ for ent in cls.objectives],

                        index=validation_objective_indexes,

                    )

                    validation_metric_tab_df.to_csv(

                        f"{m.name}_validation_metrics.csv", index_label="model"

                    )

                    validation_metric_tab_df.to_html(

                        f"{m.name}_validation_metrics.html", index_names=True

                    )

                    mlflow.log_artifact(f"{m.name}_validation_metrics.csv")

                    mlflow.log_artifact(f"{m.name}_validation_metrics.html")

                    studies.append(study)

            return studies, experiment_id



        def run_kfold_study_with_mlflow(

            models: List[classical_ml_models.skmodel],

            cls: build_kfold_objective,

            priors_dict: Optional[dict],

            pipeline_priors: Optional[dict],

            holdout_set: Union[

                JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset

            ],

            num_trials: int,

            directions: List[str],

            mlflow_experiment_name: str,

            experiment_id: Optional[int] = None,

            pipeline_list: Optional[List[Tuple[str, Callable]]] = None,

            local_abs_store_path: Optional[str] = None,

            model_key_in_pipeline: Optional[str] = "model",

            sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(

                seed=utilities.random_seed,

            ),

            training_smiles: Optional[List[str]] = None,

            experiment_description: Optional[str] = None,

        ) -> tuple[list[Any], int]:

            """

            Function to run a kfold study with mlflow logging the results and models for each trial

            Args:

                models (List[classical_ml_models.skmodel]): The models to run the kfold study on

                cls (build_kfold_objective): The class object to run the kfold study

                pipe_or_model (Union[Pipeline, Callable]): The pipeline or model to optimize

                priors_dict (Optional[dict]): The priors to optimize over

                pipeline_priors (Optional[dict]): The priors to optimize over for the pipeline

                holdout_set (Union[JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset]): The holdout set to validate the models

                num_trials (int): The number of trials to run

                experiment_id (int): The mlflow experiment id

                directions (List[str]): The directions to optimize the objectives in

                mlflow_experiment_name (str): The mlflow experiment name

                local_abs_store_path (Optional[str], optional): The local absolute path to store the models. Defaults to None.

                model_key_in_pipeline (Optional[str], optional): The key to prepend to the model parameters in the pipeline. Defaults to "model__".

                sampler (optuna.samplers.BaseSampler, optional): The sampler to use for the optimization. Defaults to optuna.samplers.TPESampler(seed=utilities.random_seed).

            Returns:

                tuple[list[Any], int]: The studies and the experiment id

            """

            if local_abs_store_path is not None:

                store_path = Path(local_abs_store_path)

                if not store_path.exists():

                    store_path.mkdir(parents=True)

            experiment_id = ml_flow_funcs.setup_for_mlflow(

                mlflow_experiment_name, utilities.mlflow_local_uri

            )

            # if training_smiles is given automatically create an applicability domain based on Tanimoto distance/similarity model and log it to mlflow.

            # It is the same for all prediction models.

            if training_smiles is not None:

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name="Applicability_domain",

                    nested=True,

                ):

                    log.debug(f"Training smiles: {training_smiles}")

                    admodel = applicability_domain.get_tanimoto_ad_model(

                        training_smiles=training_smiles,

                        radius=2,

                        hash_length=1024,

                        algorithm="brute",

                    )

                    wadmodel = WrappedKNeighbors(admodel)

                    example_input = ml_featurization.get_ecfp(

                        smiles=["c1ccccc1", "CCCCC", "C(CCN)C(=O)O", "c1ccccc1NC(=O)C"],

                        radius=2,

                        hash_length=1024,

                        return_np=True,

                    )

                    mlflow.sklearn.log_model(

                        sk_model=wadmodel,

                        artifact_path="applicability_domain_model",

                        signature=mlflow.models.infer_signature(

                            example_input, wadmodel.predict(example_input)

                        ),

                        registered_model_name="applicability_domain_kneighbours_tanimoto_distance_model",

                    )

                    mlflow.log_metric("number_of_training_points", admodel.n_samples_fit_)

                    mlflow.log_metric("number_of_features", admodel.n_features_in_)

            studies = []

            for m in models:

                if not isinstance(m, classical_ml_models.skmodel):

                    log.error(f"Model {m} is not an instance of skmodel")

                    continue

                date = datetime.now().strftime("%Y-%m-%d")

                date_and_time = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M")

                if local_abs_store_path is not None:

                    model_store_path = store_path.joinpath(f"kfold_{m.name}_{date}")

                    if not store_path.exists():

                        model_store_path.mkdir(parents=True)

                if priors_dict is None:

                    priors = m.default_param_range_priors

                else:

                    priors = priors_dict

                if pipeline_list is not None:

                    pipe_or_model = Pipeline(pipeline_list + [(model_key_in_pipeline, m.model)])

                else:

                    pipe_or_model = m.model

                log.info(f"Model: {pipe_or_model}")

                if isinstance(pipe_or_model, Pipeline):

                    priors = utilities.prepend_dictionary_keys(

                        priors, prepend=f"{model_key_in_pipeline}__"

                    )

                if pipeline_priors is not None:

                    priors = {**priors, **pipeline_priors}

                log.info(f"Priors: {priors}")

                name = f"kfold_{cls.k}_{m.name}_{str(sampler)}"

                if local_abs_store_path is not None:

                    study = optuna.create_study(

                        study_name=f"library_prod_{name}",

                        sampler=sampler,

                        storage=f"sqlite:///{model_store_path.joinpath(f'library_prod_{name}_{date_and_time}.db')}",

                        load_if_exists=True,

                        directions=directions,

                    )

                else:

                    study = optuna.create_study(

                        study_name=f"library_prod_{name}",

                        sampler=sampler,

                        storage=None,

                        load_if_exists=False,

                        directions=directions,

                    )

                log.info(

                    f"For model {m.name} we will run {num_trials} trials to optimize the hyperparameters"

                )

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"{m.name}_training_runs",

                    nested=True,

                ):

                    study.optimize(

                        lambda trial: cls.objectivefx(

                            trial,

                            pipe_or_model,

                            priors,

                            name=f"kfold_{m.name}",

                            experiment_id=experiment_id,

                            experiment_description=experiment_description,

                        ),

                        n_trials=num_trials,

                        show_progress_bar=True,

                        gc_after_trial=False,

                    )

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"{m.name}_validation_runs",

                    nested=True,

                ):

                    # Account for the absolute difference between the training and test set scores if its an objective

                    n_standard_ojectives = len(cls.objectives)

                    if cls.add_train_test_diff_objective is True:

                        n_standard_ojectives += 1

                        target_names = [ent.__name__ for ent in cls.objectives] + [

                            "abs_train_test_diff"

                        ]

                    else:

                        target_names = [ent.__name__ for ent in cls.objectives]

                    if n_standard_ojectives <= 3:

                        ax_po = optuna.visualization.plot_pareto_front(

                            study, target_names=target_names

                        )

                        log.info(type(ax_po))

                        ax_po.write_image(f"pareto_front_{m.name}.png")

                        mlflow.log_artifact(f"pareto_front_{m.name}.png")

                    all_data_as_single_train_fold_X, all_data_as_single_train_fold_y = (

                        cls.get_all_data_as_single_fold()

                    )

                    validation_ojective_values = []

                    validation_objective_indexes = []

                    for bs in study.best_trials:

                        log.info(f"Best trial: {bs.number} {bs.values}")

                        mod = pipe_or_model.set_params(**bs.params)

                        log.info(all_data_as_single_train_fold_y.shape)

                        mod.fit(

                            all_data_as_single_train_fold_X,

                            all_data_as_single_train_fold_y.ravel(),

                        )

                        mlflow.sklearn.log_model(

                            sk_model=mod,

                            artifact_path=f"model_{bs.number}",

                            signature=mlflow.models.infer_signature(

                                all_data_as_single_train_fold_X,

                                mod.predict(all_data_as_single_train_fold_X),

                            ),

                            input_example=all_data_as_single_train_fold_X,

                            registered_model_name=f"kfold_{m.name}_model_{bs.number}",

                        )

                        validation_set_pred = mod.predict(holdout_set.X)

                        validation_ojective_values.append(

                            [obj(holdout_set.y, validation_set_pred) for obj in cls.objectives]

                        )

                        validation_objective_indexes.append(f"{m.name}_{bs.number}")

                        log.info(

                            f"Validation set objective values for {validation_objective_indexes[-1]}: {validation_ojective_values[-1]}"

                        )

                        # plot the parity plot of the predictions

                        xymin = np.floor(

                            min(holdout_set.y.ravel().tolist() + validation_set_pred.tolist())

                        )

                        xymax = np.ceil(

                            max(holdout_set.y.ravel().tolist() + validation_set_pred.tolist())

                        )

                        log.info(f"min {xymin}, max {xymax}")

                        size = (10, 10)

                        title_fontsize = 27

                        parity_plt = build_kfold_objective.parity_plot(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            xymin,

                            xymax,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            parity_plt,

                            f"external_validation_set_parity_plot_fold_{bs.number}.png",

                        )

                        residual_plt = build_kfold_objective.plot_residuals(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            residual_plt,

                            f"external_validation_set_residual_plot_fold_{bs.number}.png",

                        )

                        err_plt = build_kfold_objective.plot_prediction_error(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            err_plt, f"external_validation_set_error_plot_fold_{bs.number}.png"

                        )

                        qq_plt = build_kfold_objective.plot_qq(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            qq_plt, f"external_validation_set_qq_plot_fold_{bs.number}.png"

                        )

                    validation_metric_tab_df = pd.DataFrame(

                        validation_ojective_values,

                        columns=[ent.__name__ for ent in cls.objectives],

                        index=validation_objective_indexes,

                    )

                    validation_metric_tab_df.to_csv(

                        f"{m.name}_validation_metrics.csv", index_label="model"

                    )

                    validation_metric_tab_df.to_html(

                        f"{m.name}_validation_metrics.html", index_names=True

                    )

                    mlflow.log_artifact(f"{m.name}_validation_metrics.csv")

                    mlflow.log_artifact(f"{m.name}_validation_metrics.html")

                    studies.append(study)

            return studies, experiment_id



        def build_param_grid(

            parameters, trial: optuna.trial.Trial, param_grid: Optional[dict] = None

        ) -> dict[Any, Any]:

            """

            Function to build the parameter grid for the optimization

            Args:

                parameters (dict): The parameters to optimize over

                trial (optuna.trial.Trial): The optuna trial object

                param_grid (Optional[dict], optional): The parameter grid to update. Defaults to None.

            Returns:

                dict[Any, Any]: The parameter grid

            """

            if param_grid is None:

                param_grid = {}

            for k, v in parameters.items():

                if v[0].lower().strip() == "float":

                    param_grid[k] = trial.suggest_float(k, v[1], v[2])

                elif v[0].lower().strip() == "int":

                    param_grid[k] = trial.suggest_int(k, v[1], v[2])

                if v[0].lower().strip() == "catagorical":

                    param_grid[k] = trial.suggest_categorical(k, v[1])

            return param_grid



        def objective_defined_train_test_single_rmse(

            trial: optuna.trial.Trial,

            regressor: Callable,

            parameters: dict,

            X_train: Union[pd.DataFrame, np.ndarray],

            X_test: Union[pd.DataFrame, np.ndarray],

            y_train: Union[pd.Series, np.ndarray],

            y_test: Union[pd.Series, np.ndarray],

        ) -> Union[np.float64, float, Any]:

            """

            Function to train a model on the train set and evaluate the metrics on a test set.

            This means the test set is used to optimize the hyper-parameters, therefore you MUST

            have a separate validation set tp test the model on.

            Args:

                trial (optuna.trial._trial.Trial): An Optuna trial set for hyper-parameters

                regressor (Callable): A regressor function or sklearn pipeline to set the parameter to from the trial

                parameters (dict): A dictionary of parameter ranges

                X_train (Union[pd.DataFrame, np.ndarray]): the training set and features

                X_test (Union[pd.DataFrame, np.ndarray]): the test set and features

                y_train (Union[pd.Series, np.ndarray]): the training set known target values for a property of interest

                y_test(Union[pd.Series, np.ndarray]): the test set  known target values for a property of interest

            Returns:

                Union[np.float64, float]: RMSE

            """

            param_grid = build_param_grid(parameters, trial)

            try:

                tmp_r = regressor()

                tmp_r.__getattribute__("random_state")

                param_grid = {"random_state": utilities.random_seed}

            except AttributeError:

                log.info(

                    "Regressor does not use a random_state for reproducibility on initialization"

                )

            regressor = regressor(**param_grid)

            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            error = np.sqrt(mean_squared_error(y_test, y_pred))

            return error



        def layer_size_to_network(

            trial: optuna.trial.Trial,

            param_grid: dict,

            n_layer_key: str = "n_layers",

            min_nodes: int = 10,

            max_nodes: int = 100,

            prepend: Optional[str] = None,

        ) -> dict:

            """

            _summary_

            Args:

                param_grid (dict): _description_

            Returns:

                dict: _description_

            """

            layers = []

            for ith in range(param_grid[n_layer_key]):

                layers.append(trial.suggest_int(f"{ith}_n_nodes", min_nodes, max_nodes))

            log.info(f"layers: {layers}")

            if prepend is None:

                param_grid["hidden_layer_sizes"] = tuple(layers)

            else:

                param_grid[f"{prepend}hidden_layer_sizes"] = tuple(layers)

            _ = param_grid.pop(n_layer_key)

            return param_grid



        def objective_random_cv_multi_rmse__r2__diff_train_test(

            trial: optuna.trial.Trial,

            regressorfx: Callable,

            parameters,

            X: Union[pd.DataFrame, np.ndarray],

            y: Union[pd.Series, np.ndarray],

            randomize_cv_split: bool = False,

            update_param_grid_callback: Optional[Callable] = None,

            k_fold: int = 5,

            **kwargs,

        ) -> Tuple[

            Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

        ]:

            """

            Function to do an internal random cv over X and y

            Args:

                trial (optuna.trial.Trial): _description_

                regressorfx (Callable): _description_

                parameters (_type_): _description_

                X (Union[pd.DataFrame, np.ndarray]): _description_

                y (Union[pd.Series, np.ndarray]): _description_

                randomize_cv_split (bool, optional): _description_. Defaults to False.

                update_param_grid_callback (Optional[Callable], optional): _description_. Defaults to None.

            Returns:

                Tuple[ Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float ]: _description_

            """

            param_grid = build_param_grid(parameters, trial)

            log.debug(param_grid)

            if update_param_grid_callback is not None:

                param_grid = update_param_grid_callback(trial, deepcopy(param_grid), **kwargs)

            if isinstance(X, np.ndarray):

                X = pd.DataFrame(X)

            if isinstance(y, np.ndarray):

                y = pd.Series(y)

            if randomize_cv_split is True:

                seed = utilities.random_seed * datetime.now().second

            else:

                seed = utilities.random_seed

            cv = KFold(n_splits=k_fold, shuffle=True, random_state=seed)

            cv_rmse_scores = np.empty(k_fold)

            cv_r2_scores = np.empty(k_fold)

            train_scores = np.empty(k_fold)

            for ith, (train_idx, test_idx) in enumerate(cv.split(X, y)):

                if isinstance(regressorfx, Pipeline):

                    regressor = regressorfx.set_params(**param_grid)

                else:

                    param_grid["random_state"] = utilities.random_seed

                    regressor = regressorfx(**param_grid)

                X_train = X.iloc[train_idx]

                X_test = X.iloc[test_idx]

                y_train = y[train_idx]

                y_test = y[test_idx]

                regressor.fit(X_train, y_train)

                y_pred = regressor.predict(X_test)

                log.error(X_train)

                log.error(y_train)

                train_scores[ith] = np.sqrt(

                    mean_squared_error(y_train, regressor.predict(X_train))

                )

                cv_r2_scores[ith] = r2_score(y_test, y_pred)

                cv_rmse_scores[ith] = np.sqrt(mean_squared_error(y_test, y_pred))

            mean_cv = np.mean(cv_rmse_scores)

            r2_coefficent_of_determination = np.mean(cv_r2_scores)

            train_val_abs_diff = np.abs(np.mean(train_scores) - np.mean(cv_rmse_scores))

            return mean_cv, r2_coefficent_of_determination, train_val_abs_diff



        def objective_predefined_cv_multi_rmse__r2__diff_train_test(

            trial: optuna.trial.Trial,

            regressorfx: Any,

            parameters: dict,

            predfined_kfolds: Tuple[np.ndarray],

            update_param_grid_callback: Optional[Callable] = None,

            mlflow_address: Optional[str] = None,

            **kwargs,

        ) -> Tuple[

            Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

        ]:

            """

            Function to do an internal random cv over X and y

            Args:

                trial (optuna.trial.Trial): _description_

                regressorfx (Callable): _description_

                parameters (_type_): _description_

                predfined_kfolds (Tuple[np.ndarray]): _description_

                update_param_grid_callback (Optional[Callable], optional): _description_. Defaults to None.

            Returns:

                Tuple[ Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float ]: _description_

            """

            param_grid = build_param_grid(parameters, trial)

            log.debug(param_grid)

            if update_param_grid_callback is not None:

                param_grid = update_param_grid_callback(trial, deepcopy(param_grid), **kwargs)

            cv_rmse_scores = np.empty(len(predfined_kfolds))

            cv_r2_scores = np.empty(len(predfined_kfolds))

            train_scores = np.empty(len(predfined_kfolds))

            for ith, (train, test) in enumerate(predfined_kfolds):

                if isinstance(regressorfx, Pipeline):

                    regressor = regressorfx.set_params(**param_grid)

                else:

                    param_grid["random_state"] = utilities.random_seed

                    regressor = regressorfx(**param_grid)

                if len(train.y.shape) > 1:

                    regressor.fit(train.X, train.y)

                else:

                    regressor.fit(train.X, train.y.ravel())

                y_pred = regressor.predict(test.X)

                # Get training set predcition scores to minimize the difference between this and the test set score

                train_scores[ith] = np.sqrt(

                    mean_squared_error(train.y, regressor.predict(train.X))

                )

                # regression metric scores coefficent of determination and RMSE

                if len(test.y.shape) > 1:

                    cv_r2_scores[ith] = r2_score(test.y, y_pred)

                    cv_rmse_scores[ith] = np.sqrt(mean_squared_error(test.y, y_pred))

                else:

                    cv_r2_scores[ith] = r2_score(test.y.ravel(), y_pred)

                    cv_rmse_scores[ith] = np.sqrt(mean_squared_error(test.y.ravel(), y_pred))

            mean_cv = np.mean(cv_rmse_scores)

            r2_coefficent_of_determination = np.mean(cv_r2_scores)

            train_val_abs_diff = np.abs(np.mean(train_scores) - np.mean(cv_rmse_scores))

            return mean_cv, r2_coefficent_of_determination, train_val_abs_diff

## Variables

```python3
log
```

## Functions


### build_param_grid

```python3
def build_param_grid(
    parameters,
    trial: optuna.trial._trial.Trial,
    param_grid: Optional[dict] = None
) -> dict[typing.Any, typing.Any]
```

Function to build the parameter grid for the optimization

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| parameters | dict | The parameters to optimize over | None |
| trial | optuna.trial.Trial | The optuna trial object | None |
| param_grid | Optional[dict] | The parameter grid to update. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| dict[Any, Any] | The parameter grid |

??? example "View Source"
        def build_param_grid(

            parameters, trial: optuna.trial.Trial, param_grid: Optional[dict] = None

        ) -> dict[Any, Any]:

            """

            Function to build the parameter grid for the optimization

            Args:

                parameters (dict): The parameters to optimize over

                trial (optuna.trial.Trial): The optuna trial object

                param_grid (Optional[dict], optional): The parameter grid to update. Defaults to None.

            Returns:

                dict[Any, Any]: The parameter grid

            """

            if param_grid is None:

                param_grid = {}

            for k, v in parameters.items():

                if v[0].lower().strip() == "float":

                    param_grid[k] = trial.suggest_float(k, v[1], v[2])

                elif v[0].lower().strip() == "int":

                    param_grid[k] = trial.suggest_int(k, v[1], v[2])

                if v[0].lower().strip() == "catagorical":

                    param_grid[k] = trial.suggest_categorical(k, v[1])

            return param_grid


### layer_size_to_network

```python3
def layer_size_to_network(
    trial: optuna.trial._trial.Trial,
    param_grid: dict,
    n_layer_key: str = 'n_layers',
    min_nodes: int = 10,
    max_nodes: int = 100,
    prepend: Optional[str] = None
) -> dict
```

_summary_

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| param_grid | dict | _description_ | None |

**Returns:**

| Type | Description |
|---|---|
| dict | _description_ |

??? example "View Source"
        def layer_size_to_network(

            trial: optuna.trial.Trial,

            param_grid: dict,

            n_layer_key: str = "n_layers",

            min_nodes: int = 10,

            max_nodes: int = 100,

            prepend: Optional[str] = None,

        ) -> dict:

            """

            _summary_

            Args:

                param_grid (dict): _description_

            Returns:

                dict: _description_

            """

            layers = []

            for ith in range(param_grid[n_layer_key]):

                layers.append(trial.suggest_int(f"{ith}_n_nodes", min_nodes, max_nodes))

            log.info(f"layers: {layers}")

            if prepend is None:

                param_grid["hidden_layer_sizes"] = tuple(layers)

            else:

                param_grid[f"{prepend}hidden_layer_sizes"] = tuple(layers)

            _ = param_grid.pop(n_layer_key)

            return param_grid


### objective_defined_train_test_single_rmse

```python3
def objective_defined_train_test_single_rmse(
    trial: optuna.trial._trial.Trial,
    regressor: Callable,
    parameters: dict,
    X_train: Union[pandas.core.frame.DataFrame, numpy.ndarray],
    X_test: Union[pandas.core.frame.DataFrame, numpy.ndarray],
    y_train: Union[pandas.core.series.Series, numpy.ndarray],
    y_test: Union[pandas.core.series.Series, numpy.ndarray]
) -> Union[numpy.float64, float, Any]
```

Function to train a model on the train set and evaluate the metrics on a test set.

This means the test set is used to optimize the hyper-parameters, therefore you MUST
have a separate validation set tp test the model on.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| trial | optuna.trial._trial.Trial | An Optuna trial set for hyper-parameters | None |
| regressor | Callable | A regressor function or sklearn pipeline to set the parameter to from the trial | None |
| parameters | dict | A dictionary of parameter ranges | None |
| X_train | Union[pd.DataFrame, np.ndarray] | the training set and features | None |
| X_test | Union[pd.DataFrame, np.ndarray] | the test set and features | None |
| y_train | Union[pd.Series, np.ndarray] | the training set known target values for a property of interest | None |
| y_test | Union[pd.Series, np.ndarray] | the test set  known target values for a property of interest | None |

**Returns:**

| Type | Description |
|---|---|
| Union[np.float64, float] | RMSE |

??? example "View Source"
        def objective_defined_train_test_single_rmse(

            trial: optuna.trial.Trial,

            regressor: Callable,

            parameters: dict,

            X_train: Union[pd.DataFrame, np.ndarray],

            X_test: Union[pd.DataFrame, np.ndarray],

            y_train: Union[pd.Series, np.ndarray],

            y_test: Union[pd.Series, np.ndarray],

        ) -> Union[np.float64, float, Any]:

            """

            Function to train a model on the train set and evaluate the metrics on a test set.

            This means the test set is used to optimize the hyper-parameters, therefore you MUST

            have a separate validation set tp test the model on.

            Args:

                trial (optuna.trial._trial.Trial): An Optuna trial set for hyper-parameters

                regressor (Callable): A regressor function or sklearn pipeline to set the parameter to from the trial

                parameters (dict): A dictionary of parameter ranges

                X_train (Union[pd.DataFrame, np.ndarray]): the training set and features

                X_test (Union[pd.DataFrame, np.ndarray]): the test set and features

                y_train (Union[pd.Series, np.ndarray]): the training set known target values for a property of interest

                y_test(Union[pd.Series, np.ndarray]): the test set  known target values for a property of interest

            Returns:

                Union[np.float64, float]: RMSE

            """

            param_grid = build_param_grid(parameters, trial)

            try:

                tmp_r = regressor()

                tmp_r.__getattribute__("random_state")

                param_grid = {"random_state": utilities.random_seed}

            except AttributeError:

                log.info(

                    "Regressor does not use a random_state for reproducibility on initialization"

                )

            regressor = regressor(**param_grid)

            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            error = np.sqrt(mean_squared_error(y_test, y_pred))

            return error


### objective_predefined_cv_multi_rmse__r2__diff_train_test

```python3
def objective_predefined_cv_multi_rmse__r2__diff_train_test(
    trial: optuna.trial._trial.Trial,
    regressorfx: Any,
    parameters: dict,
    predfined_kfolds: Tuple[numpy.ndarray],
    update_param_grid_callback: Optional[Callable] = None,
    mlflow_address: Optional[str] = None,
    **kwargs
) -> Tuple[Union[numpy.float64, float], Union[numpy.float64, float], float]
```

Function to do an internal random cv over X and y

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| trial | optuna.trial.Trial | _description_ | None |
| regressorfx | Callable | _description_ | None |
| parameters | _type_ | _description_ | None |
| predfined_kfolds | Tuple[np.ndarray] | _description_ | None |
| update_param_grid_callback | Optional[Callable] | _description_. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| Tuple[ Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float ] | _description_ |

??? example "View Source"
        def objective_predefined_cv_multi_rmse__r2__diff_train_test(

            trial: optuna.trial.Trial,

            regressorfx: Any,

            parameters: dict,

            predfined_kfolds: Tuple[np.ndarray],

            update_param_grid_callback: Optional[Callable] = None,

            mlflow_address: Optional[str] = None,

            **kwargs,

        ) -> Tuple[

            Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

        ]:

            """

            Function to do an internal random cv over X and y

            Args:

                trial (optuna.trial.Trial): _description_

                regressorfx (Callable): _description_

                parameters (_type_): _description_

                predfined_kfolds (Tuple[np.ndarray]): _description_

                update_param_grid_callback (Optional[Callable], optional): _description_. Defaults to None.

            Returns:

                Tuple[ Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float ]: _description_

            """

            param_grid = build_param_grid(parameters, trial)

            log.debug(param_grid)

            if update_param_grid_callback is not None:

                param_grid = update_param_grid_callback(trial, deepcopy(param_grid), **kwargs)

            cv_rmse_scores = np.empty(len(predfined_kfolds))

            cv_r2_scores = np.empty(len(predfined_kfolds))

            train_scores = np.empty(len(predfined_kfolds))

            for ith, (train, test) in enumerate(predfined_kfolds):

                if isinstance(regressorfx, Pipeline):

                    regressor = regressorfx.set_params(**param_grid)

                else:

                    param_grid["random_state"] = utilities.random_seed

                    regressor = regressorfx(**param_grid)

                if len(train.y.shape) > 1:

                    regressor.fit(train.X, train.y)

                else:

                    regressor.fit(train.X, train.y.ravel())

                y_pred = regressor.predict(test.X)

                # Get training set predcition scores to minimize the difference between this and the test set score

                train_scores[ith] = np.sqrt(

                    mean_squared_error(train.y, regressor.predict(train.X))

                )

                # regression metric scores coefficent of determination and RMSE

                if len(test.y.shape) > 1:

                    cv_r2_scores[ith] = r2_score(test.y, y_pred)

                    cv_rmse_scores[ith] = np.sqrt(mean_squared_error(test.y, y_pred))

                else:

                    cv_r2_scores[ith] = r2_score(test.y.ravel(), y_pred)

                    cv_rmse_scores[ith] = np.sqrt(mean_squared_error(test.y.ravel(), y_pred))

            mean_cv = np.mean(cv_rmse_scores)

            r2_coefficent_of_determination = np.mean(cv_r2_scores)

            train_val_abs_diff = np.abs(np.mean(train_scores) - np.mean(cv_rmse_scores))

            return mean_cv, r2_coefficent_of_determination, train_val_abs_diff


### objective_random_cv_multi_rmse__r2__diff_train_test

```python3
def objective_random_cv_multi_rmse__r2__diff_train_test(
    trial: optuna.trial._trial.Trial,
    regressorfx: Callable,
    parameters,
    X: Union[pandas.core.frame.DataFrame, numpy.ndarray],
    y: Union[pandas.core.series.Series, numpy.ndarray],
    randomize_cv_split: bool = False,
    update_param_grid_callback: Optional[Callable] = None,
    k_fold: int = 5,
    **kwargs
) -> Tuple[Union[numpy.float64, float], Union[numpy.float64, float], float]
```

Function to do an internal random cv over X and y

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| trial | optuna.trial.Trial | _description_ | None |
| regressorfx | Callable | _description_ | None |
| parameters | _type_ | _description_ | None |
| X | Union[pd.DataFrame, np.ndarray] | _description_ | None |
| y | Union[pd.Series, np.ndarray] | _description_ | None |
| randomize_cv_split | bool | _description_. Defaults to False. | False |
| update_param_grid_callback | Optional[Callable] | _description_. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| Tuple[ Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float ] | _description_ |

??? example "View Source"
        def objective_random_cv_multi_rmse__r2__diff_train_test(

            trial: optuna.trial.Trial,

            regressorfx: Callable,

            parameters,

            X: Union[pd.DataFrame, np.ndarray],

            y: Union[pd.Series, np.ndarray],

            randomize_cv_split: bool = False,

            update_param_grid_callback: Optional[Callable] = None,

            k_fold: int = 5,

            **kwargs,

        ) -> Tuple[

            Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

        ]:

            """

            Function to do an internal random cv over X and y

            Args:

                trial (optuna.trial.Trial): _description_

                regressorfx (Callable): _description_

                parameters (_type_): _description_

                X (Union[pd.DataFrame, np.ndarray]): _description_

                y (Union[pd.Series, np.ndarray]): _description_

                randomize_cv_split (bool, optional): _description_. Defaults to False.

                update_param_grid_callback (Optional[Callable], optional): _description_. Defaults to None.

            Returns:

                Tuple[ Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float ]: _description_

            """

            param_grid = build_param_grid(parameters, trial)

            log.debug(param_grid)

            if update_param_grid_callback is not None:

                param_grid = update_param_grid_callback(trial, deepcopy(param_grid), **kwargs)

            if isinstance(X, np.ndarray):

                X = pd.DataFrame(X)

            if isinstance(y, np.ndarray):

                y = pd.Series(y)

            if randomize_cv_split is True:

                seed = utilities.random_seed * datetime.now().second

            else:

                seed = utilities.random_seed

            cv = KFold(n_splits=k_fold, shuffle=True, random_state=seed)

            cv_rmse_scores = np.empty(k_fold)

            cv_r2_scores = np.empty(k_fold)

            train_scores = np.empty(k_fold)

            for ith, (train_idx, test_idx) in enumerate(cv.split(X, y)):

                if isinstance(regressorfx, Pipeline):

                    regressor = regressorfx.set_params(**param_grid)

                else:

                    param_grid["random_state"] = utilities.random_seed

                    regressor = regressorfx(**param_grid)

                X_train = X.iloc[train_idx]

                X_test = X.iloc[test_idx]

                y_train = y[train_idx]

                y_test = y[test_idx]

                regressor.fit(X_train, y_train)

                y_pred = regressor.predict(X_test)

                log.error(X_train)

                log.error(y_train)

                train_scores[ith] = np.sqrt(

                    mean_squared_error(y_train, regressor.predict(X_train))

                )

                cv_r2_scores[ith] = r2_score(y_test, y_pred)

                cv_rmse_scores[ith] = np.sqrt(mean_squared_error(y_test, y_pred))

            mean_cv = np.mean(cv_rmse_scores)

            r2_coefficent_of_determination = np.mean(cv_r2_scores)

            train_val_abs_diff = np.abs(np.mean(train_scores) - np.mean(cv_rmse_scores))

            return mean_cv, r2_coefficent_of_determination, train_val_abs_diff


### optimize_dnn_arch_and_train

```python3
def optimize_dnn_arch_and_train(
    trial: optuna.trial._trial.Trial,
    assay_targets: List[str],
    decreasing_neurons_only: bool = False,
    uncertainty: bool = True,
    epochs: int = 100,
    fit_transformers: Optional[List[transformers.Transformer]] = None,
    train_set: Optional[deepchem.data.data_loader.DataLoader] = None,
    valid_set: Optional[deepchem.data.data_loader.DataLoader] = None,
    test_set: Optional[deepchem.data.data_loader.DataLoader] = None,
    df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    name_column: Optional[str] = None,
    parameters: Optional[dict] = None,
    verbose: bool = False
)
```

??? example "View Source"
        def optimize_dnn_arch_and_train(

            trial: optuna.trial.Trial,

            assay_targets: List[str],

            decreasing_neurons_only: bool = False,

            uncertainty: bool = True,

            epochs: int = 100,

            fit_transformers: Optional[List[dc.trans.Transformer]] = None,

            train_set: Optional[dc.data.data_loader.DataLoader] = None,

            valid_set: Optional[dc.data.data_loader.DataLoader] = None,

            test_set: Optional[dc.data.data_loader.DataLoader] = None,

            df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            name_column: Optional[str] = None,

            parameters: Optional[dict] = None,

            verbose: bool = False,

        ):

            """ """

            if parameters is None:

                parameters = {

                    "n_hidden_layers": ("int", 1, 4),

                    "n_neurons": ("int", 100, 1000),

                    "dropout_rate": ("float", 0.1, 1.0),

                    "activation": ("categorical", ["relu", "sigmoid", "tanh"]),

                    "learning_rate": ("float", 1e-6, 1e-2),

                    "batch_size": ("int", 4, 256),

                }

            # Ensures all the expected keys are present in the parameters dictionary

            else:

                default_parameters = {

                    "n_hidden_layers": ("int", 1, 4),

                    "n_neurons": ("int", 100, 1000),

                    "dropout_rate": ("float", 0.1, 1.0),

                    "activation": ("categorical", ["relu", "sigmoid", "tanh"]),

                    "learning_rate": ("float", 1e-6, 1e-2),

                    "batch_size": ("int", 4, 128),

                }

                parameters = {**default_parameters, **parameters}

            param_grid = {}

            param_grid["n_hidden_layers"] = trial.suggest_int(

                "n_hidden_layers",

                parameters["n_hidden_layers"][1],

                parameters["n_hidden_layers"][2],

            )

            param_grid["learning_rate"] = trial.suggest_float(

                "learning_rate", parameters["learning_rate"][1], parameters["learning_rate"][2]

            )

            param_grid["batch_size"] = trial.suggest_int(

                "batch_size", parameters["batch_size"][1], parameters["batch_size"][2]

            )

            layer_sizes = []

            activation_funcs = []

            drop_out = []

            for ith in range(param_grid["n_hidden_layers"]):

                if decreasing_neurons_only is True and ith > 0:

                    lower = max(int(layer_sizes / 10), parameters["n_neurons"][1])

                    upper = max(int(layer_sizes / 2), parameters["n_neurons"][1])

                else:

                    lower = parameters["n_neurons"][1]

                    upper = parameters["n_neurons"][2]

                layer_sizes.append(trial.suggest_int(f"n_neurons_{ith}", lower, upper))

                activation_funcs.append(

                    trial.suggest_categorical(f"activation_{ith}", parameters["activation"][1])

                )

                drop_out.append(

                    trial.suggest_float(

                        f"dropout_rate_{ith}",

                        parameters["dropout_rate"][1],

                        parameters["dropout_rate"][2],

                    )

                )

            param_grid["hidden_layer_sizes"] = layer_sizes

            param_grid["activation"] = activation_funcs

            param_grid["dropout_rate"] = drop_out

            if verbose is True:

                log.info(f"Parameter grid: {param_grid}")

            if train_set is None and test_set is None and df is not None:

                _, _, _, _, test_set_df = deep_net_models.train_multitask_regressor(

                    tasks=assay_targets,

                    data_df=df,

                    fit_transformers=fit_transformers,

                    smiles_column=smiles_column,

                    ids_column=name_column,

                    layer_sizes=param_grid["hidden_layer_sizes"],

                    epochs=epochs,

                    batch_size=param_grid["batch_size"],

                    learning_rate=param_grid["learning_rate"],

                    dropout_rate=param_grid["dropout_rate"],

                    activation_fns=param_grid["activation"],

                    uncertainty=uncertainty,

                )

                error = test_set_df.loc["RMS", "mean over tasks"]

                return error

            elif train_set is not None and test_set is not None and df is None:

                _, _, _, _, test_set_df = deep_net_models.train_multitask_regressor(

                    tasks=assay_targets,

                    train_dataset=train_set,

                    test_dataset=test_set,

                    valid_dataset=valid_set,

                    fit_transformers=fit_transformers,

                    layer_sizes=param_grid["hidden_layer_sizes"],

                    epochs=epochs,

                    batch_size=param_grid["batch_size"],

                    learning_rate=param_grid["learning_rate"],

                    dropout_rate=param_grid["dropout_rate"],

                    activation_fns=param_grid["activation"],

                    uncertainty=uncertainty,

                )

                error = test_set_df.loc["RMS", "mean over tasks"]

                return error

            else:

                log.critical("NO TRAINING IS HAPPENING")


### run_kfold_study_with_mlflow

```python3
def run_kfold_study_with_mlflow(
    models: List[redxregressors.classical_ml_models.skmodel],
    cls: tune.build_kfold_objective,
    priors_dict: Optional[dict],
    pipeline_priors: Optional[dict],
    holdout_set: Union[tune.JoinKfoldData, deepchem.data.datasets.NumpyDataset, deepchem.data.datasets.DiskDataset],
    num_trials: int,
    directions: List[str],
    mlflow_experiment_name: str,
    experiment_id: Optional[int] = None,
    pipeline_list: Optional[List[Tuple[str, Callable]]] = None,
    local_abs_store_path: Optional[str] = None,
    model_key_in_pipeline: Optional[str] = 'model',
    sampler: optuna.samplers._base.BaseSampler = <optuna.samplers._tpe.sampler.TPESampler object at 0x30c639e50>,
    training_smiles: Optional[List[str]] = None,
    experiment_description: Optional[str] = None
) -> tuple[list[typing.Any], int]
```

Function to run a kfold study with mlflow logging the results and models for each trial

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| models | List[classical_ml_models.skmodel] | The models to run the kfold study on | None |
| cls | build_kfold_objective | The class object to run the kfold study | None |
| pipe_or_model | Union[Pipeline, Callable] | The pipeline or model to optimize | None |
| priors_dict | Optional[dict] | The priors to optimize over | None |
| pipeline_priors | Optional[dict] | The priors to optimize over for the pipeline | None |
| holdout_set | Union[JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset] | The holdout set to validate the models | None |
| num_trials | int | The number of trials to run | None |
| experiment_id | int | The mlflow experiment id | None |
| directions | List[str] | The directions to optimize the objectives in | None |
| mlflow_experiment_name | str | The mlflow experiment name | None |
| local_abs_store_path | Optional[str] | The local absolute path to store the models. Defaults to None. | None |
| model_key_in_pipeline | Optional[str] | The key to prepend to the model parameters in the pipeline. Defaults to "model__". | "model__" |
| sampler | optuna.samplers.BaseSampler | The sampler to use for the optimization. Defaults to optuna.samplers.TPESampler(seed=utilities.random_seed). | optuna.samplers.TPESampler(seed=utilities.random_seed) |

**Returns:**

| Type | Description |
|---|---|
| tuple[list[Any], int] | The studies and the experiment id |

??? example "View Source"
        def run_kfold_study_with_mlflow(

            models: List[classical_ml_models.skmodel],

            cls: build_kfold_objective,

            priors_dict: Optional[dict],

            pipeline_priors: Optional[dict],

            holdout_set: Union[

                JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset

            ],

            num_trials: int,

            directions: List[str],

            mlflow_experiment_name: str,

            experiment_id: Optional[int] = None,

            pipeline_list: Optional[List[Tuple[str, Callable]]] = None,

            local_abs_store_path: Optional[str] = None,

            model_key_in_pipeline: Optional[str] = "model",

            sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(

                seed=utilities.random_seed,

            ),

            training_smiles: Optional[List[str]] = None,

            experiment_description: Optional[str] = None,

        ) -> tuple[list[Any], int]:

            """

            Function to run a kfold study with mlflow logging the results and models for each trial

            Args:

                models (List[classical_ml_models.skmodel]): The models to run the kfold study on

                cls (build_kfold_objective): The class object to run the kfold study

                pipe_or_model (Union[Pipeline, Callable]): The pipeline or model to optimize

                priors_dict (Optional[dict]): The priors to optimize over

                pipeline_priors (Optional[dict]): The priors to optimize over for the pipeline

                holdout_set (Union[JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset]): The holdout set to validate the models

                num_trials (int): The number of trials to run

                experiment_id (int): The mlflow experiment id

                directions (List[str]): The directions to optimize the objectives in

                mlflow_experiment_name (str): The mlflow experiment name

                local_abs_store_path (Optional[str], optional): The local absolute path to store the models. Defaults to None.

                model_key_in_pipeline (Optional[str], optional): The key to prepend to the model parameters in the pipeline. Defaults to "model__".

                sampler (optuna.samplers.BaseSampler, optional): The sampler to use for the optimization. Defaults to optuna.samplers.TPESampler(seed=utilities.random_seed).

            Returns:

                tuple[list[Any], int]: The studies and the experiment id

            """

            if local_abs_store_path is not None:

                store_path = Path(local_abs_store_path)

                if not store_path.exists():

                    store_path.mkdir(parents=True)

            experiment_id = ml_flow_funcs.setup_for_mlflow(

                mlflow_experiment_name, utilities.mlflow_local_uri

            )

            # if training_smiles is given automatically create an applicability domain based on Tanimoto distance/similarity model and log it to mlflow.

            # It is the same for all prediction models.

            if training_smiles is not None:

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name="Applicability_domain",

                    nested=True,

                ):

                    log.debug(f"Training smiles: {training_smiles}")

                    admodel = applicability_domain.get_tanimoto_ad_model(

                        training_smiles=training_smiles,

                        radius=2,

                        hash_length=1024,

                        algorithm="brute",

                    )

                    wadmodel = WrappedKNeighbors(admodel)

                    example_input = ml_featurization.get_ecfp(

                        smiles=["c1ccccc1", "CCCCC", "C(CCN)C(=O)O", "c1ccccc1NC(=O)C"],

                        radius=2,

                        hash_length=1024,

                        return_np=True,

                    )

                    mlflow.sklearn.log_model(

                        sk_model=wadmodel,

                        artifact_path="applicability_domain_model",

                        signature=mlflow.models.infer_signature(

                            example_input, wadmodel.predict(example_input)

                        ),

                        registered_model_name="applicability_domain_kneighbours_tanimoto_distance_model",

                    )

                    mlflow.log_metric("number_of_training_points", admodel.n_samples_fit_)

                    mlflow.log_metric("number_of_features", admodel.n_features_in_)

            studies = []

            for m in models:

                if not isinstance(m, classical_ml_models.skmodel):

                    log.error(f"Model {m} is not an instance of skmodel")

                    continue

                date = datetime.now().strftime("%Y-%m-%d")

                date_and_time = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M")

                if local_abs_store_path is not None:

                    model_store_path = store_path.joinpath(f"kfold_{m.name}_{date}")

                    if not store_path.exists():

                        model_store_path.mkdir(parents=True)

                if priors_dict is None:

                    priors = m.default_param_range_priors

                else:

                    priors = priors_dict

                if pipeline_list is not None:

                    pipe_or_model = Pipeline(pipeline_list + [(model_key_in_pipeline, m.model)])

                else:

                    pipe_or_model = m.model

                log.info(f"Model: {pipe_or_model}")

                if isinstance(pipe_or_model, Pipeline):

                    priors = utilities.prepend_dictionary_keys(

                        priors, prepend=f"{model_key_in_pipeline}__"

                    )

                if pipeline_priors is not None:

                    priors = {**priors, **pipeline_priors}

                log.info(f"Priors: {priors}")

                name = f"kfold_{cls.k}_{m.name}_{str(sampler)}"

                if local_abs_store_path is not None:

                    study = optuna.create_study(

                        study_name=f"library_prod_{name}",

                        sampler=sampler,

                        storage=f"sqlite:///{model_store_path.joinpath(f'library_prod_{name}_{date_and_time}.db')}",

                        load_if_exists=True,

                        directions=directions,

                    )

                else:

                    study = optuna.create_study(

                        study_name=f"library_prod_{name}",

                        sampler=sampler,

                        storage=None,

                        load_if_exists=False,

                        directions=directions,

                    )

                log.info(

                    f"For model {m.name} we will run {num_trials} trials to optimize the hyperparameters"

                )

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"{m.name}_training_runs",

                    nested=True,

                ):

                    study.optimize(

                        lambda trial: cls.objectivefx(

                            trial,

                            pipe_or_model,

                            priors,

                            name=f"kfold_{m.name}",

                            experiment_id=experiment_id,

                            experiment_description=experiment_description,

                        ),

                        n_trials=num_trials,

                        show_progress_bar=True,

                        gc_after_trial=False,

                    )

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"{m.name}_validation_runs",

                    nested=True,

                ):

                    # Account for the absolute difference between the training and test set scores if its an objective

                    n_standard_ojectives = len(cls.objectives)

                    if cls.add_train_test_diff_objective is True:

                        n_standard_ojectives += 1

                        target_names = [ent.__name__ for ent in cls.objectives] + [

                            "abs_train_test_diff"

                        ]

                    else:

                        target_names = [ent.__name__ for ent in cls.objectives]

                    if n_standard_ojectives <= 3:

                        ax_po = optuna.visualization.plot_pareto_front(

                            study, target_names=target_names

                        )

                        log.info(type(ax_po))

                        ax_po.write_image(f"pareto_front_{m.name}.png")

                        mlflow.log_artifact(f"pareto_front_{m.name}.png")

                    all_data_as_single_train_fold_X, all_data_as_single_train_fold_y = (

                        cls.get_all_data_as_single_fold()

                    )

                    validation_ojective_values = []

                    validation_objective_indexes = []

                    for bs in study.best_trials:

                        log.info(f"Best trial: {bs.number} {bs.values}")

                        mod = pipe_or_model.set_params(**bs.params)

                        log.info(all_data_as_single_train_fold_y.shape)

                        mod.fit(

                            all_data_as_single_train_fold_X,

                            all_data_as_single_train_fold_y.ravel(),

                        )

                        mlflow.sklearn.log_model(

                            sk_model=mod,

                            artifact_path=f"model_{bs.number}",

                            signature=mlflow.models.infer_signature(

                                all_data_as_single_train_fold_X,

                                mod.predict(all_data_as_single_train_fold_X),

                            ),

                            input_example=all_data_as_single_train_fold_X,

                            registered_model_name=f"kfold_{m.name}_model_{bs.number}",

                        )

                        validation_set_pred = mod.predict(holdout_set.X)

                        validation_ojective_values.append(

                            [obj(holdout_set.y, validation_set_pred) for obj in cls.objectives]

                        )

                        validation_objective_indexes.append(f"{m.name}_{bs.number}")

                        log.info(

                            f"Validation set objective values for {validation_objective_indexes[-1]}: {validation_ojective_values[-1]}"

                        )

                        # plot the parity plot of the predictions

                        xymin = np.floor(

                            min(holdout_set.y.ravel().tolist() + validation_set_pred.tolist())

                        )

                        xymax = np.ceil(

                            max(holdout_set.y.ravel().tolist() + validation_set_pred.tolist())

                        )

                        log.info(f"min {xymin}, max {xymax}")

                        size = (10, 10)

                        title_fontsize = 27

                        parity_plt = build_kfold_objective.parity_plot(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            xymin,

                            xymax,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            parity_plt,

                            f"external_validation_set_parity_plot_fold_{bs.number}.png",

                        )

                        residual_plt = build_kfold_objective.plot_residuals(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            residual_plt,

                            f"external_validation_set_residual_plot_fold_{bs.number}.png",

                        )

                        err_plt = build_kfold_objective.plot_prediction_error(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            err_plt, f"external_validation_set_error_plot_fold_{bs.number}.png"

                        )

                        qq_plt = build_kfold_objective.plot_qq(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            qq_plt, f"external_validation_set_qq_plot_fold_{bs.number}.png"

                        )

                    validation_metric_tab_df = pd.DataFrame(

                        validation_ojective_values,

                        columns=[ent.__name__ for ent in cls.objectives],

                        index=validation_objective_indexes,

                    )

                    validation_metric_tab_df.to_csv(

                        f"{m.name}_validation_metrics.csv", index_label="model"

                    )

                    validation_metric_tab_df.to_html(

                        f"{m.name}_validation_metrics.html", index_names=True

                    )

                    mlflow.log_artifact(f"{m.name}_validation_metrics.csv")

                    mlflow.log_artifact(f"{m.name}_validation_metrics.html")

                    studies.append(study)

            return studies, experiment_id


### run_train_test_study_with_mlflow

```python3
def run_train_test_study_with_mlflow(
    models: List[redxregressors.classical_ml_models.skmodel],
    cls: tune.build_train_test_objective,
    priors_dict: Optional[dict],
    holdout_set: Union[tune.JoinKfoldData, deepchem.data.datasets.NumpyDataset, deepchem.data.datasets.DiskDataset],
    num_trials: int,
    directions: List[str],
    mlflow_experiment_name: str,
    experiment_id: Optional[int] = None,
    pipeline_list: Optional[List[Tuple[str, Callable]]] = None,
    local_abs_store_path: Optional[str] = None,
    model_key_in_pipeline: Optional[str] = 'model',
    sampler: optuna.samplers._base.BaseSampler = <optuna.samplers._tpe.sampler.TPESampler object at 0x30c6398d0>,
    training_smiles: Optional[List[str]] = None,
    experiment_description: Optional[str] = None
) -> tuple[list[typing.Any], int]
```

Function to run a kfold study with mlflow logging the results and models for each trial

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| models | List[classical_ml_models.skmodel] | The models to run the kfold study on | None |
| cls | build_train_test_objective | The class object to run the train test study | None |
| pipe_or_model | Union[Pipeline, Callable] | The pipeline or model to optimize | None |
| priors_dict | Optional[dict] | The priors to optimize over | None |
| holdout_set | Union[JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset] | The holdout set to validate the models | None |
| num_trials | int | The number of trials to run | None |
| experiment_id | int | The mlflow experiment id | None |
| directions | List[str] | The directions to optimize the objectives in | None |
| mlflow_experiment_name | str | The mlflow experiment name | None |
| local_abs_store_path | Optional[str] | The local absolute path to store the models. Defaults to None. | None |
| model_key_in_pipeline | Optional[str] | The key to prepend to the model parameters in the pipeline. Defaults to "model__". | "model__" |
| sampler | optuna.samplers.BaseSampler | The sampler to use for the optimization. Defaults to optuna.samplers.TPESampler(seed=utilities.random_seed). | optuna.samplers.TPESampler(seed=utilities.random_seed) |

**Returns:**

| Type | Description |
|---|---|
| tuple[list[Any], int] | The studies and the experiment id |

??? example "View Source"
        def run_train_test_study_with_mlflow(

            models: List[classical_ml_models.skmodel],

            cls: build_train_test_objective,

            priors_dict: Optional[dict],

            holdout_set: Union[

                JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset

            ],

            num_trials: int,

            directions: List[str],

            mlflow_experiment_name: str,

            experiment_id: Optional[int] = None,

            pipeline_list: Optional[List[Tuple[str, Callable]]] = None,

            local_abs_store_path: Optional[str] = None,

            model_key_in_pipeline: Optional[str] = "model",

            sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(

                seed=utilities.random_seed,

            ),

            training_smiles: Optional[List[str]] = None,

            experiment_description: Optional[str] = None,

        ) -> tuple[list[Any], int]:

            """

            Function to run a kfold study with mlflow logging the results and models for each trial

            Args:

                models (List[classical_ml_models.skmodel]): The models to run the kfold study on

                cls (build_train_test_objective): The class object to run the train test study

                pipe_or_model (Union[Pipeline, Callable]): The pipeline or model to optimize

                priors_dict (Optional[dict]): The priors to optimize over

                holdout_set (Union[JoinKfoldData, deepchem.data.NumpyDataset, deepchem.data.DiskDataset]): The holdout set to validate the models

                num_trials (int): The number of trials to run

                experiment_id (int): The mlflow experiment id

                directions (List[str]): The directions to optimize the objectives in

                mlflow_experiment_name (str): The mlflow experiment name

                local_abs_store_path (Optional[str], optional): The local absolute path to store the models. Defaults to None.

                model_key_in_pipeline (Optional[str], optional): The key to prepend to the model parameters in the pipeline. Defaults to "model__".

                sampler (optuna.samplers.BaseSampler, optional): The sampler to use for the optimization. Defaults to optuna.samplers.TPESampler(seed=utilities.random_seed).

            Returns:

                tuple[list[Any], int]: The studies and the experiment id

            """

            if local_abs_store_path is not None:

                store_path = Path(local_abs_store_path)

                if not store_path.exists():

                    store_path.mkdir(parents=True)

            experiment_id = ml_flow_funcs.setup_for_mlflow(

                mlflow_experiment_name, utilities.mlflow_local_uri

            )

            # if training_smiles is given automatically create an applicability domain based on Tanimoto distance/similarity model and log it to mlflow.

            # It is the same for all prediction models.

            if training_smiles is not None:

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name="Applicability_domain",

                    nested=True,

                ):

                    log.debug(f"Training smiles: {training_smiles}")

                    admodel = applicability_domain.get_tanimoto_ad_model(

                        training_smiles=training_smiles,

                        radius=2,

                        hash_length=1024,

                        algorithm="brute",

                    )

                    wadmodel = WrappedKNeighbors(admodel)

                    example_input = ml_featurization.get_ecfp(

                        smiles=["c1ccccc1", "CCCCC", "C(CCN)C(=O)O", "c1ccccc1NC(=O)C"],

                        radius=2,

                        hash_length=1024,

                        return_np=True,

                    )

                    mlflow.sklearn.log_model(

                        sk_model=wadmodel,

                        artifact_path="applicability_domain_model",

                        signature=mlflow.models.infer_signature(

                            example_input, wadmodel.predict(example_input)

                        ),

                        registered_model_name="applicability_domain_kneighbours_tanimoto_distance_model",

                    )

                    mlflow.log_metric("number_of_training_points", admodel.n_samples_fit_)

                    mlflow.log_metric("number_of_features", admodel.n_features_in_)

            studies = []

            for m in models:

                if not isinstance(m, classical_ml_models.skmodel):

                    log.error(f"Model {m} is not an instance of skmodel")

                    continue

                date = datetime.now().strftime("%Y-%m-%d")

                date_and_time = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M")

                if local_abs_store_path is not None:

                    model_store_path = store_path.joinpath(f"kfold_{m.name}_{date}")

                    if not store_path.exists():

                        model_store_path.mkdir(parents=True)

                if priors_dict is None:

                    priors = m.default_param_range_priors

                else:

                    priors = priors_dict

                if pipeline_list is not None:

                    pipe_or_model = Pipeline(pipeline_list + [(model_key_in_pipeline, m.model)])

                else:

                    pipe_or_model = m.model

                log.info(f"Model: {pipe_or_model}")

                if isinstance(pipe_or_model, Pipeline):

                    priors = utilities.prepend_dictionary_keys(

                        priors, prepend=f"{model_key_in_pipeline}__"

                    )

                log.info(f"Priors: {priors}")

                name = f"train_test_{m.name}_{str(sampler)}"

                if local_abs_store_path is not None:

                    study = optuna.create_study(

                        study_name=f"library_prod_{name}",

                        sampler=sampler,

                        storage=f"sqlite:///{model_store_path.joinpath(f'library_prod_{name}_{date_and_time}.db')}",

                        load_if_exists=True,

                        directions=directions,

                    )

                else:

                    study = optuna.create_study(

                        study_name=f"library_prod_{name}",

                        sampler=sampler,

                        storage=None,

                        load_if_exists=False,

                        directions=directions,

                    )

                log.info(

                    f"For model {m.name} we will run {num_trials} trials to optimize the hyperparameters"

                )

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"{m.name}_training_runs",

                    nested=True,

                ):

                    study.optimize(

                        lambda trial: cls.objectivefx(

                            trial,

                            pipe_or_model,

                            priors,

                            name=f"train_and_testing_{m.name}",

                            experiment_id=experiment_id,

                            experiment_description=experiment_description,

                        ),

                        n_trials=num_trials,

                        show_progress_bar=True,

                        gc_after_trial=False,

                    )

                with mlflow.start_run(

                    experiment_id=experiment_id,

                    run_name=f"{m.name}_validation_runs",

                    nested=True,

                ):

                    # Account for the absolute difference between the training and test set scores if its an objective

                    n_standard_ojectives = len(cls.objectives)

                    if cls.add_train_test_diff_objective is True:

                        n_standard_ojectives += 1

                        target_names = [ent.__name__ for ent in cls.objectives] + [

                            "abs_train_test_diff"

                        ]

                    else:

                        target_names = [ent.__name__ for ent in cls.objectives]

                    if n_standard_ojectives <= 3:

                        ax_po = optuna.visualization.plot_pareto_front(

                            study, target_names=target_names

                        )

                        log.info(type(ax_po))

                        ax_po.write_image(f"pareto_front_{m.name}.png")

                        mlflow.log_artifact(f"pareto_front_{m.name}.png")

                    all_data_as_single_train_set_x, all_data_as_single_train_set_y = (

                        cls.get_all_data_as_single_set()

                    )

                    validation_ojective_values = []

                    validation_objective_indexes = []

                    for bs in study.best_trials:

                        log.info(f"Best trial: {bs.number} {bs.values}")

                        mod = pipe_or_model.set_params(**bs.params)

                        log.info(all_data_as_single_train_set_y.shape)

                        mod.fit(

                            all_data_as_single_train_set_x,

                            all_data_as_single_train_set_y.ravel(),

                        )

                        mlflow.sklearn.log_model(

                            sk_model=mod,

                            artifact_path=f"model_{bs.number}",

                            signature=mlflow.models.infer_signature(

                                all_data_as_single_train_set_x,

                                mod.predict(all_data_as_single_train_set_x),

                            ),

                            input_example=all_data_as_single_train_set_x,

                            registered_model_name=f"train_test_{m.name}_model_{bs.number}",

                        )

                        validation_set_pred = mod.predict(holdout_set.X)

                        validation_ojective_values.append(

                            [obj(holdout_set.y, validation_set_pred) for obj in cls.objectives]

                        )

                        validation_objective_indexes.append(f"{m.name}_{bs.number}")

                        log.info(

                            f"Validation set objective values for {validation_objective_indexes[-1]}: {validation_ojective_values[-1]}"

                        )

                        # plot the parity plot of the predictions

                        xymin = np.floor(

                            min(holdout_set.y.ravel().tolist() + validation_set_pred.tolist())

                        )

                        xymax = np.ceil(

                            max(holdout_set.y.ravel().tolist() + validation_set_pred.tolist())

                        )

                        log.info(f"min {xymin}, max {xymax}")

                        size = (10, 10)

                        title_fontsize = 27

                        parity_plt = build_kfold_objective.parity_plot(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            xymin,

                            xymax,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            parity_plt,

                            "external_validation_set_parity_plot_train_test_validation.png",

                        )

                        residual_plt = build_kfold_objective.plot_residuals(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            residual_plt,

                            "external_validation_set_residual_plot_train_test_validation.png",

                        )

                        err_plt = build_kfold_objective.plot_prediction_error(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            err_plt,

                            "external_validation_set_error_plot_train_test_validation.png",

                        )

                        qq_plt = build_kfold_objective.plot_qq(

                            holdout_set.y.ravel(),

                            validation_set_pred,

                            size=size,

                            title_fontsize=title_fontsize,

                        )

                        mlflow.log_figure(

                            qq_plt, "external_validation_set_qq_plot_train_test_validation.png"

                        )

                    validation_metric_tab_df = pd.DataFrame(

                        validation_ojective_values,

                        columns=[ent.__name__ for ent in cls.objectives],

                        index=validation_objective_indexes,

                    )

                    validation_metric_tab_df.to_csv(

                        f"{m.name}_validation_metrics.csv", index_label="model"

                    )

                    validation_metric_tab_df.to_html(

                        f"{m.name}_validation_metrics.html", index_names=True

                    )

                    mlflow.log_artifact(f"{m.name}_validation_metrics.csv")

                    mlflow.log_artifact(f"{m.name}_validation_metrics.html")

                    studies.append(study)

            return studies, experiment_id

## Classes

### JoinKfoldData

```python3
class JoinKfoldData(
    X: numpy.ndarray,
    y: numpy.ndarray
)
```

JoinKfoldData(X: numpy.ndarray, y: numpy.ndarray)

### WrappedKNeighbors

```python3
class WrappedKNeighbors(
    model
)
```

#### Methods


#### predict

```python3
def predict(
    self,
    X: numpy.ndarray,
    n_neighbors: int = 1,
    **kwargs
) -> numpy.ndarray
```

??? example "View Source"
            def predict(self, X: np.ndarray, n_neighbors: int = 1, **kwargs) -> np.ndarray:

                distances, _ = self.model.kneighbors(X, n_neighbors=n_neighbors, **kwargs)

                return distances

### build_kfold_objective

```python3
class build_kfold_objective(
    name: str,
    k: int = 5,
    **kwargs
)
```

#### Static methods


#### parity_plot

```python3
def parity_plot(
    y_test,
    y_pred,
    xymin,
    xymax,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    fname: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the parity plot of the test set predictions

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_test | JoinKfoldData | The test set data | None |
| y_pred | np.ndarray | The predicted values | None |
| xymin | float | The minimum value for the x and y axis | None |
| xymax | float | The maximum value for the x and y axis | None |
| size | Tuple[int, int] | The size of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| fname | Optional[str] | The file name and path to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The parity plot |

??? example "View Source"
            @staticmethod

            def parity_plot(

                y_test,

                y_pred,

                xymin,

                xymax,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the parity plot of the test set predictions

                Args:

                    y_test (JoinKfoldData): The test set data

                    y_pred (np.ndarray): The predicted values

                    xymin (float): The minimum value for the x and y axis

                    xymax (float): The maximum value for the x and y axis

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The parity plot

                """

                # Least squares regression line

                m, c = np.polyfit(y_test, y_pred, deg=1)

                xseq = np.linspace(xymin, xymax, num=100)

                # plot the parity plot figure

                with plt.style.context(style=style):

                    fig = plt.figure(figsize=size)

                    ticks = np.arange(xymin, xymax + 1, 1.0)

                    plt.scatter(

                        y_test.ravel(),

                        y_pred.ravel(),

                        label="Prefect Prediction",

                        c="#89a0b0",

                        alpha=0.25,

                    )

                    plt.plot([xymin, xymax], [xymin, xymax], "k--", label="x = y")

                    plt.plot(

                        xseq, m * xseq + c, "m-.", lw=1.5, label="Least Squares Regression Line"

                    )  # y = mx + c

                    plt.scatter(

                        y_test,

                        y_pred.ravel(),

                        label=f"Model predictions RMSE: {root_mean_squared_error(y_test, y_pred):.2f} R2 Coefficent of determination {r2_score(y_test, y_pred):.2f}",

                    )

                    plt.grid()

                    plt.legend()

                    plt.xlabel("Experimental", fontsize=max(title_fontsize - 2, 10))

                    plt.ylabel("Prediction", fontsize=max(title_fontsize - 2, 10))

                    plt.title("Test Set Experimental Vs. Prediction", fontsize=title_fontsize)

                    ax = plt.gca()

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 2, 7)

                    )

                    ax.set_yticks(ticks)

                    ax.set_xticks(ticks)

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig


#### plot_prediction_error

```python3
def plot_prediction_error(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    fname: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the prediction error plot of the test set predictions

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_test | np.ndarray | The test set target values | None |
| y_pred | np.ndarray | The predicted values | None |
| style | str | The style of the plot | None |
| size | Tuple[int, int] | The size of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| fname | Optional[str] | The file name and path to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The prediction error plot |

??? example "View Source"
            @staticmethod

            def plot_prediction_error(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the prediction error plot of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    style (str): The style of the plot

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The prediction error plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    ax.scatter(y_pred, y_test - y_pred)

                    ax.axhline(y=0, color="orange", linestyle="-.")

                    ax.set_title("Prediction Error Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Predictions", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Errors", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig


#### plot_qq

```python3
def plot_qq(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    fname: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the QQ plot of the residuals of the test set predictions

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_test | np.ndarray | The test set target values | None |
| y_pred | np.ndarray | The predicted values | None |
| style | str | The style of the plot | None |
| size | Tuple[int, int] | The size of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| fname | Optional[str] | The file name and path to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The QQ plot |

??? example "View Source"
            @staticmethod

            def plot_qq(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the QQ plot of the residuals of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    style (str): The style of the plot

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The QQ plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    scipy.stats.probplot(y_test - y_pred, dist="norm", plot=ax)

                    ax.set_title("QQ Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Theoretical Quantiles", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Ordered Values", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig


#### plot_residuals

```python3
def plot_residuals(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    fname: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the residuals of the test set predictions

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_test | np.ndarray | The test set target values | None |
| y_pred | np.ndarray | The predicted values | None |
| size | Tuple[int, int] | The size of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| fname | Optional[str] | The file name and path to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.figure | The residuals plot |

??? example "View Source"
            @staticmethod

            def plot_residuals(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the residuals of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.figure: The residuals plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    sns.residplot(

                        x=y_pred,

                        y=y_test - y_pred,

                        lowess=False,

                        ax=ax,

                        line_kws={"color": "orange", "lw": 1.5},

                    )

                    ax.axhline(y=0, color="black")

                    ax.set_title("Residual Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Prediction", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Residuals", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

#### Methods


#### abs_train_test_diff_objective

```python3
def abs_train_test_diff_objective(
    self,
    train_scores: numpy.ndarray,
    test_scores: numpy.ndarray
) -> float
```

Function to calculate the difference between the training and test set scores. This is used as a restraining metric to prevent overfitting.

Note this is the difference between the mean scores for all other objectives on all folds of the k fold.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| train_scores | np.ndarray | The training set scores | None |
| test_scores | np.ndarray | The test set scores | None |

**Returns:**

| Type | Description |
|---|---|
| float | The absolute difference between the training and test set scores |

??? example "View Source"
            def abs_train_test_diff_objective(

                self, train_scores: np.ndarray, test_scores: np.ndarray

            ) -> float:

                """

                Function to calculate the difference between the training and test set scores. This is used as a restraining metric to prevent overfitting.

                Note this is the difference between the mean scores for all other objectives on all folds of the k fold.

                Args:

                    train_scores (np.ndarray): The training set scores

                    test_scores (np.ndarray): The test set scores

                Returns:

                    float: The absolute difference between the training and test set scores

                """

                return np.abs(np.mean(train_scores) - np.mean(test_scores))


#### build_param_grid

```python3
def build_param_grid(
    self,
    parameters,
    trial: optuna.trial._trial.Trial,
    param_grid: Optional[dict] = None
) -> dict[typing.Any, typing.Any]
```

Function to build the parameter grid for the optimization

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| parameters | dict | The parameters to optimize over | None |
| trial | optuna.trial.Trial | The optuna trial object | None |
| param_grid | Optional[dict] | The parameter grid to update. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| dict[Any, Any] | The parameter grid |

??? example "View Source"
            def build_param_grid(

                self, parameters, trial: optuna.trial.Trial, param_grid: Optional[dict] = None

            ) -> dict[Any, Any]:

                """

                Function to build the parameter grid for the optimization

                Args:

                    parameters (dict): The parameters to optimize over

                    trial (optuna.trial.Trial): The optuna trial object

                    param_grid (Optional[dict], optional): The parameter grid to update. Defaults to None.

                Returns:

                    dict[Any, Any]: The parameter grid

                """

                if param_grid is None:

                    param_grid = {}

                for k, v in parameters.items():

                    if v[0].lower().strip() == "float":

                        param_grid[k] = trial.suggest_float(k, v[1], v[2])

                    elif v[0].lower().strip() == "int":

                        param_grid[k] = trial.suggest_int(k, v[1], v[2])

                    if v[0].lower().strip() == "catagorical":

                        param_grid[k] = trial.suggest_categorical(k, v[1])

                return param_grid


#### get_all_data_as_single_fold

```python3
def get_all_data_as_single_fold(
    self,
    kf: Optional[List[Tuple[Union[numpy.ndarray, deepchem.data.datasets.DiskDataset], Union[numpy.ndarray, deepchem.data.datasets.DiskDataset]]]] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Function to get all the data as a single fold. It combines each test set from the k fold into a single set. This is useful for training a model on all the data.

The returned data is a tuple of the features and target values as numpy.

**Returns:**

| Type | Description |
|---|---|
| Tuple[np.ndarray, np.ndarray] | The features and target values |

??? example "View Source"
            def get_all_data_as_single_fold(

                self,

                kf: Optional[

                    List[

                        Tuple[

                            Union[np.ndarray, data.datasets.DiskDataset],

                            Union[np.ndarray, data.datasets.DiskDataset],

                        ]

                    ]

                ] = None,

            ) -> Tuple[np.ndarray, np.ndarray]:

                """

                Function to get all the data as a single fold. It combines each test set from the k fold into a single set. This is useful for training a model on all the data.

                The returned data is a tuple of the features and target values as numpy.

                Returns:

                    Tuple[np.ndarray, np.ndarray]: The features and target values

                """

                if kf is None:

                    X = np.concatenate([test.X for _, test in self.cv_data])

                    y = np.concatenate([test.y for _, test in self.cv_data])

                else:

                    X = np.concatenate([test.X for _, test in kf])

                    y = np.concatenate([test.y for _, test in kf])

                return X, y


#### get_data

```python3
def get_data(
    self,
    X: Union[pandas.core.frame.DataFrame, numpy.ndarray, NoneType] = None,
    y: Union[numpy.ndarray, pandas.core.series.Series, NoneType] = None,
    kf: Optional[List[Tuple[Union[numpy.ndarray, deepchem.data.datasets.DiskDataset], Union[numpy.ndarray, deepchem.data.datasets.DiskDataset]]]] = None
) -> None
```

Data should be passed in as either lists or tuples containing numpy arrays or pandas dataframes/Series objects to X and y

which will be randomly split into k folds or a predefined kfold object can be passed in as kf. The predefined kf should be
either a deepchem kfold object or list of tuples of JoinKfoldData classes with X and y being numpy arrays. The data is then
stored in the cv_data and cv_ids attributes of the class object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| X | Optional[Union[np.ndarray, pd.DataFrame]] | The features | None |
| y | Optional[Union[np.ndarray, pd.Series]] | The target values | None |
| kf | Optional[List[Tuple[Union[np.ndarray, data.datasets.DiskDataset], Union[np.ndarray, data.datasets.DiskDataset]]] | The predefined kfold object | None |

??? example "View Source"
            def get_data(

                self,

                X: Optional[Union[np.ndarray, pd.DataFrame]] = None,

                y: Optional[Union[np.ndarray, pd.Series]] = None,

                kf: Optional[

                    List[

                        Tuple[

                            Union[np.ndarray, data.datasets.DiskDataset],

                            Union[np.ndarray, data.datasets.DiskDataset],

                        ]

                    ]

                ] = None,

            ) -> None:

                """

                Data should be passed in as either lists or tuples containing numpy arrays or pandas dataframes/Series objects to X and y

                which will be randomly split into k folds or a predefined kfold object can be passed in as kf. The predefined kf should be

                either a deepchem kfold object or list of tuples of JoinKfoldData classes with X and y being numpy arrays. The data is then

                stored in the cv_data and cv_ids attributes of the class object.

                Args:

                    X (Optional[Union[np.ndarray, pd.DataFrame]]): The features

                    y (Optional[Union[np.ndarray, pd.Series]]): The target values

                    kf (Optional[List[Tuple[Union[np.ndarray, data.datasets.DiskDataset], Union[np.ndarray, data.datasets.DiskDataset]]]): The predefined kfold object

                """

                if X is not None and y is not None:

                    if isinstance(X, pd.DataFrame):

                        X = X.values

                    if isinstance(y, pd.Series):

                        y = y.values

                    cv = KFold(n_splits=self.k, **self.kwargs)

                    for train_indx, test_indx in cv.split(X, y):

                        self.cv_data.append(

                            (

                                JoinKfoldData(X=X[train_indx], y=y[train_indx]),

                                JoinKfoldData(X=X[test_indx], y=y[test_indx]),

                            )

                        )

                        self.cv_ids.append((train_indx, test_indx))

                elif kf is not None:

                    try:

                        if "deepchem" in str(type(kf[0][0])):

                            for trainf, testf in kf:

                                self.cv_data.append(

                                    (

                                        JoinKfoldData(X=trainf.X, y=trainf.y),

                                        JoinKfoldData(X=testf.X, y=testf.y),

                                    )

                                )

                                self.cv_ids.append((trainf.ids, testf.ids))

                        else:

                            if isinstance(kf[0][0], JoinKfoldData):

                                self.cv_data = kf

                                for ith, (trainf, testf) in enumerate(kf):

                                    self.cv_ids.append(

                                        (

                                            [

                                                f"fold_{ith}_train_row_{jth}"

                                                for jth in range(len(trainf.X))

                                            ],

                                            [

                                                f"fold_{ith}_testrow_{jth}"

                                                for jth in range(len(testf.X))

                                            ],

                                        )

                                    )

                            else:

                                log.warning(

                                    "The kfolds do not have a deepchem or JoinKfoldData object format will try to format"

                                )

                                for ith, (trainf, testf) in enumerate(kf):

                                    self.cv_data.append(

                                        (

                                            JoinKfoldData(X=trainf[0], y=trainf[1]),

                                            JoinKfoldData(X=testf[0], y=testf[1]),

                                        )

                                    )

                                    self.cv_ids.append(

                                        (

                                            [

                                                f"fold_{ith}_train_row_{jth}"

                                                for jth in range(len(trainf[0]))

                                            ],

                                            [

                                                f"fold_{ith}_testrow_{jth}"

                                                for jth in range(len(testf[0]))

                                            ],

                                        )

                                    )

                    except IndexError:

                        log.error(

                            "The kfolds object is not in the correct format, it should be either a deepchem kfold data object or a list of tuples of numpy arrays"

                        )


#### objectivefx

```python3
def objectivefx(
    self,
    trial: optuna.trial._trial.Trial,
    regressorfx: Union[sklearn.pipeline.Pipeline, Callable],
    parameters: dict,
    update_param_grid_callback: Optional[Callable] = None,
    name: Optional[str] = 'kfold_study',
    experiment_id: Optional[int] = None,
    experiment_description: Optional[str] = None,
    without_mlflow: bool = False,
    **kwargs
) -> Tuple[Union[numpy.float64, float], Union[numpy.float64, float], float]
```

Function to do a k fold cross valuation over the data and return the mean scores for the objectives. This is built for use with optuna and the objective function should be a function that takes a trial object and returns the scores for the objectives.

It should be used like:

```python

cls = build_kfold_objective("test", 5)
cls.get_data(X, y)
directions = cls.set_objectives([root_mean_squared_error, r2_score], ["minimize", "maximize"])

experiment_id = ml_flow_funcs.setup_for_mlflow("kold_study_4", utilities.mlflow_local_uri)

pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))])

study = optuna.create_study(directions=directions, study_name="test", storage=f'sqlite:///test.db', load_if_exists=True)

func = lambda trial: cls.objectivefx(
    trial,
    pipe,
    priors,
    name="kfold_run",
    experiment_id=experiment_id
)

study.optimize(
    func,
    n_trials=40,
)
```

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| trial | optuna.trial.Trial | The optuna trial object | None |
| regressorfx | Any | The regressor function to optimize | None |
| parameters | dict | The parameter grid to optimize over | None |
| update_param_grid_callback | Optional[Callable] | A callback function to update the parameter grid. Defaults to None. | None |
| name | Optional[str] | The name of the study. Defaults to "kfold_study". | "kfold_study" |
| experiment_id | Optional[int] | The experiment id for mlflow. Defaults to None. | None |
| experiment_description | Optional[str] | The experiment description for mlflow. Defaults to None. | None |
| with_mlflow | bool | Whether to use mlflow or not | None |

**Returns:**

| Type | Description |
|---|---|
| Tuple[Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float] | The scores for the objectives |

??? example "View Source"
            def objectivefx(

                self,

                trial: optuna.trial.Trial,

                regressorfx: Union[Pipeline, Callable],

                parameters: dict,

                update_param_grid_callback: Optional[Callable] = None,

                name: Optional[str] = "kfold_study",

                experiment_id: Optional[int] = None,

                experiment_description: Optional[str] = None,

                without_mlflow: bool = False,

                **kwargs,

            ) -> Tuple[

                Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

            ]:

                """

                Function to do a k fold cross valuation over the data and return the mean scores for the objectives. This is built for use with optuna and the objective function should be a function that takes a trial object and returns the scores for the objectives.

                It should be used like:

                ```python

                cls = build_kfold_objective("test", 5)

                cls.get_data(X, y)

                directions = cls.set_objectives([root_mean_squared_error, r2_score], ["minimize", "maximize"])

                experiment_id = ml_flow_funcs.setup_for_mlflow("kold_study_4", utilities.mlflow_local_uri)

                pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))])

                study = optuna.create_study(directions=directions, study_name="test", storage=f'sqlite:///test.db', load_if_exists=True)

                func = lambda trial: cls.objectivefx(

                    trial,

                    pipe,

                    priors,

                    name="kfold_run",

                    experiment_id=experiment_id

                )

                study.optimize(

                    func,

                    n_trials=40,

                )

                ```

                Args:

                    trial (optuna.trial.Trial): The optuna trial object

                    regressorfx (Any): The regressor function to optimize

                    parameters (dict): The parameter grid to optimize over

                    update_param_grid_callback (Optional[Callable], optional): A callback function to update the parameter grid. Defaults to None.

                    name (Optional[str], optional): The name of the study. Defaults to "kfold_study".

                    experiment_id (Optional[int], optional): The experiment id for mlflow. Defaults to None.

                    experiment_description (Optional[str], optional): The experiment description for mlflow. Defaults to None.

                    with_mlflow (bool): Whether to use mlflow or not

                Returns:

                    Tuple[Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float]: The scores for the objectives

                """

                if without_mlflow is False:

                    return self._objectivefx_with_mlflow(

                        trial,

                        regressorfx,

                        parameters,

                        update_param_grid_callback,

                        name,

                        experiment_id,

                        experiment_description,

                        **kwargs,

                    )

                else:

                    return self._objectivefx_without_mlflow(

                        trial,

                        regressorfx,

                        parameters,

                        update_param_grid_callback,

                        name,

                        experiment_description,

                        **kwargs,

                    )


#### set_objectives

```python3
def set_objectives(
    self,
    objectives: Optional[List[Callable]] = None,
    directions: Optional[List[str]] = None,
    add_train_test_diff_objective: bool = False
) -> List[str]
```

Set the objectives and directions for the optimization. These should be metrics with an interface of objective = func(y_true, y_pred) and direction to be one of ["minimize", "maximize"].

If add_train_test_diff_objective is set to True the difference between the training and test set scores will be added as an objective to minimize. This provides a restraining metric to prevent overfitting.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| objectives | Optional[List[Callable]] | A list of objective functions to optimize | None |
| directions | Optional[List[str]] | A list of directions to optimize the objectives in | None |
| add_train_test_diff_objective | bool | Whether to add a restraining metric to prevent overfitting | None |

**Returns:**

| Type | Description |
|---|---|
| List[str] | A list of directions to optimize the objectives |

??? example "View Source"
            def set_objectives(

                self,

                objectives: Optional[List[Callable]] = None,

                directions: Optional[List[str]] = None,

                add_train_test_diff_objective: bool = False,

            ) -> List[str]:

                """

                Set the objectives and directions for the optimization. These should be metrics with an interface of objective = func(y_true, y_pred) and direction to be one of ["minimize", "maximize"].

                If add_train_test_diff_objective is set to True the difference between the training and test set scores will be added as an objective to minimize. This provides a restraining metric to prevent overfitting.

                Args:

                    objectives (Optional[List[Callable]]): A list of objective functions to optimize

                    directions (Optional[List[str]]): A list of directions to optimize the objectives in

                    add_train_test_diff_objective (bool): Whether to add a restraining metric to prevent overfitting

                Returns:

                    List[str]: A list of directions to optimize the objectives

                """

                if objectives is None:

                    self.objectives = [root_mean_squared_error]

                    self.directions = ["minimize"]

                else:

                    self.objectives = objectives

                    self.directions = directions

                # self.objective_values = [np.zeros(self.k) for _ in range(len(self.objectives))]

                self.objective_values = [np.zeros(len(self.objectives)) for _ in range(self.k)]

                self.add_train_test_diff_objective = add_train_test_diff_objective

                log.debug(self.objective_values)

                if add_train_test_diff_objective is True:

                    # log.debug("LOOK HERE: ADDED TRAIN TEST DIFF OBJECTIVE")

                    # log.debug(f"Before adding {self.objective_values}")

                    # self.objective_values = np.array(

                    #     [np.append(ent, [0.0]) for ent in self.objective_values]

                    # )

                    log.debug(f"After adding {self.objective_values}")

                    # self.objectives += [self.abs_train_test_diff_objective]

                    return np.array(self.directions + ["minimize"])

                else:

                    log.info("Not added train test diff objective as requested")

                    self.objective_values = np.array(self.objective_values)

                    return np.array(self.directions)

### build_train_test_objective

```python3
class build_train_test_objective(
    name: str,
    train_frac: Optional[float] = None,
    n_train: Optional[int] = None,
    **kwargs
)
```

#### Static methods


#### parity_plot

```python3
def parity_plot(
    y_test,
    y_pred,
    xymin,
    xymax,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    fname: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the parity plot of the test set predictions

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_test | JoinKfoldData | The test set data | None |
| y_pred | np.ndarray | The predicted values | None |
| xymin | float | The minimum value for the x and y axis | None |
| xymax | float | The maximum value for the x and y axis | None |
| size | Tuple[int, int] | The size of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| fname | Optional[str] | The file name and path to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The parity plot |

??? example "View Source"
            @staticmethod

            def parity_plot(

                y_test,

                y_pred,

                xymin,

                xymax,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the parity plot of the test set predictions

                Args:

                    y_test (JoinKfoldData): The test set data

                    y_pred (np.ndarray): The predicted values

                    xymin (float): The minimum value for the x and y axis

                    xymax (float): The maximum value for the x and y axis

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The parity plot

                """

                # Least squares regression line

                m, c = np.polyfit(y_test, y_pred, deg=1)

                xseq = np.linspace(xymin, xymax, num=100)

                # plot the parity plot figure

                with plt.style.context(style=style):

                    fig = plt.figure(figsize=size)

                    ticks = np.arange(xymin, xymax + 1, 1.0)

                    plt.scatter(

                        y_test.ravel(),

                        y_pred.ravel(),

                        label="Prefect Prediction",

                        c="#89a0b0",

                        alpha=0.25,

                    )

                    plt.plot([xymin, xymax], [xymin, xymax], "k--", label="x = y")

                    plt.plot(

                        xseq, m * xseq + c, "m-.", lw=1.5, label="Least Squares Regression Line"

                    )  # y = mx + c

                    plt.scatter(

                        y_test,

                        y_pred.ravel(),

                        label=f"Model predictions RMSE: {root_mean_squared_error(y_test, y_pred):.2f} R2 Coefficent of determination {r2_score(y_test, y_pred):.2f}",

                    )

                    plt.grid()

                    plt.legend()

                    plt.xlabel("Experimental", fontsize=max(title_fontsize - 2, 10))

                    plt.ylabel("Prediction", fontsize=max(title_fontsize - 2, 10))

                    plt.title("Test Set Experimental Vs. Prediction", fontsize=title_fontsize)

                    ax = plt.gca()

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 2, 7)

                    )

                    ax.set_yticks(ticks)

                    ax.set_xticks(ticks)

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig


#### plot_prediction_error

```python3
def plot_prediction_error(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    fname: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the prediction error plot of the test set predictions

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_test | np.ndarray | The test set target values | None |
| y_pred | np.ndarray | The predicted values | None |
| style | str | The style of the plot | None |
| size | Tuple[int, int] | The size of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| fname | Optional[str] | The file name and path to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The prediction error plot |

??? example "View Source"
            @staticmethod

            def plot_prediction_error(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the prediction error plot of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    style (str): The style of the plot

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The prediction error plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    ax.scatter(y_pred, y_test - y_pred)

                    ax.axhline(y=0, color="orange", linestyle="-.")

                    ax.set_title("Prediction Error Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Predictions", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Errors", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig


#### plot_qq

```python3
def plot_qq(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    fname: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the QQ plot of the residuals of the test set predictions

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_test | np.ndarray | The test set target values | None |
| y_pred | np.ndarray | The predicted values | None |
| style | str | The style of the plot | None |
| size | Tuple[int, int] | The size of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| fname | Optional[str] | The file name and path to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The QQ plot |

??? example "View Source"
            @staticmethod

            def plot_qq(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the QQ plot of the residuals of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    style (str): The style of the plot

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.Figure: The QQ plot

                """

                log.critical(f"LOOK: {fname}")

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    scipy.stats.probplot(y_test - y_pred, dist="norm", plot=ax)

                    ax.set_title("QQ Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Theoretical Quantiles", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Ordered Values", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                        return None

                plt.close(fig)

                return fig


#### plot_residuals

```python3
def plot_residuals(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    fname: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the residuals of the test set predictions

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_test | np.ndarray | The test set target values | None |
| y_pred | np.ndarray | The predicted values | None |
| size | Tuple[int, int] | The size of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| fname | Optional[str] | The file name and path to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.figure | The residuals plot |

??? example "View Source"
            @staticmethod

            def plot_residuals(

                y_test,

                y_pred,

                style="seaborn-v0_8-dark-palette",

                size=(10, 10),

                title_fontsize: int = 27,

                fname: Optional[str] = None,

            ) -> plt.Figure:

                """

                Function to plot the residuals of the test set predictions

                Args:

                    y_test (np.ndarray): The test set target values

                    y_pred (np.ndarray): The predicted values

                    size (Tuple[int, int]): The size of the plot

                    title_fontsize (int): The fontsize of the title

                    fname (Optional[str]): The file name and path to save the plot to

                Returns:

                    plt.figure: The residuals plot

                """

                with plt.style.context(style=style):

                    fig, ax = plt.subplots(figsize=size)

                    sns.residplot(

                        x=y_pred,

                        y=y_test - y_pred,

                        lowess=False,

                        ax=ax,

                        line_kws={"color": "orange", "lw": 1.5},

                    )

                    ax.axhline(y=0, color="black")

                    ax.set_title("Residual Plot", fontsize=title_fontsize)

                    ax.set_xlabel("Prediction", fontsize=max(title_fontsize - 2, 10))

                    ax.set_ylabel("Residuals", fontsize=max(title_fontsize - 2, 10))

                    ax.tick_params(

                        axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                    )

                    plt.tight_layout()

                    if fname is not None:

                        plt.savefig(fname)

                plt.close(fig)

                return fig

#### Methods


#### abs_train_test_diff_objective

```python3
def abs_train_test_diff_objective(
    self,
    train_scores: numpy.ndarray,
    test_scores: numpy.ndarray
) -> float
```

Function to calculate the difference between the training and test set scores. This is used as a restraining metric to prevent overfitting.

Note this is the difference between the mean scores for all other objectives on all folds of the k fold.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| train_scores | np.ndarray | The training set scores | None |
| test_scores | np.ndarray | The test set scores | None |

**Returns:**

| Type | Description |
|---|---|
| float | The absolute difference between the training and test set scores |

??? example "View Source"
            def abs_train_test_diff_objective(

                self, train_scores: np.ndarray, test_scores: np.ndarray

            ) -> float:

                """

                Function to calculate the difference between the training and test set scores. This is used as a restraining metric to prevent overfitting.

                Note this is the difference between the mean scores for all other objectives on all folds of the k fold.

                Args:

                    train_scores (np.ndarray): The training set scores

                    test_scores (np.ndarray): The test set scores

                Returns:

                    float: The absolute difference between the training and test set scores

                """

                return np.abs(np.mean(train_scores) - np.mean(test_scores))


#### build_param_grid

```python3
def build_param_grid(
    self,
    parameters,
    trial: optuna.trial._trial.Trial,
    param_grid: Optional[dict] = None
) -> dict[typing.Any, typing.Any]
```

Function to build the parameter grid for the optimization

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| parameters | dict | The parameters to optimize over | None |
| trial | optuna.trial.Trial | The optuna trial object | None |
| param_grid | Optional[dict] | The parameter grid to update. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| dict[Any, Any] | The parameter grid |

??? example "View Source"
            def build_param_grid(

                self, parameters, trial: optuna.trial.Trial, param_grid: Optional[dict] = None

            ) -> dict[Any, Any]:

                """

                Function to build the parameter grid for the optimization

                Args:

                    parameters (dict): The parameters to optimize over

                    trial (optuna.trial.Trial): The optuna trial object

                    param_grid (Optional[dict], optional): The parameter grid to update. Defaults to None.

                Returns:

                    dict[Any, Any]: The parameter grid

                """

                if param_grid is None:

                    param_grid = {}

                for k, v in parameters.items():

                    if v[0].lower().strip() == "float":

                        param_grid[k] = trial.suggest_float(k, v[1], v[2])

                    elif v[0].lower().strip() == "int":

                        param_grid[k] = trial.suggest_int(k, v[1], v[2])

                    if v[0].lower().strip() == "catagorical":

                        param_grid[k] = trial.suggest_categorical(k, v[1])

                return param_grid


#### get_all_data_as_single_set

```python3
def get_all_data_as_single_set(
    self,
    kf: Optional[List[Tuple[Union[numpy.ndarray, deepchem.data.datasets.DiskDataset], Union[numpy.ndarray, deepchem.data.datasets.DiskDataset]]]] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Function to get all the data as a single set. It combines te train and test set into a single set. This is useful for training a model on all the data.

The returned data is a tuple of the features and target values as numpy ndarrays.

**Returns:**

| Type | Description |
|---|---|
| Tuple[np.ndarray, np.ndarray] | The features and target values |

??? example "View Source"
            def get_all_data_as_single_set(

                self,

                kf: Optional[

                    List[

                        Tuple[

                            Union[np.ndarray, data.datasets.DiskDataset],

                            Union[np.ndarray, data.datasets.DiskDataset],

                        ]

                    ]

                ] = None,

            ) -> Tuple[np.ndarray, np.ndarray]:

                """

                Function to get all the data as a single set. It combines te train and test set into a single set. This is useful for training a model on all the data.

                The returned data is a tuple of the features and target values as numpy ndarrays.

                Returns:

                    Tuple[np.ndarray, np.ndarray]: The features and target values

                """

                X = np.concatenate([ent.X for ent in [self.train, self.test]])

                y = np.concatenate([ent.y for ent in [self.train, self.test]])

                return X, y


#### get_data

```python3
def get_data(
    self,
    X: Union[pandas.core.frame.DataFrame, numpy.ndarray, NoneType] = None,
    y: Union[numpy.ndarray, pandas.core.series.Series, NoneType] = None,
    train_test_predefined: Optional[Tuple[Union[numpy.ndarray, deepchem.data.datasets.DiskDataset], Union[numpy.ndarray, deepchem.data.datasets.DiskDataset]]] = None,
    train_frac: Optional[float] = None,
    n_train: Optional[int] = None
) -> None
```

Data should be passed in as either lists or tuples containing numpy arrays or pandas dataframes/Series objects to X and y

which will be randomly split into k folds or a predefined kfold object can be passed in as kf. The predefined kf should be
either a deepchem kfold object or list of tuples of JoinKfoldData classes with X and y being numpy arrays. The data is then
stored in the cv_data and cv_ids attributes of the class object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| X | Optional[Union[np.ndarray, pd.DataFrame]] | The features | None |
| y | Optional[Union[np.ndarray, pd.Series]] | The target values | None |
| kf | Optional[List[Tuple[Union[np.ndarray, data.datasets.DiskDataset], Union[np.ndarray, data.datasets.DiskDataset]]] | The predefined kfold object | None |

??? example "View Source"
            def get_data(

                self,

                X: Optional[Union[np.ndarray, pd.DataFrame]] = None,

                y: Optional[Union[np.ndarray, pd.Series]] = None,

                train_test_predefined: Optional[

                    Tuple[

                        Union[np.ndarray, data.datasets.DiskDataset],

                        Union[np.ndarray, data.datasets.DiskDataset],

                    ]

                ] = None,

                train_frac: Optional[float] = None,

                n_train: Optional[int] = None,

            ) -> None:

                """

                Data should be passed in as either lists or tuples containing numpy arrays or pandas dataframes/Series objects to X and y

                which will be randomly split into k folds or a predefined kfold object can be passed in as kf. The predefined kf should be

                either a deepchem kfold object or list of tuples of JoinKfoldData classes with X and y being numpy arrays. The data is then

                stored in the cv_data and cv_ids attributes of the class object.

                Args:

                    X (Optional[Union[np.ndarray, pd.DataFrame]]): The features

                    y (Optional[Union[np.ndarray, pd.Series]]): The target values

                    kf (Optional[List[Tuple[Union[np.ndarray, data.datasets.DiskDataset], Union[np.ndarray, data.datasets.DiskDataset]]]): The predefined kfold object

                """

                # deal with the case where the data is passed in as numpy arrays or pandas dataframes

                if X is not None and y is not None:

                    if isinstance(X, pd.DataFrame):

                        X = X.values

                    if isinstance(y, pd.Series):

                        y = y.values

                    # if the train_frac is not set then set it to the default value of 0.8

                    if self.train_frac is None and train_frac is None:

                        # if the n_train is not set then set it to the default value of None and use 0.8 fraction

                        if self.n_train is None and n_train is None:

                            log.warning(

                                "Both the class level train_frac variable and function level train_frac argument are set to None, the train_frac will be set to the default value of 0.8"

                            )

                            self.train_frac = 0.8

                        elif n_train is not None:

                            self.n_train = n_train

                    elif train_frac is not None:

                        self.train_frac = train_frac

                    # if the train_frac is set then split the data into a training and test set

                    if self.train_frac is not None:

                        X_train, X_test, y_train, y_test = train_test_split(

                            X,

                            y,

                            train_size=self.train_frac,

                            random_state=utilities.random_seed,

                            shuffle=True,

                        )

                        self.train = JoinKfoldData(X=X_train, y=y_train)

                        self.test = JoinKfoldData(X=X_test, y=y_test)

                    # if the n_train is set then split the data into a training and test set

                    else:

                        X_train, X_test, y_train, y_test = train_test_split(

                            X,

                            y,

                            train_size=self.n_train,

                            random_state=utilities.random_seed,

                            shuffle=True,

                        )

                        self.train = JoinKfoldData(X=X_train, y=y_train)

                        self.test = JoinKfoldData(X=X_test, y=y_test)

                # deal with the case where the data is passed in as a predefined train test split

                elif train_test_predefined is not None:

                    try:

                        # if the train_test_predefined is a deepchem object then set the train and test attributes

                        if "deepchem" in str(type(train_test_predefined[0])):

                            self.train = JoinKfoldData(

                                X=train_test_predefined[0].X, y=train_test_predefined[0].y

                            )

                            self.test = JoinKfoldData(

                                X=train_test_predefined[1].X, y=train_test_predefined[1].y

                            )

                            self.train_test_ids.append(

                                (train_test_predefined[0].ids, train_test_predefined[1].ids)

                            )

                        # if the train_test_predefined is a list of tuples of numpy arrays then set the train and test attributes

                        else:

                            # if the train_test_predefined is a JoinKfoldData object then set the train and test attributes

                            if isinstance(train_test_predefined[0], JoinKfoldData):

                                self.train_test_ids.append(

                                    (

                                        [

                                            f"train_row_{jth}"

                                            for jth in range(len(train_test_predefined[0].X))

                                        ],

                                        [

                                            f"test_row_{jth}"

                                            for jth in range(len(train_test_predefined[1].X))

                                        ],

                                    )

                                )

                            # if the train_test_predefined is a list of tuples of numpy arrays then set the train and test attributes

                            else:

                                log.warning(

                                    "The train_test_predefined do not have a deepchem or JoinKfoldData object format will try to format"

                                )

                                self.train = JoinKfoldData(

                                    X=train_test_predefined[0][0], y=train_test_predefined[0][1]

                                )

                                self.test = (

                                    JoinKfoldData(

                                        X=train_test_predefined[1][0],

                                        y=train_test_predefined[1][1],

                                    ),

                                )

                                self.train_test_ids.append(

                                    (

                                        [

                                            f"train_row_{jth}"

                                            for jth in range(len(train_test_predefined[0][0]))

                                        ],

                                        [

                                            f"test_row_{jth}"

                                            for jth in range(len(train_test_predefined[0][1]))

                                        ],

                                    )

                                )

                    # if the train_test_predefined is not in the correct format then raise an error

                    except IndexError:

                        log.error(

                            "The train_test_predefined object is not in the correct format, it should be either a deepchem data object or a list of tuples of numpy arrays"

                        )


#### objectivefx

```python3
def objectivefx(
    self,
    trial: optuna.trial._trial.Trial,
    regressorfx: Union[sklearn.pipeline.Pipeline, Callable],
    parameters: dict,
    update_param_grid_callback: Optional[Callable] = None,
    name: Optional[str] = 'kfold_study',
    experiment_id: Optional[int] = None,
    experiment_description: Optional[str] = None,
    without_mlflow: bool = False,
    **kwargs
) -> Tuple[Union[numpy.float64, float], Union[numpy.float64, float], float]
```

Function to do a k fold cross valuation over the data and return the mean scores for the objectives. This is built for use with optuna and the objective function should be a function that takes a trial object and returns the scores for the objectives.

It should be used like:

```python

cls = build_kfold_objective("test", 5)
cls.get_data(X, y)
directions = cls.set_objectives([root_mean_squared_error, r2_score], ["minimize", "maximize"])

experiment_id = ml_flow_funcs.setup_for_mlflow("kold_study_4", utilities.mlflow_local_uri)

pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))])

study = optuna.create_study(directions=directions, study_name="test", storage=f'sqlite:///test.db', load_if_exists=True)

func = lambda trial: cls.objectivefx(
    trial,
    pipe,
    priors,
    name="kfold_run",
    experiment_id=experiment_id
)

study.optimize(
    func,
    n_trials=40,
)
```

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| trial | optuna.trial.Trial | The optuna trial object | None |
| regressorfx | Any | The regressor function to optimize | None |
| parameters | dict | The parameter grid to optimize over | None |
| update_param_grid_callback | Optional[Callable] | A callback function to update the parameter grid. Defaults to None. | None |
| name | Optional[str] | The name of the study. Defaults to "kfold_study". | "kfold_study" |
| experiment_id | Optional[int] | The experiment id for mlflow. Defaults to None. | None |
| experiment_description | Optional[str] | The experiment description for mlflow. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| Tuple[Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float] | The scores for the objectives |

??? example "View Source"
            def objectivefx(

                self,

                trial: optuna.trial.Trial,

                regressorfx: Union[Pipeline, Callable],

                parameters: dict,

                update_param_grid_callback: Optional[Callable] = None,

                name: Optional[str] = "kfold_study",

                experiment_id: Optional[int] = None,

                experiment_description: Optional[str] = None,

                without_mlflow: bool = False,

                **kwargs,

            ) -> Tuple[

                Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float

            ]:

                """

                Function to do a k fold cross valuation over the data and return the mean scores for the objectives. This is built for use with optuna and the objective function should be a function that takes a trial object and returns the scores for the objectives.

                It should be used like:

                ```python

                cls = build_kfold_objective("test", 5)

                cls.get_data(X, y)

                directions = cls.set_objectives([root_mean_squared_error, r2_score], ["minimize", "maximize"])

                experiment_id = ml_flow_funcs.setup_for_mlflow("kold_study_4", utilities.mlflow_local_uri)

                pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor(random_state=50))])

                study = optuna.create_study(directions=directions, study_name="test", storage=f'sqlite:///test.db', load_if_exists=True)

                func = lambda trial: cls.objectivefx(

                    trial,

                    pipe,

                    priors,

                    name="kfold_run",

                    experiment_id=experiment_id

                )

                study.optimize(

                    func,

                    n_trials=40,

                )

                ```

                Args:

                    trial (optuna.trial.Trial): The optuna trial object

                    regressorfx (Any): The regressor function to optimize

                    parameters (dict): The parameter grid to optimize over

                    update_param_grid_callback (Optional[Callable], optional): A callback function to update the parameter grid. Defaults to None.

                    name (Optional[str], optional): The name of the study. Defaults to "kfold_study".

                    experiment_id (Optional[int], optional): The experiment id for mlflow. Defaults to None.

                    experiment_description (Optional[str], optional): The experiment description for mlflow. Defaults to None.

                Returns:

                    Tuple[Union[np.float64, float], Union[np.float64, Union[np.float64, float]], float]: The scores for the objectives

                """

                if without_mlflow is True:

                    log.critical("Training without MLFlow")

                    scores = self._objectivefx_without_mlflow(

                        trial,

                        regressorfx,

                        parameters,

                        update_param_grid_callback,

                        name,

                        **kwargs,

                    )

                else:

                    log.critical("Training with MLFlow")

                    scores = self._objectivefx_with_mlflow(

                        trial,

                        regressorfx,

                        parameters,

                        update_param_grid_callback,

                        name,

                        experiment_id,

                        experiment_description,

                        **kwargs,

                    )

                return scores


#### set_objectives

```python3
def set_objectives(
    self,
    objectives: Optional[List[Callable]] = None,
    directions: Optional[List[str]] = None,
    add_train_test_diff_objective: bool = False
) -> List[str]
```

Set the objectives and directions for the optimization. These should be metrics with an interface of objective = func(y_true, y_pred) and direction to be one of ["minimize", "maximize"].

If add_train_test_diff_objective is set to True the difference between the training and test set scores will be added as an objective to minimize. This provides a restraining metric to prevent overfitting.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| objectives | Optional[List[Callable]] | A list of objective functions to optimize | None |
| directions | Optional[List[str]] | A list of directions to optimize the objectives in | None |
| add_train_test_diff_objective | bool | Whether to add a restraining metric to prevent overfitting | None |

**Returns:**

| Type | Description |
|---|---|
| List[str] | A list of directions to optimize the objectives |

??? example "View Source"
            def set_objectives(

                self,

                objectives: Optional[List[Callable]] = None,

                directions: Optional[List[str]] = None,

                add_train_test_diff_objective: bool = False,

            ) -> List[str]:

                """

                Set the objectives and directions for the optimization. These should be metrics with an interface of objective = func(y_true, y_pred) and direction to be one of ["minimize", "maximize"].

                If add_train_test_diff_objective is set to True the difference between the training and test set scores will be added as an objective to minimize. This provides a restraining metric to prevent overfitting.

                Args:

                    objectives (Optional[List[Callable]]): A list of objective functions to optimize

                    directions (Optional[List[str]]): A list of directions to optimize the objectives in

                    add_train_test_diff_objective (bool): Whether to add a restraining metric to prevent overfitting

                Returns:

                    List[str]: A list of directions to optimize the objectives

                """

                if objectives is None:

                    self.objectives = [root_mean_squared_error]

                    self.directions = ["minimize"]

                else:

                    self.objectives = objectives

                    self.directions = directions

                self.objective_values = np.zeros(len(self.objectives))

                self.add_train_test_diff_objective = add_train_test_diff_objective

                log.debug(self.objective_values)

                if add_train_test_diff_objective is True:

                    log.debug(f"After adding {self.objective_values}")

                    return np.array(self.directions + ["minimize"])

                else:

                    self.objective_values = np.array(self.objective_values)

                    return np.array(self.directions)
