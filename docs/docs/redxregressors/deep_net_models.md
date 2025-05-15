# Module redxregressors.deep_net_models

Module of deep learning regressor DNNs

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module of deep learning regressor DNNs

        """

        import torch

        import logging

        import deepchem as dc

        import numpy as np

        import pandas as pd

        from typing import List, Optional, Sequence, Callable, Any

        from numpy.typing import ArrayLike

        from tqdm import tqdm

        from sklearn.metrics import mean_absolute_percentage_error

        from redxregressors import utilities, evaluate

        from redxregressors.utilities import seed_all

        log = logging.getLogger(__name__)



        def build_in_memory_loader_using_dataframe(

            df: pd.DataFrame,

            task_columns: List[str],

            featurizer: dc.feat.Featurizer,

            ids_column: str,

            smiles_column: str,

            weights: Optional[ArrayLike] = None,

            splitter: Optional[dc.splits.Splitter] = None,

            frac_train: float = 0.8,

            frac_valid: float = 0.1,

            frac_test: float = 0.1,

        ) -> tuple:

            """

            Function to build the in memory loader for the deepchem multitask regressor model.

            Args:

                tasks (List[str]): the names of the tasks to train on

                featurizer (dc.feat.Featurizer): the featurizer to use

                ids_column (Optional[str]): the name of the column containing the IDs

                smiles (List[str]): the SMILES strings

                targets (np.ndarray): the target values

                weights (np.ndarray): the weights for each task

                splitter (Optional[dc.splits.Splitter]): the splitter to use

            Returns:

                tuple: the training, validation and testing datasets and the splitter object if a splitter was provided otherwise the dataset and None, None, None

            """

            torch.use_deterministic_algorithms(True)

            tasks = task_columns

            targets = df[task_columns].values

            if weights is None:

                weights = np.ones((len(df), len(tasks)), dtype=np.float16)

            smiles = df[smiles_column].values

            ids = df[ids_column].values

            # Create the loader object to load the data into the model from memory

            loader = dc.data.InMemoryLoader(

                tasks=tasks,

                featurizer=featurizer,

                id_field=ids_column,

                log_every_n=10000,

            )

            # Create the dataset object

            dataset = loader.create_dataset(zip(smiles, targets, weights, ids), shard_size=2)

            if splitter is not None:

                # Split the dataset into training, validation and testing sets

                train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(

                    dataset,

                    frac_train=frac_train,

                    frac_valid=frac_valid,

                    frac_test=frac_test,

                    seed=utilities.random_seed,

                    log_every_n=10000,

                )

                return train_dataset, valid_dataset, test_dataset, splitter

            else:

                return dataset, None, None, None



        def build_in_memory_loader(

            tasks: ArrayLike,

            featurizer: dc.feat.Featurizer,

            ids_column: str,

            ids: ArrayLike,

            smiles: ArrayLike,

            targets: ArrayLike,

            weights: ArrayLike,

            splitter: Optional[dc.splits.Splitter] = None,

            frac_train: float = 0.8,

            frac_valid: float = 0.1,

            frac_test: float = 0.1,

        ) -> tuple:

            """

            Function to build the in memory loader for the deepchem multitask regressor model.

            Args:

                tasks (List[str]): the names of the tasks to train on

                featurizer (dc.feat.Featurizer): the featurizer to use

                ids_column (Optional[str]): the name of the column containing the IDs

                smiles (List[str]): the SMILES strings

                targets (np.ndarray): the target values

                weights (np.ndarray): the weights for each task

                splitter (Optional[dc.splits.Splitter]): the splitter to use

            Returns:

                tuple: the training, validation and testing datasets and the splitter object if a splitter was provided otherwise the dataset and None, None, None

            """

            torch.use_deterministic_algorithms(True)

            # Create the loader object to load the data into the model from memory

            loader = dc.data.InMemoryLoader(

                tasks=tasks,

                featurizer=featurizer,

                id_field=ids_column,

                log_every_n=10000,

            )

            # Create the dataset object

            dataset = loader.create_dataset(zip(smiles, targets, weights, ids), shard_size=2)

            if splitter is not None:

                # Split the dataset into training, validation and testing sets

                train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(

                    dataset,

                    frac_train=frac_train,

                    frac_valid=frac_valid,

                    frac_test=frac_test,

                    seed=utilities.random_seed,

                    log_every_n=10000,

                )

                return train_dataset, valid_dataset, test_dataset, splitter

            else:

                return dataset, None, None, None



        def fit_mtr_pytorch_model(

            model: dc.models.torch_models.torch_model.TorchModel,

            train_dataset: dc.data.data_loader.DataLoader,

            valid_dataset: dc.data.data_loader.DataLoader,

            epochs: int = 100,

            unique_string: Optional[str] = None,

        ) -> dc.models.torch_models.torch_model.TorchModel:

            """

            Function to fit a multitask regressor pytorch model using the deepchem library.

            Args:

                model (dc.models.torch_models.torch_model.TorchModel): the model to train

                train_dataset (dc.data.data_loader.DataLoader): the training dataset

                valid_dataset (dc.data.data_loader.DataLoader): the validation dataset

                epochs (int): the number of epochs to train for

                unique_string (Optional[str]): a unique string to append to the output files

            Returns:

                dc.models.torch_models.torch_model.TorchModel: the trained model

            """

            torch.use_deterministic_algorithms(True)

            # train the model using an explicit loop to allow for intermediate evaluation

            pbar = tqdm(range(epochs))

            train_scores = []

            valid_scores = []

            plot_epoch_numbers = []

            ts = {"mean-mean_squared_error": np.nan}

            vs = {"mean-mean_squared_error": np.nan}

            for i in pbar:

                pbar.set_description(

                    f"Processing epoch {i}: latest train MSE (L2-loss) mean over tasks {ts.get('mean-mean_squared_error'):.2f} lastest validation MSE (L2-loss) mean over tasks {vs.get('mean-mean_squared_error'):.2f}"

                )

                model.fit(train_dataset, nb_epoch=1, deterministic=True)

                if i % max(int(epochs * 0.1), 1) == 0:

                    ts = model.evaluate(

                        train_dataset,

                        [dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)],

                    )

                    train_scores.append([v for _, v in ts.items()][0])

                    vs = model.evaluate(

                        valid_dataset,

                        [dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)],

                    )

                    valid_scores.append([v for _, v in vs.items()][0])

                    plot_epoch_numbers.append(i)

                    log.debug(f"Train scores epoch {i}: {train_scores}")

                    log.debug(f"Validation scores epoch {i}: {valid_scores}")

                    evaluate.plot_metric_curves(

                        metrics=[train_scores, valid_scores],

                        metric_labels=["Training MSE", "Validation MSE"],

                        x=plot_epoch_numbers,

                        filename="multitask_regression_training_curves.png"

                        if unique_string is None

                        else f"multitask_regression_training_curves_{unique_string}.png",

                    )

            return model



        def fit_mtr_pytorch_model_per_task(

            model: dc.models.torch_models.torch_model.TorchModel,

            train_dataset: dc.data.data_loader.DataLoader,

            valid_dataset: dc.data.data_loader.DataLoader,

            epochs: int = 100,

            unique_string: Optional[str] = None,

        ) -> dc.models.torch_models.torch_model.TorchModel:

            """

            Function to fit a multitask regressor pytorch model using the deepchem library.

            Args:

                model (dc.models.torch_models.torch_model.TorchModel): the model to train

                train_dataset (dc.data.data_loader.DataLoader): the training dataset

                valid_dataset (dc.data.data_loader.DataLoader): the validation dataset

                epochs (int): the number of epochs to train for

                unique_string (Optional[str]): a unique string to append to the output files

            Returns:

                dc.models.torch_models.torch_model.TorchModel: the trained model

            """

            torch.use_deterministic_algorithms(True)

            # train the model using an explicit loop to allow for intermediate evaluation

            pbar = tqdm(range(epochs))

            train_scores = []

            valid_scores = []

            plot_epoch_numbers = []

            ts = {"mean-mean_squared_error": np.nan}

            vs = {"mean-mean_squared_error": np.nan}

            for i in pbar:

                pbar.set_description(

                    f"Processing epoch {i}: latest train MSE (L2-loss) mean over tasks {ts.get('mean-mean_squared_error'):.2f} lastest validation MSE (L2-loss) mean over tasks {vs.get('mean-mean_squared_error'):.2f}"

                )

                log.debug(

                    f"There are {train_dataset.y.shape[1]} tasks according to the shape of the y array"

                )

                for ith in range(train_dataset.y.shape[1]):

                    model.fit_task(train_dataset, task=ith, nb_epoch=1, deterministic=True)

                if i % max(int(epochs * 0.1), 1) == 0:

                    ts = model.evaluate(

                        train_dataset,

                        [dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)],

                    )

                    train_scores.append([v for _, v in ts.items()][0])

                    vs = model.evaluate(

                        valid_dataset,

                        [dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)],

                    )

                    valid_scores.append([v for _, v in vs.items()][0])

                    plot_epoch_numbers.append(i)

                    log.debug(f"Train scores epoch {i}: {train_scores}")

                    log.debug(f"Validation scores epoch {i}: {valid_scores}")

                    evaluate.plot_metric_curves(

                        metrics=[train_scores, valid_scores],

                        metric_labels=["Training MSE", "Validation MSE"],

                        x=plot_epoch_numbers,

                        filename="multitask_regression_training_curves.png"

                        if unique_string is None

                        else f"multitask_regression_training_curves_{unique_string}.png",

                    )

            return model



        def evaluate_mtr_pytorch_model(

            model: dc.models.torch_models.torch_model.TorchModel,

            test_dataset: dc.data.data_loader.DataLoader,

            tasks: List[str],

            unique_string: Optional[str] = None,

        ) -> pd.DataFrame:

            """

            Function to evaluate a multitask regressor pytorch model using the deepchem library.

            Args:

                model (dc.models.torch_models.torch_model.TorchModel): the model to evaluate

                test_dataset (dc.data.data_loader.DataLoader): the test dataset

                tasks (List[str]): the names of the tasks to evaluate

                unique_string (Optional[str]): a unique string to append to the output files

            Returns:

                pd.DataFrame: the test set metrics predictions

            """

            # evaluate the model

            avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

            avg_r2 = dc.metrics.Metric(dc.metrics.r2_score, np.mean)

            avg_perarson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

            avg_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean)

            avg_mape = dc.metrics.Metric(

                mean_absolute_percentage_error, np.mean, mode="regression"

            )

            # get the test set mean scores over tasks

            test_scores = model.evaluate(

                test_dataset,

                [avg_rms, avg_r2, avg_perarson_r2, avg_mae],

            )

            log.info(f"Test scores: {test_scores}")

            # get the test set predictions

            test_set_prediction = model.predict(test_dataset)

            # get the test set mean scores over tasks and per task scores

            mean_rmse_over_tasks, rmses = avg_rms.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            mean_cod_r2_over_tasks, cod_r2s = avg_r2.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            mean_pearsonr2_over_tasks, perarson_r2s = avg_perarson_r2.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            mean_mae_over_tasks, maes = avg_mae.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            mean_mape_over_tasks, mapes = avg_mape.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            test_set_metric_table = []

            for metric, means, values in zip(

                ["RMS", "R2", "Pearson R2", "MAE", "MAPE"],

                [

                    mean_rmse_over_tasks,

                    mean_cod_r2_over_tasks,

                    mean_pearsonr2_over_tasks,

                    mean_mae_over_tasks,

                    mean_mape_over_tasks,

                ],

                [rmses, cod_r2s, perarson_r2s, maes, mapes],

            ):

                # log.info(f"{metric} {values}")

                log.info(

                    f"Test {metric} mean cross task score: {means:.2f} per task: {' '.join([f'Task {tasks[ith]} {v:.2f}' for ith, v in enumerate(values)])}"

                )

                test_set_metric_table.append([metric, means] + values)

            test_set_metric_table_df = pd.DataFrame(

                test_set_metric_table,

                columns=["metric", "mean over tasks"] + [ent.lower() for ent in tasks],

            )

            test_set_metric_table_df.set_index("metric", inplace=True)

            test_set_metric_table_df.to_csv(

                "multitask_regression_test_set_metric_table.csv"

                if unique_string is None

                else f"multitask_regression_test_set_metric_table_{unique_string}.csv"

            )

            # plot the parity plot of the predictions

            for ith_task, task in enumerate(tasks):

                log.debug(test_set_prediction)

                log.debug(test_dataset.y[:, ith_task])

                log.debug(test_set_prediction[:, ith_task].flatten())

                _ = evaluate.parity_plot(

                    test_dataset.y[:, ith_task],

                    test_set_prediction[:, ith_task].flatten(),

                    filename=f"multitask_regression_task_{ith_task}_test_set_parity_plot.png"

                    if unique_string is None

                    else f"multitask_regression_task_{ith_task}_test_set_parity_plot_{unique_string}.png",

                )

                _ = evaluate.plot_residuals(

                    test_dataset.y[:, ith_task],

                    test_set_prediction[:, ith_task].flatten(),

                    filename=f"multitask_regression_task_{ith_task}_test_set_resdual_plot.png"

                    if unique_string is None

                    else f"multitask_regression_task_{ith_task}_test_set_residual_plot_{unique_string}.png",

                )

                _ = evaluate.plot_qq(

                    test_dataset.y[:, ith_task],

                    test_set_prediction[:, ith_task].flatten(),

                    filename=f"multitask_regression_task_{ith_task}_test_set_qq_plot.png"

                    if unique_string is None

                    else f"multitask_regression_task_{ith_task}_test_set_qq_plot_{unique_string}.png",

                )

            return test_set_metric_table_df



        def train_multitask_regressor(

            tasks: List[str],

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: str = "smiles",

            ids_column: Optional[str] = None,

            task_weights: Optional[List[float]] = None,

            epochs: int = 100,

            learning_rate: float = 0.0001,

            layer_sizes: List[int] = [1000, 1000],

            batch_size: int = 50,

            featurizer: Optional[dc.feat.Featurizer] = dc.feat.CircularFingerprint(

                size=1024, radius=2, chiral=True, bonds=True

            ),

            splitter: Optional[dc.splits.Splitter] = dc.splits.RandomSplitter(),

            unique_string: Optional[str] = None,

            fit_transformers: Optional[List[dc.trans.Transformer]] = None,

            pre_seed: bool = True,

            frac_train: float = 0.8,

            frac_valid: float = 0.1,

            frac_test: float = 0.1,

            uncertainty: bool = False,

            residual: bool = False,

            weight_decay_penalty: float = 0.0,

            weight_decay_penalty_type: str = "l2",

            dropouts: float | Sequence[float] = 0.5,

            activation_fns: Callable | str | Sequence[Callable | str] = "relu",

            train_dataset: Optional[dc.data.data_loader.DataLoader] = None,

            test_dataset: Optional[dc.data.data_loader.DataLoader] = None,

            valid_dataset: Optional[dc.data.data_loader.DataLoader] = None,

            **kwargs,

        ) -> tuple[Any, Any, Any, Any, pd.DataFrame]:

            """

            Function to train a multitask regressor using the deepchem library.

            Note the deepchem library is a wrapper around pytorch, Jax, tensorflow and keras.

            The examples online are unclear for the use of this model so this function is a wrapper to make it easier to use.

            The main difficulty is building the data set. By default the dispite naming the tasks when you define teh loader

            object it does not by defult pull these out as the tasks. In addition, the weights in the data set for each data

            point default to 0.0. Both of these mean that there are no target values for the model to train on.

            The symptom this is happening is a continous average loss of 0.0.

            It seem that one has to explicitly add these to the data set when running .create_dataset(). However again this is not clear

            as the fucntion signature does not have keys for these values. Instead it assumes an order of SMILES, targets, weights, ids

            passed as an iterable. This is implemented in this function to make it easier to use and abstract the details of the interfaces

            of the deepchem library.

            Example:

            ```python

            import logging

            from redxregressors import deep_net_models

            dc_logger = logging.getLogger("deepchem")

            dc_logger.setLevel(logging.WARNING)

            data_df_bind = pd.read_csv("data/kinase_data_bind.csv")

            tasks = ["kinase_1_pIC50", "kinase_2_pIC50"]

            smiles_column = "smiles"

            ids_column = "name_id"

            reg, tr, test, val, test_set_metrics_df = deep_net_models.train_multitask_regressor(

                data_df=data_df_bind,

                tasks=tasks,

                smiles_column=smiles_column,

                ids_column=ids_column,

                epochs=200,

                layer_sizes=[1000, 1000],

                )

            ```

            Args:

                data_df (pd.DataFrame): the data frame containing the data

                tasks (List[str]): the names of the tasks to train on

                smiles_column (str): the name of the column containing the SMILES strings

                ids_column (Optional[str]): the name of the column containing the IDs

                task_weights (Optional[List[float]]): the weights for each task

                epochs (int): the number of epochs to train for

                learning_rate (float): the learning rate for the model

                layer_sizes (List[int]): the sizes of the layers in the model

                batch_size (int): the batch size for the model

                featurizer (Optional[dc.feat.Featurizer]): the featurizer to use

                splitter (Optional[dc.splits.Splitter]): the splitter to use

                unique_string (Optional[str]): a unique string to append to the output files

                pre_seed (bool): whether to pre seed the random number generators

                frac_train (float): the fraction of the data to use for training

                frac_valid (float): the fraction of the data to use for validation

                frac_test (float): the fraction of the data to use for testing

                uncertainty (bool): whether to use uncertainty in the model

                residual (bool): whether to use residual connections in the model

                weight_decay_penalty (float): the weight decay penalty to use

                weight_decay_penalty_type (str): the type of weight decay penalty to use

                dropouts (float | Sequence[float]): the dropouts to use

                activation_fns (Callable | str | Sequence[Callable | str]): the activation functions to use

                **kwargs: additional keyword arguments to pass to the model

            Returns:

                tuple[Any, Any, Any, Any, pd.DataFrame]: the trained model, the training data set, the validation data set, the test data set, tests set metrics predictions

            """

            # Pre seed the random number generators

            if pre_seed is True:

                seed_all()

                torch.use_deterministic_algorithms(True)

            # Get all of the pieces that form the data set as rew data

            if train_dataset is None and test_dataset is None and valid_dataset is None:

                log.debug(

                    "Building the data sets from pandas data frame input and column headers"

                )

                smiles = data_df[smiles_column].values

                if ids_column is not None:

                    ids = data_df[ids_column].values

                else:

                    ids = data_df.index.values

                if task_weights is None:

                    weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)

                else:

                    weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)

                    for indx, tw in enumerate(task_weights):

                        weights[:, indx] *= task_weights[indx]

                if len(tasks) == 1:

                    log.warning(

                        "Only one task was provided, this is not a multitask model consider a different model type"

                    )

                else:

                    targets = data_df[tasks].values

                # get the data sets for training, validation and testing

                train_dataset, valid_dataset, test_dataset, splitter = build_in_memory_loader(

                    tasks=tasks,

                    featurizer=featurizer,

                    ids_column=ids_column,

                    ids=ids,

                    smiles=smiles,

                    targets=targets,

                    weights=weights,

                    splitter=splitter,

                    frac_train=frac_train,

                    frac_valid=frac_valid,

                    frac_test=frac_test,

                )

            # if the data sets are provided then use them

            else:

                log.debug("Using the provided data sets")

                splitter = None

            log.debug(f"Training dataset shape: {train_dataset.get_data_shape()}")

            log.debug(train_dataset)

            log.debug(train_dataset.X)

            log.debug(train_dataset.y)

            # instantiate the model

            if fit_transformers is None:

                model = dc.models.MultitaskRegressor(

                    len(tasks),

                    train_dataset.get_data_shape()[0],

                    layer_sizes=layer_sizes,

                    batch_size=batch_size,

                    learning_rate=learning_rate,

                    weight_decay_penalty=weight_decay_penalty,

                    weight_decay_penalty_type=weight_decay_penalty_type,

                    dropouts=dropouts,

                    activation_fns=activation_fns,

                    uncertainty=uncertainty,

                    residual=residual,

                    **kwargs,

                )

            else:

                fit_transformers = [

                    ent(train_dataset, transform_X=True, transform_y=False)

                    for ent in fit_transformers

                ]

                log.debug(train_dataset.get_data_shape()[0])

                model = dc.models.MultitaskFitTransformRegressor(

                    len(tasks),

                    train_dataset.get_data_shape()[0],

                    layer_sizes=layer_sizes,

                    batch_size=batch_size,

                    learning_rate=learning_rate,

                    weight_decay_penalty=weight_decay_penalty,

                    weight_decay_penalty_type=weight_decay_penalty_type,

                    dropouts=dropouts,

                    activation_fns=activation_fns,

                    uncertainty=uncertainty,

                    residual=residual,

                    fit_transformers=fit_transformers,

                    **kwargs,

                )

                # train the model using an explicit loop to allow for intermediate evaluation

            model = fit_mtr_pytorch_model(

                model=model,

                train_dataset=train_dataset,

                valid_dataset=valid_dataset,

                epochs=epochs,

                unique_string=unique_string,

            )

            # evaluate the model using the test set

            test_set_df = evaluate_mtr_pytorch_model(

                model=model, test_dataset=test_dataset, tasks=tasks, unique_string=unique_string

            )

            return model, train_dataset, valid_dataset, test_dataset, test_set_df



        def train_progressive_multitask_regressor(

            data_df: pd.DataFrame,

            tasks: List[str],

            smiles_column: str = "smiles",

            ids_column: Optional[str] = None,

            task_weights: Optional[List[float]] = None,

            epochs: int = 100,

            learning_rate: float = 0.0001,

            layer_sizes: List[int] = [1000, 1000],

            batch_size: int = 50,

            featurizer: Optional[dc.feat.Featurizer] = dc.feat.CircularFingerprint(

                size=1024, radius=2, chiral=True, bonds=True

            ),

            splitter: Optional[dc.splits.Splitter] = dc.splits.RandomSplitter(),

            unique_string: Optional[str] = None,

            pre_seed: bool = True,

            frac_train: float = 0.8,

            frac_valid: float = 0.1,

            frac_test: float = 0.1,

            uncertainty: bool = False,

            residual: bool = False,

            weight_decay_penalty: float = 0.0,

            weight_decay_penalty_type: str = "l2",

            dropouts: float | Sequence[float] = 0.5,

            activation_fns: Callable | str | Sequence[Callable | str] = "relu",

            train_per_task: bool = False,

            **kwargs,

        ) -> tuple[Any, Any, Any, Any, pd.DataFrame]:

            """

            Function to train a multitask regressor using the deepchem library.

            Note the deepchem library is a wrapper around pytorch, Jax, tensorflow and keras.

            The examples online are unclear for the use of this model so this function is a wrapper to make it easier to use.

            The main difficulty is building the data set. By default the dispite naming the tasks when you define teh loader

            object it does not by defult pull these out as the tasks. In addition, the weights in the data set for each data

            point default to 0.0. Both of these mean that there are no target values for the model to train on.

            The symptom this is happening is a continous average loss of 0.0.

            It seem that one has to explicitly add these to the data set when running .create_dataset(). However again this is not clear

            as the fucntion signature does not have keys for these values. Instead it assumes an order of SMILES, targets, weights, ids

            passed as an iterable. This is implemented in this function to make it easier to use and abstract the details of the interfaces

            of the deepchem library.

            Example:

            ```python

            import logging

            from redxregressors import deep_net_models

            dc_logger = logging.getLogger("deepchem")

            dc_logger.setLevel(logging.WARNING)

            data_df_bind = pd.read_csv("data/kinase_data_bind.csv")

            tasks = ["kinase_1_pIC50", "kinase_2_pIC50"]

            smiles_column = "smiles"

            ids_column = "name_id"

            reg, tr, test, val, test_set_metrics_df = deep_net_models.train_multitask_regressor(

                data_df=data_df_bind,

                tasks=tasks,

                smiles_column=smiles_column,

                ids_column=ids_column,

                epochs=200,

                layer_sizes=[1000, 1000],

                )

            ```

            Args:

                data_df (pd.DataFrame): the data frame containing the data

                tasks (List[str]): the names of the tasks to train on

                smiles_column (str): the name of the column containing the SMILES strings

                ids_column (Optional[str]): the name of the column containing the IDs

                task_weights (Optional[List[float]]): the weights for each task

                epochs (int): the number of epochs to train for

                learning_rate (float): the learning rate for the model

                layer_sizes (List[int]): the sizes of the layers in the model

                batch_size (int): the batch size for the model

                featurizer (Optional[dc.feat.Featurizer]): the featurizer to use

                splitter (Optional[dc.splits.Splitter]): the splitter to use

                unique_string (Optional[str]): a unique string to append to the output files

                pre_seed (bool): whether to pre seed the random number generators

                frac_train (float): the fraction of the data to use for training

                frac_valid (float): the fraction of the data to use for validation

                frac_test (float): the fraction of the data to use for testing

                uncertainty (bool): whether to use uncertainty in the model

                residual (bool): whether to use residual connections in the model

                weight_decay_penalty (float): the weight decay penalty to use

                weight_decay_penalty_type (str): the type of weight decay penalty to use

                dropouts (float | Sequence[float]): the dropouts to use

                activation_fns (Callable | str | Sequence[Callable | str]): the activation functions to use

                train_per_task (bool): whether to train the model per task sequentially

                **kwargs: additional keyword arguments to pass to the model

            Returns:

                tuple[Any, Any, Any, Any, pd.DataFrame]: the trained model, the training data set, the validation data set, the test data set, tests set metrics predictions

            """

            # Pre seed the random number generators

            if pre_seed is True:

                seed_all()

                torch.use_deterministic_algorithms(True)

            # Get all of the pieces that form the data set as rew data

            smiles = data_df[smiles_column].values

            if ids_column is not None:

                ids = data_df[ids_column].values

            else:

                ids = data_df.index.values

            if task_weights is None:

                weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)

            else:

                weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)

                for indx, tw in enumerate(task_weights):

                    weights[:, indx] *= task_weights[indx]

            if len(tasks) == 1:

                log.warning(

                    "Only one task was provided, this is not a multitask model consider a different model type"

                )

            else:

                targets = data_df[tasks].values

            # get the data sets for training, validation and testing

            train_dataset, valid_dataset, test_dataset, splitter = build_in_memory_loader(

                tasks=tasks,

                featurizer=featurizer,

                ids_column=ids_column,

                ids=ids,

                smiles=smiles,

                targets=targets,

                weights=weights,

                splitter=splitter,

                frac_train=frac_train,

                frac_valid=frac_valid,

                frac_test=frac_test,

            )

            log.debug(f"Training dataset shape: {train_dataset.get_data_shape()}")

            log.debug(train_dataset)

            log.debug(train_dataset.X)

            log.debug(train_dataset.y)

            # instantiate the model

            model = dc.models.torch_models.ProgressiveMultitaskModel(

                len(tasks),

                train_dataset.get_data_shape()[0],

                mode="regression",

                layer_sizes=layer_sizes,

                batch_size=batch_size,

                learning_rate=learning_rate,

                weight_decay_penalty=weight_decay_penalty,

                weight_decay_penalty_type=weight_decay_penalty_type,

                dropouts=dropouts,

                activation_fns=activation_fns,

                uncertainty=uncertainty,

                residual=residual,

                **kwargs,

            )

            # train the model using an explicit loop to allow for intermediate evaluation

            if train_per_task is False:

                model = fit_mtr_pytorch_model(

                    model=model,

                    train_dataset=train_dataset,

                    valid_dataset=valid_dataset,

                    epochs=epochs,

                    unique_string=unique_string,

                )

            else:

                fit_mtr_pytorch_model_per_task(

                    model=model,

                    train_dataset=train_dataset,

                    valid_dataset=valid_dataset,

                    epochs=epochs,

                    unique_string=unique_string,

                )

            # evaluate the model using the test set

            test_set_df = evaluate_mtr_pytorch_model(

                model=model, test_dataset=test_dataset, tasks=tasks, unique_string=unique_string

            )

            return model, train_dataset, valid_dataset, test_dataset, test_set_df



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### build_in_memory_loader

```python3
def build_in_memory_loader(
    tasks: Union[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]],
    featurizer: deepchem.feat.base_classes.Featurizer,
    ids_column: str,
    ids: Union[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]],
    smiles: Union[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]],
    targets: Union[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]],
    weights: Union[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]],
    splitter: Optional[deepchem.splits.splitters.Splitter] = None,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1
) -> tuple
```

Function to build the in memory loader for the deepchem multitask regressor model.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| tasks | List[str] | the names of the tasks to train on | None |
| featurizer | dc.feat.Featurizer | the featurizer to use | None |
| ids_column | Optional[str] | the name of the column containing the IDs | None |
| smiles | List[str] | the SMILES strings | None |
| targets | np.ndarray | the target values | None |
| weights | np.ndarray | the weights for each task | None |
| splitter | Optional[dc.splits.Splitter] | the splitter to use | None |

**Returns:**

| Type | Description |
|---|---|
| tuple | the training, validation and testing datasets and the splitter object if a splitter was provided otherwise the dataset and None, None, None |

??? example "View Source"
        def build_in_memory_loader(

            tasks: ArrayLike,

            featurizer: dc.feat.Featurizer,

            ids_column: str,

            ids: ArrayLike,

            smiles: ArrayLike,

            targets: ArrayLike,

            weights: ArrayLike,

            splitter: Optional[dc.splits.Splitter] = None,

            frac_train: float = 0.8,

            frac_valid: float = 0.1,

            frac_test: float = 0.1,

        ) -> tuple:

            """

            Function to build the in memory loader for the deepchem multitask regressor model.

            Args:

                tasks (List[str]): the names of the tasks to train on

                featurizer (dc.feat.Featurizer): the featurizer to use

                ids_column (Optional[str]): the name of the column containing the IDs

                smiles (List[str]): the SMILES strings

                targets (np.ndarray): the target values

                weights (np.ndarray): the weights for each task

                splitter (Optional[dc.splits.Splitter]): the splitter to use

            Returns:

                tuple: the training, validation and testing datasets and the splitter object if a splitter was provided otherwise the dataset and None, None, None

            """

            torch.use_deterministic_algorithms(True)

            # Create the loader object to load the data into the model from memory

            loader = dc.data.InMemoryLoader(

                tasks=tasks,

                featurizer=featurizer,

                id_field=ids_column,

                log_every_n=10000,

            )

            # Create the dataset object

            dataset = loader.create_dataset(zip(smiles, targets, weights, ids), shard_size=2)

            if splitter is not None:

                # Split the dataset into training, validation and testing sets

                train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(

                    dataset,

                    frac_train=frac_train,

                    frac_valid=frac_valid,

                    frac_test=frac_test,

                    seed=utilities.random_seed,

                    log_every_n=10000,

                )

                return train_dataset, valid_dataset, test_dataset, splitter

            else:

                return dataset, None, None, None


### build_in_memory_loader_using_dataframe

```python3
def build_in_memory_loader_using_dataframe(
    df: pandas.core.frame.DataFrame,
    task_columns: List[str],
    featurizer: deepchem.feat.base_classes.Featurizer,
    ids_column: str,
    smiles_column: str,
    weights: Union[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes], NoneType] = None,
    splitter: Optional[deepchem.splits.splitters.Splitter] = None,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1
) -> tuple
```

Function to build the in memory loader for the deepchem multitask regressor model.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| tasks | List[str] | the names of the tasks to train on | None |
| featurizer | dc.feat.Featurizer | the featurizer to use | None |
| ids_column | Optional[str] | the name of the column containing the IDs | None |
| smiles | List[str] | the SMILES strings | None |
| targets | np.ndarray | the target values | None |
| weights | np.ndarray | the weights for each task | None |
| splitter | Optional[dc.splits.Splitter] | the splitter to use | None |

**Returns:**

| Type | Description |
|---|---|
| tuple | the training, validation and testing datasets and the splitter object if a splitter was provided otherwise the dataset and None, None, None |

??? example "View Source"
        def build_in_memory_loader_using_dataframe(

            df: pd.DataFrame,

            task_columns: List[str],

            featurizer: dc.feat.Featurizer,

            ids_column: str,

            smiles_column: str,

            weights: Optional[ArrayLike] = None,

            splitter: Optional[dc.splits.Splitter] = None,

            frac_train: float = 0.8,

            frac_valid: float = 0.1,

            frac_test: float = 0.1,

        ) -> tuple:

            """

            Function to build the in memory loader for the deepchem multitask regressor model.

            Args:

                tasks (List[str]): the names of the tasks to train on

                featurizer (dc.feat.Featurizer): the featurizer to use

                ids_column (Optional[str]): the name of the column containing the IDs

                smiles (List[str]): the SMILES strings

                targets (np.ndarray): the target values

                weights (np.ndarray): the weights for each task

                splitter (Optional[dc.splits.Splitter]): the splitter to use

            Returns:

                tuple: the training, validation and testing datasets and the splitter object if a splitter was provided otherwise the dataset and None, None, None

            """

            torch.use_deterministic_algorithms(True)

            tasks = task_columns

            targets = df[task_columns].values

            if weights is None:

                weights = np.ones((len(df), len(tasks)), dtype=np.float16)

            smiles = df[smiles_column].values

            ids = df[ids_column].values

            # Create the loader object to load the data into the model from memory

            loader = dc.data.InMemoryLoader(

                tasks=tasks,

                featurizer=featurizer,

                id_field=ids_column,

                log_every_n=10000,

            )

            # Create the dataset object

            dataset = loader.create_dataset(zip(smiles, targets, weights, ids), shard_size=2)

            if splitter is not None:

                # Split the dataset into training, validation and testing sets

                train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(

                    dataset,

                    frac_train=frac_train,

                    frac_valid=frac_valid,

                    frac_test=frac_test,

                    seed=utilities.random_seed,

                    log_every_n=10000,

                )

                return train_dataset, valid_dataset, test_dataset, splitter

            else:

                return dataset, None, None, None


### evaluate_mtr_pytorch_model

```python3
def evaluate_mtr_pytorch_model(
    model: deepchem.models.torch_models.torch_model.TorchModel,
    test_dataset: deepchem.data.data_loader.DataLoader,
    tasks: List[str],
    unique_string: Optional[str] = None
) -> pandas.core.frame.DataFrame
```

Function to evaluate a multitask regressor pytorch model using the deepchem library.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model | dc.models.torch_models.torch_model.TorchModel | the model to evaluate | None |
| test_dataset | dc.data.data_loader.DataLoader | the test dataset | None |
| tasks | List[str] | the names of the tasks to evaluate | None |
| unique_string | Optional[str] | a unique string to append to the output files | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the test set metrics predictions |

??? example "View Source"
        def evaluate_mtr_pytorch_model(

            model: dc.models.torch_models.torch_model.TorchModel,

            test_dataset: dc.data.data_loader.DataLoader,

            tasks: List[str],

            unique_string: Optional[str] = None,

        ) -> pd.DataFrame:

            """

            Function to evaluate a multitask regressor pytorch model using the deepchem library.

            Args:

                model (dc.models.torch_models.torch_model.TorchModel): the model to evaluate

                test_dataset (dc.data.data_loader.DataLoader): the test dataset

                tasks (List[str]): the names of the tasks to evaluate

                unique_string (Optional[str]): a unique string to append to the output files

            Returns:

                pd.DataFrame: the test set metrics predictions

            """

            # evaluate the model

            avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

            avg_r2 = dc.metrics.Metric(dc.metrics.r2_score, np.mean)

            avg_perarson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

            avg_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean)

            avg_mape = dc.metrics.Metric(

                mean_absolute_percentage_error, np.mean, mode="regression"

            )

            # get the test set mean scores over tasks

            test_scores = model.evaluate(

                test_dataset,

                [avg_rms, avg_r2, avg_perarson_r2, avg_mae],

            )

            log.info(f"Test scores: {test_scores}")

            # get the test set predictions

            test_set_prediction = model.predict(test_dataset)

            # get the test set mean scores over tasks and per task scores

            mean_rmse_over_tasks, rmses = avg_rms.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            mean_cod_r2_over_tasks, cod_r2s = avg_r2.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            mean_pearsonr2_over_tasks, perarson_r2s = avg_perarson_r2.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            mean_mae_over_tasks, maes = avg_mae.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            mean_mape_over_tasks, mapes = avg_mape.compute_metric(

                test_dataset.y, test_set_prediction, per_task_metrics=True

            )

            test_set_metric_table = []

            for metric, means, values in zip(

                ["RMS", "R2", "Pearson R2", "MAE", "MAPE"],

                [

                    mean_rmse_over_tasks,

                    mean_cod_r2_over_tasks,

                    mean_pearsonr2_over_tasks,

                    mean_mae_over_tasks,

                    mean_mape_over_tasks,

                ],

                [rmses, cod_r2s, perarson_r2s, maes, mapes],

            ):

                # log.info(f"{metric} {values}")

                log.info(

                    f"Test {metric} mean cross task score: {means:.2f} per task: {' '.join([f'Task {tasks[ith]} {v:.2f}' for ith, v in enumerate(values)])}"

                )

                test_set_metric_table.append([metric, means] + values)

            test_set_metric_table_df = pd.DataFrame(

                test_set_metric_table,

                columns=["metric", "mean over tasks"] + [ent.lower() for ent in tasks],

            )

            test_set_metric_table_df.set_index("metric", inplace=True)

            test_set_metric_table_df.to_csv(

                "multitask_regression_test_set_metric_table.csv"

                if unique_string is None

                else f"multitask_regression_test_set_metric_table_{unique_string}.csv"

            )

            # plot the parity plot of the predictions

            for ith_task, task in enumerate(tasks):

                log.debug(test_set_prediction)

                log.debug(test_dataset.y[:, ith_task])

                log.debug(test_set_prediction[:, ith_task].flatten())

                _ = evaluate.parity_plot(

                    test_dataset.y[:, ith_task],

                    test_set_prediction[:, ith_task].flatten(),

                    filename=f"multitask_regression_task_{ith_task}_test_set_parity_plot.png"

                    if unique_string is None

                    else f"multitask_regression_task_{ith_task}_test_set_parity_plot_{unique_string}.png",

                )

                _ = evaluate.plot_residuals(

                    test_dataset.y[:, ith_task],

                    test_set_prediction[:, ith_task].flatten(),

                    filename=f"multitask_regression_task_{ith_task}_test_set_resdual_plot.png"

                    if unique_string is None

                    else f"multitask_regression_task_{ith_task}_test_set_residual_plot_{unique_string}.png",

                )

                _ = evaluate.plot_qq(

                    test_dataset.y[:, ith_task],

                    test_set_prediction[:, ith_task].flatten(),

                    filename=f"multitask_regression_task_{ith_task}_test_set_qq_plot.png"

                    if unique_string is None

                    else f"multitask_regression_task_{ith_task}_test_set_qq_plot_{unique_string}.png",

                )

            return test_set_metric_table_df


### fit_mtr_pytorch_model

```python3
def fit_mtr_pytorch_model(
    model: deepchem.models.torch_models.torch_model.TorchModel,
    train_dataset: deepchem.data.data_loader.DataLoader,
    valid_dataset: deepchem.data.data_loader.DataLoader,
    epochs: int = 100,
    unique_string: Optional[str] = None
) -> deepchem.models.torch_models.torch_model.TorchModel
```

Function to fit a multitask regressor pytorch model using the deepchem library.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model | dc.models.torch_models.torch_model.TorchModel | the model to train | None |
| train_dataset | dc.data.data_loader.DataLoader | the training dataset | None |
| valid_dataset | dc.data.data_loader.DataLoader | the validation dataset | None |
| epochs | int | the number of epochs to train for | None |
| unique_string | Optional[str] | a unique string to append to the output files | None |

**Returns:**

| Type | Description |
|---|---|
| dc.models.torch_models.torch_model.TorchModel | the trained model |

??? example "View Source"
        def fit_mtr_pytorch_model(

            model: dc.models.torch_models.torch_model.TorchModel,

            train_dataset: dc.data.data_loader.DataLoader,

            valid_dataset: dc.data.data_loader.DataLoader,

            epochs: int = 100,

            unique_string: Optional[str] = None,

        ) -> dc.models.torch_models.torch_model.TorchModel:

            """

            Function to fit a multitask regressor pytorch model using the deepchem library.

            Args:

                model (dc.models.torch_models.torch_model.TorchModel): the model to train

                train_dataset (dc.data.data_loader.DataLoader): the training dataset

                valid_dataset (dc.data.data_loader.DataLoader): the validation dataset

                epochs (int): the number of epochs to train for

                unique_string (Optional[str]): a unique string to append to the output files

            Returns:

                dc.models.torch_models.torch_model.TorchModel: the trained model

            """

            torch.use_deterministic_algorithms(True)

            # train the model using an explicit loop to allow for intermediate evaluation

            pbar = tqdm(range(epochs))

            train_scores = []

            valid_scores = []

            plot_epoch_numbers = []

            ts = {"mean-mean_squared_error": np.nan}

            vs = {"mean-mean_squared_error": np.nan}

            for i in pbar:

                pbar.set_description(

                    f"Processing epoch {i}: latest train MSE (L2-loss) mean over tasks {ts.get('mean-mean_squared_error'):.2f} lastest validation MSE (L2-loss) mean over tasks {vs.get('mean-mean_squared_error'):.2f}"

                )

                model.fit(train_dataset, nb_epoch=1, deterministic=True)

                if i % max(int(epochs * 0.1), 1) == 0:

                    ts = model.evaluate(

                        train_dataset,

                        [dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)],

                    )

                    train_scores.append([v for _, v in ts.items()][0])

                    vs = model.evaluate(

                        valid_dataset,

                        [dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)],

                    )

                    valid_scores.append([v for _, v in vs.items()][0])

                    plot_epoch_numbers.append(i)

                    log.debug(f"Train scores epoch {i}: {train_scores}")

                    log.debug(f"Validation scores epoch {i}: {valid_scores}")

                    evaluate.plot_metric_curves(

                        metrics=[train_scores, valid_scores],

                        metric_labels=["Training MSE", "Validation MSE"],

                        x=plot_epoch_numbers,

                        filename="multitask_regression_training_curves.png"

                        if unique_string is None

                        else f"multitask_regression_training_curves_{unique_string}.png",

                    )

            return model


### fit_mtr_pytorch_model_per_task

```python3
def fit_mtr_pytorch_model_per_task(
    model: deepchem.models.torch_models.torch_model.TorchModel,
    train_dataset: deepchem.data.data_loader.DataLoader,
    valid_dataset: deepchem.data.data_loader.DataLoader,
    epochs: int = 100,
    unique_string: Optional[str] = None
) -> deepchem.models.torch_models.torch_model.TorchModel
```

Function to fit a multitask regressor pytorch model using the deepchem library.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model | dc.models.torch_models.torch_model.TorchModel | the model to train | None |
| train_dataset | dc.data.data_loader.DataLoader | the training dataset | None |
| valid_dataset | dc.data.data_loader.DataLoader | the validation dataset | None |
| epochs | int | the number of epochs to train for | None |
| unique_string | Optional[str] | a unique string to append to the output files | None |

**Returns:**

| Type | Description |
|---|---|
| dc.models.torch_models.torch_model.TorchModel | the trained model |

??? example "View Source"
        def fit_mtr_pytorch_model_per_task(

            model: dc.models.torch_models.torch_model.TorchModel,

            train_dataset: dc.data.data_loader.DataLoader,

            valid_dataset: dc.data.data_loader.DataLoader,

            epochs: int = 100,

            unique_string: Optional[str] = None,

        ) -> dc.models.torch_models.torch_model.TorchModel:

            """

            Function to fit a multitask regressor pytorch model using the deepchem library.

            Args:

                model (dc.models.torch_models.torch_model.TorchModel): the model to train

                train_dataset (dc.data.data_loader.DataLoader): the training dataset

                valid_dataset (dc.data.data_loader.DataLoader): the validation dataset

                epochs (int): the number of epochs to train for

                unique_string (Optional[str]): a unique string to append to the output files

            Returns:

                dc.models.torch_models.torch_model.TorchModel: the trained model

            """

            torch.use_deterministic_algorithms(True)

            # train the model using an explicit loop to allow for intermediate evaluation

            pbar = tqdm(range(epochs))

            train_scores = []

            valid_scores = []

            plot_epoch_numbers = []

            ts = {"mean-mean_squared_error": np.nan}

            vs = {"mean-mean_squared_error": np.nan}

            for i in pbar:

                pbar.set_description(

                    f"Processing epoch {i}: latest train MSE (L2-loss) mean over tasks {ts.get('mean-mean_squared_error'):.2f} lastest validation MSE (L2-loss) mean over tasks {vs.get('mean-mean_squared_error'):.2f}"

                )

                log.debug(

                    f"There are {train_dataset.y.shape[1]} tasks according to the shape of the y array"

                )

                for ith in range(train_dataset.y.shape[1]):

                    model.fit_task(train_dataset, task=ith, nb_epoch=1, deterministic=True)

                if i % max(int(epochs * 0.1), 1) == 0:

                    ts = model.evaluate(

                        train_dataset,

                        [dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)],

                    )

                    train_scores.append([v for _, v in ts.items()][0])

                    vs = model.evaluate(

                        valid_dataset,

                        [dc.metrics.Metric(dc.metrics.mean_squared_error, np.mean)],

                    )

                    valid_scores.append([v for _, v in vs.items()][0])

                    plot_epoch_numbers.append(i)

                    log.debug(f"Train scores epoch {i}: {train_scores}")

                    log.debug(f"Validation scores epoch {i}: {valid_scores}")

                    evaluate.plot_metric_curves(

                        metrics=[train_scores, valid_scores],

                        metric_labels=["Training MSE", "Validation MSE"],

                        x=plot_epoch_numbers,

                        filename="multitask_regression_training_curves.png"

                        if unique_string is None

                        else f"multitask_regression_training_curves_{unique_string}.png",

                    )

            return model


### train_multitask_regressor

```python3
def train_multitask_regressor(
    tasks: List[str],
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: str = 'smiles',
    ids_column: Optional[str] = None,
    task_weights: Optional[List[float]] = None,
    epochs: int = 100,
    learning_rate: float = 0.0001,
    layer_sizes: List[int] = [1000, 1000],
    batch_size: int = 50,
    featurizer: Optional[deepchem.feat.base_classes.Featurizer] = CircularFingerprint[radius=2, size=1024, chiral=True, bonds=True, features=False, sparse=False, smiles=False, is_counts_based=False],
    splitter: Optional[deepchem.splits.splitters.Splitter] = RandomSplitter[],
    unique_string: Optional[str] = None,
    fit_transformers: Optional[List[transformers.Transformer]] = None,
    pre_seed: bool = True,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    uncertainty: bool = False,
    residual: bool = False,
    weight_decay_penalty: float = 0.0,
    weight_decay_penalty_type: str = 'l2',
    dropouts: Union[float, Sequence[float]] = 0.5,
    activation_fns: Union[Callable, str, Sequence[Union[Callable, str]]] = 'relu',
    train_dataset: Optional[deepchem.data.data_loader.DataLoader] = None,
    test_dataset: Optional[deepchem.data.data_loader.DataLoader] = None,
    valid_dataset: Optional[deepchem.data.data_loader.DataLoader] = None,
    **kwargs
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any, pandas.core.frame.DataFrame]
```

Function to train a multitask regressor using the deepchem library.

Note the deepchem library is a wrapper around pytorch, Jax, tensorflow and keras.

The examples online are unclear for the use of this model so this function is a wrapper to make it easier to use.
The main difficulty is building the data set. By default the dispite naming the tasks when you define teh loader
object it does not by defult pull these out as the tasks. In addition, the weights in the data set for each data
point default to 0.0. Both of these mean that there are no target values for the model to train on.
The symptom this is happening is a continous average loss of 0.0.

It seem that one has to explicitly add these to the data set when running .create_dataset(). However again this is not clear
as the fucntion signature does not have keys for these values. Instead it assumes an order of SMILES, targets, weights, ids
passed as an iterable. This is implemented in this function to make it easier to use and abstract the details of the interfaces
of the deepchem library.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data_df | pd.DataFrame | the data frame containing the data | None |
| tasks | List[str] | the names of the tasks to train on | None |
| smiles_column | str | the name of the column containing the SMILES strings | None |
| ids_column | Optional[str] | the name of the column containing the IDs | None |
| task_weights | Optional[List[float]] | the weights for each task | None |
| epochs | int | the number of epochs to train for | None |
| learning_rate | float | the learning rate for the model | None |
| layer_sizes | List[int] | the sizes of the layers in the model | None |
| batch_size | int | the batch size for the model | None |
| featurizer | Optional[dc.feat.Featurizer] | the featurizer to use | None |
| splitter | Optional[dc.splits.Splitter] | the splitter to use | None |
| unique_string | Optional[str] | a unique string to append to the output files | None |
| pre_seed | bool | whether to pre seed the random number generators | None |
| frac_train | float | the fraction of the data to use for training | None |
| frac_valid | float | the fraction of the data to use for validation | None |
| frac_test | float | the fraction of the data to use for testing | None |
| uncertainty | bool | whether to use uncertainty in the model | None |
| residual | bool | whether to use residual connections in the model | None |
| weight_decay_penalty | float | the weight decay penalty to use | None |
| weight_decay_penalty_type | str | the type of weight decay penalty to use | None |
| dropouts | float | Sequence[float] | the dropouts to use | None |
| activation_fns | Callable | str | Sequence[Callable | str] | the activation functions to use | None |
| **kwargs | None | additional keyword arguments to pass to the model | None |

**Returns:**

| Type | Description |
|---|---|
| tuple[Any, Any, Any, Any, pd.DataFrame] | the trained model, the training data set, the validation data set, the test data set, tests set metrics predictions |

??? example "View Source"
        def train_multitask_regressor(

            tasks: List[str],

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: str = "smiles",

            ids_column: Optional[str] = None,

            task_weights: Optional[List[float]] = None,

            epochs: int = 100,

            learning_rate: float = 0.0001,

            layer_sizes: List[int] = [1000, 1000],

            batch_size: int = 50,

            featurizer: Optional[dc.feat.Featurizer] = dc.feat.CircularFingerprint(

                size=1024, radius=2, chiral=True, bonds=True

            ),

            splitter: Optional[dc.splits.Splitter] = dc.splits.RandomSplitter(),

            unique_string: Optional[str] = None,

            fit_transformers: Optional[List[dc.trans.Transformer]] = None,

            pre_seed: bool = True,

            frac_train: float = 0.8,

            frac_valid: float = 0.1,

            frac_test: float = 0.1,

            uncertainty: bool = False,

            residual: bool = False,

            weight_decay_penalty: float = 0.0,

            weight_decay_penalty_type: str = "l2",

            dropouts: float | Sequence[float] = 0.5,

            activation_fns: Callable | str | Sequence[Callable | str] = "relu",

            train_dataset: Optional[dc.data.data_loader.DataLoader] = None,

            test_dataset: Optional[dc.data.data_loader.DataLoader] = None,

            valid_dataset: Optional[dc.data.data_loader.DataLoader] = None,

            **kwargs,

        ) -> tuple[Any, Any, Any, Any, pd.DataFrame]:

            """

            Function to train a multitask regressor using the deepchem library.

            Note the deepchem library is a wrapper around pytorch, Jax, tensorflow and keras.

            The examples online are unclear for the use of this model so this function is a wrapper to make it easier to use.

            The main difficulty is building the data set. By default the dispite naming the tasks when you define teh loader

            object it does not by defult pull these out as the tasks. In addition, the weights in the data set for each data

            point default to 0.0. Both of these mean that there are no target values for the model to train on.

            The symptom this is happening is a continous average loss of 0.0.

            It seem that one has to explicitly add these to the data set when running .create_dataset(). However again this is not clear

            as the fucntion signature does not have keys for these values. Instead it assumes an order of SMILES, targets, weights, ids

            passed as an iterable. This is implemented in this function to make it easier to use and abstract the details of the interfaces

            of the deepchem library.

            Example:

            ```python

            import logging

            from redxregressors import deep_net_models

            dc_logger = logging.getLogger("deepchem")

            dc_logger.setLevel(logging.WARNING)

            data_df_bind = pd.read_csv("data/kinase_data_bind.csv")

            tasks = ["kinase_1_pIC50", "kinase_2_pIC50"]

            smiles_column = "smiles"

            ids_column = "name_id"

            reg, tr, test, val, test_set_metrics_df = deep_net_models.train_multitask_regressor(

                data_df=data_df_bind,

                tasks=tasks,

                smiles_column=smiles_column,

                ids_column=ids_column,

                epochs=200,

                layer_sizes=[1000, 1000],

                )

            ```

            Args:

                data_df (pd.DataFrame): the data frame containing the data

                tasks (List[str]): the names of the tasks to train on

                smiles_column (str): the name of the column containing the SMILES strings

                ids_column (Optional[str]): the name of the column containing the IDs

                task_weights (Optional[List[float]]): the weights for each task

                epochs (int): the number of epochs to train for

                learning_rate (float): the learning rate for the model

                layer_sizes (List[int]): the sizes of the layers in the model

                batch_size (int): the batch size for the model

                featurizer (Optional[dc.feat.Featurizer]): the featurizer to use

                splitter (Optional[dc.splits.Splitter]): the splitter to use

                unique_string (Optional[str]): a unique string to append to the output files

                pre_seed (bool): whether to pre seed the random number generators

                frac_train (float): the fraction of the data to use for training

                frac_valid (float): the fraction of the data to use for validation

                frac_test (float): the fraction of the data to use for testing

                uncertainty (bool): whether to use uncertainty in the model

                residual (bool): whether to use residual connections in the model

                weight_decay_penalty (float): the weight decay penalty to use

                weight_decay_penalty_type (str): the type of weight decay penalty to use

                dropouts (float | Sequence[float]): the dropouts to use

                activation_fns (Callable | str | Sequence[Callable | str]): the activation functions to use

                **kwargs: additional keyword arguments to pass to the model

            Returns:

                tuple[Any, Any, Any, Any, pd.DataFrame]: the trained model, the training data set, the validation data set, the test data set, tests set metrics predictions

            """

            # Pre seed the random number generators

            if pre_seed is True:

                seed_all()

                torch.use_deterministic_algorithms(True)

            # Get all of the pieces that form the data set as rew data

            if train_dataset is None and test_dataset is None and valid_dataset is None:

                log.debug(

                    "Building the data sets from pandas data frame input and column headers"

                )

                smiles = data_df[smiles_column].values

                if ids_column is not None:

                    ids = data_df[ids_column].values

                else:

                    ids = data_df.index.values

                if task_weights is None:

                    weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)

                else:

                    weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)

                    for indx, tw in enumerate(task_weights):

                        weights[:, indx] *= task_weights[indx]

                if len(tasks) == 1:

                    log.warning(

                        "Only one task was provided, this is not a multitask model consider a different model type"

                    )

                else:

                    targets = data_df[tasks].values

                # get the data sets for training, validation and testing

                train_dataset, valid_dataset, test_dataset, splitter = build_in_memory_loader(

                    tasks=tasks,

                    featurizer=featurizer,

                    ids_column=ids_column,

                    ids=ids,

                    smiles=smiles,

                    targets=targets,

                    weights=weights,

                    splitter=splitter,

                    frac_train=frac_train,

                    frac_valid=frac_valid,

                    frac_test=frac_test,

                )

            # if the data sets are provided then use them

            else:

                log.debug("Using the provided data sets")

                splitter = None

            log.debug(f"Training dataset shape: {train_dataset.get_data_shape()}")

            log.debug(train_dataset)

            log.debug(train_dataset.X)

            log.debug(train_dataset.y)

            # instantiate the model

            if fit_transformers is None:

                model = dc.models.MultitaskRegressor(

                    len(tasks),

                    train_dataset.get_data_shape()[0],

                    layer_sizes=layer_sizes,

                    batch_size=batch_size,

                    learning_rate=learning_rate,

                    weight_decay_penalty=weight_decay_penalty,

                    weight_decay_penalty_type=weight_decay_penalty_type,

                    dropouts=dropouts,

                    activation_fns=activation_fns,

                    uncertainty=uncertainty,

                    residual=residual,

                    **kwargs,

                )

            else:

                fit_transformers = [

                    ent(train_dataset, transform_X=True, transform_y=False)

                    for ent in fit_transformers

                ]

                log.debug(train_dataset.get_data_shape()[0])

                model = dc.models.MultitaskFitTransformRegressor(

                    len(tasks),

                    train_dataset.get_data_shape()[0],

                    layer_sizes=layer_sizes,

                    batch_size=batch_size,

                    learning_rate=learning_rate,

                    weight_decay_penalty=weight_decay_penalty,

                    weight_decay_penalty_type=weight_decay_penalty_type,

                    dropouts=dropouts,

                    activation_fns=activation_fns,

                    uncertainty=uncertainty,

                    residual=residual,

                    fit_transformers=fit_transformers,

                    **kwargs,

                )

                # train the model using an explicit loop to allow for intermediate evaluation

            model = fit_mtr_pytorch_model(

                model=model,

                train_dataset=train_dataset,

                valid_dataset=valid_dataset,

                epochs=epochs,

                unique_string=unique_string,

            )

            # evaluate the model using the test set

            test_set_df = evaluate_mtr_pytorch_model(

                model=model, test_dataset=test_dataset, tasks=tasks, unique_string=unique_string

            )

            return model, train_dataset, valid_dataset, test_dataset, test_set_df


### train_progressive_multitask_regressor

```python3
def train_progressive_multitask_regressor(
    data_df: pandas.core.frame.DataFrame,
    tasks: List[str],
    smiles_column: str = 'smiles',
    ids_column: Optional[str] = None,
    task_weights: Optional[List[float]] = None,
    epochs: int = 100,
    learning_rate: float = 0.0001,
    layer_sizes: List[int] = [1000, 1000],
    batch_size: int = 50,
    featurizer: Optional[deepchem.feat.base_classes.Featurizer] = CircularFingerprint[radius=2, size=1024, chiral=True, bonds=True, features=False, sparse=False, smiles=False, is_counts_based=False],
    splitter: Optional[deepchem.splits.splitters.Splitter] = RandomSplitter[],
    unique_string: Optional[str] = None,
    pre_seed: bool = True,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    uncertainty: bool = False,
    residual: bool = False,
    weight_decay_penalty: float = 0.0,
    weight_decay_penalty_type: str = 'l2',
    dropouts: Union[float, Sequence[float]] = 0.5,
    activation_fns: Union[Callable, str, Sequence[Union[Callable, str]]] = 'relu',
    train_per_task: bool = False,
    **kwargs
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any, pandas.core.frame.DataFrame]
```

Function to train a multitask regressor using the deepchem library.

Note the deepchem library is a wrapper around pytorch, Jax, tensorflow and keras.

The examples online are unclear for the use of this model so this function is a wrapper to make it easier to use.
The main difficulty is building the data set. By default the dispite naming the tasks when you define teh loader
object it does not by defult pull these out as the tasks. In addition, the weights in the data set for each data
point default to 0.0. Both of these mean that there are no target values for the model to train on.
The symptom this is happening is a continous average loss of 0.0.

It seem that one has to explicitly add these to the data set when running .create_dataset(). However again this is not clear
as the fucntion signature does not have keys for these values. Instead it assumes an order of SMILES, targets, weights, ids
passed as an iterable. This is implemented in this function to make it easier to use and abstract the details of the interfaces
of the deepchem library.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data_df | pd.DataFrame | the data frame containing the data | None |
| tasks | List[str] | the names of the tasks to train on | None |
| smiles_column | str | the name of the column containing the SMILES strings | None |
| ids_column | Optional[str] | the name of the column containing the IDs | None |
| task_weights | Optional[List[float]] | the weights for each task | None |
| epochs | int | the number of epochs to train for | None |
| learning_rate | float | the learning rate for the model | None |
| layer_sizes | List[int] | the sizes of the layers in the model | None |
| batch_size | int | the batch size for the model | None |
| featurizer | Optional[dc.feat.Featurizer] | the featurizer to use | None |
| splitter | Optional[dc.splits.Splitter] | the splitter to use | None |
| unique_string | Optional[str] | a unique string to append to the output files | None |
| pre_seed | bool | whether to pre seed the random number generators | None |
| frac_train | float | the fraction of the data to use for training | None |
| frac_valid | float | the fraction of the data to use for validation | None |
| frac_test | float | the fraction of the data to use for testing | None |
| uncertainty | bool | whether to use uncertainty in the model | None |
| residual | bool | whether to use residual connections in the model | None |
| weight_decay_penalty | float | the weight decay penalty to use | None |
| weight_decay_penalty_type | str | the type of weight decay penalty to use | None |
| dropouts | float | Sequence[float] | the dropouts to use | None |
| activation_fns | Callable | str | Sequence[Callable | str] | the activation functions to use | None |
| train_per_task | bool | whether to train the model per task sequentially | None |
| **kwargs | None | additional keyword arguments to pass to the model | None |

**Returns:**

| Type | Description |
|---|---|
| tuple[Any, Any, Any, Any, pd.DataFrame] | the trained model, the training data set, the validation data set, the test data set, tests set metrics predictions |

??? example "View Source"
        def train_progressive_multitask_regressor(

            data_df: pd.DataFrame,

            tasks: List[str],

            smiles_column: str = "smiles",

            ids_column: Optional[str] = None,

            task_weights: Optional[List[float]] = None,

            epochs: int = 100,

            learning_rate: float = 0.0001,

            layer_sizes: List[int] = [1000, 1000],

            batch_size: int = 50,

            featurizer: Optional[dc.feat.Featurizer] = dc.feat.CircularFingerprint(

                size=1024, radius=2, chiral=True, bonds=True

            ),

            splitter: Optional[dc.splits.Splitter] = dc.splits.RandomSplitter(),

            unique_string: Optional[str] = None,

            pre_seed: bool = True,

            frac_train: float = 0.8,

            frac_valid: float = 0.1,

            frac_test: float = 0.1,

            uncertainty: bool = False,

            residual: bool = False,

            weight_decay_penalty: float = 0.0,

            weight_decay_penalty_type: str = "l2",

            dropouts: float | Sequence[float] = 0.5,

            activation_fns: Callable | str | Sequence[Callable | str] = "relu",

            train_per_task: bool = False,

            **kwargs,

        ) -> tuple[Any, Any, Any, Any, pd.DataFrame]:

            """

            Function to train a multitask regressor using the deepchem library.

            Note the deepchem library is a wrapper around pytorch, Jax, tensorflow and keras.

            The examples online are unclear for the use of this model so this function is a wrapper to make it easier to use.

            The main difficulty is building the data set. By default the dispite naming the tasks when you define teh loader

            object it does not by defult pull these out as the tasks. In addition, the weights in the data set for each data

            point default to 0.0. Both of these mean that there are no target values for the model to train on.

            The symptom this is happening is a continous average loss of 0.0.

            It seem that one has to explicitly add these to the data set when running .create_dataset(). However again this is not clear

            as the fucntion signature does not have keys for these values. Instead it assumes an order of SMILES, targets, weights, ids

            passed as an iterable. This is implemented in this function to make it easier to use and abstract the details of the interfaces

            of the deepchem library.

            Example:

            ```python

            import logging

            from redxregressors import deep_net_models

            dc_logger = logging.getLogger("deepchem")

            dc_logger.setLevel(logging.WARNING)

            data_df_bind = pd.read_csv("data/kinase_data_bind.csv")

            tasks = ["kinase_1_pIC50", "kinase_2_pIC50"]

            smiles_column = "smiles"

            ids_column = "name_id"

            reg, tr, test, val, test_set_metrics_df = deep_net_models.train_multitask_regressor(

                data_df=data_df_bind,

                tasks=tasks,

                smiles_column=smiles_column,

                ids_column=ids_column,

                epochs=200,

                layer_sizes=[1000, 1000],

                )

            ```

            Args:

                data_df (pd.DataFrame): the data frame containing the data

                tasks (List[str]): the names of the tasks to train on

                smiles_column (str): the name of the column containing the SMILES strings

                ids_column (Optional[str]): the name of the column containing the IDs

                task_weights (Optional[List[float]]): the weights for each task

                epochs (int): the number of epochs to train for

                learning_rate (float): the learning rate for the model

                layer_sizes (List[int]): the sizes of the layers in the model

                batch_size (int): the batch size for the model

                featurizer (Optional[dc.feat.Featurizer]): the featurizer to use

                splitter (Optional[dc.splits.Splitter]): the splitter to use

                unique_string (Optional[str]): a unique string to append to the output files

                pre_seed (bool): whether to pre seed the random number generators

                frac_train (float): the fraction of the data to use for training

                frac_valid (float): the fraction of the data to use for validation

                frac_test (float): the fraction of the data to use for testing

                uncertainty (bool): whether to use uncertainty in the model

                residual (bool): whether to use residual connections in the model

                weight_decay_penalty (float): the weight decay penalty to use

                weight_decay_penalty_type (str): the type of weight decay penalty to use

                dropouts (float | Sequence[float]): the dropouts to use

                activation_fns (Callable | str | Sequence[Callable | str]): the activation functions to use

                train_per_task (bool): whether to train the model per task sequentially

                **kwargs: additional keyword arguments to pass to the model

            Returns:

                tuple[Any, Any, Any, Any, pd.DataFrame]: the trained model, the training data set, the validation data set, the test data set, tests set metrics predictions

            """

            # Pre seed the random number generators

            if pre_seed is True:

                seed_all()

                torch.use_deterministic_algorithms(True)

            # Get all of the pieces that form the data set as rew data

            smiles = data_df[smiles_column].values

            if ids_column is not None:

                ids = data_df[ids_column].values

            else:

                ids = data_df.index.values

            if task_weights is None:

                weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)

            else:

                weights = np.ones((len(smiles), len(tasks)), dtype=np.float16)

                for indx, tw in enumerate(task_weights):

                    weights[:, indx] *= task_weights[indx]

            if len(tasks) == 1:

                log.warning(

                    "Only one task was provided, this is not a multitask model consider a different model type"

                )

            else:

                targets = data_df[tasks].values

            # get the data sets for training, validation and testing

            train_dataset, valid_dataset, test_dataset, splitter = build_in_memory_loader(

                tasks=tasks,

                featurizer=featurizer,

                ids_column=ids_column,

                ids=ids,

                smiles=smiles,

                targets=targets,

                weights=weights,

                splitter=splitter,

                frac_train=frac_train,

                frac_valid=frac_valid,

                frac_test=frac_test,

            )

            log.debug(f"Training dataset shape: {train_dataset.get_data_shape()}")

            log.debug(train_dataset)

            log.debug(train_dataset.X)

            log.debug(train_dataset.y)

            # instantiate the model

            model = dc.models.torch_models.ProgressiveMultitaskModel(

                len(tasks),

                train_dataset.get_data_shape()[0],

                mode="regression",

                layer_sizes=layer_sizes,

                batch_size=batch_size,

                learning_rate=learning_rate,

                weight_decay_penalty=weight_decay_penalty,

                weight_decay_penalty_type=weight_decay_penalty_type,

                dropouts=dropouts,

                activation_fns=activation_fns,

                uncertainty=uncertainty,

                residual=residual,

                **kwargs,

            )

            # train the model using an explicit loop to allow for intermediate evaluation

            if train_per_task is False:

                model = fit_mtr_pytorch_model(

                    model=model,

                    train_dataset=train_dataset,

                    valid_dataset=valid_dataset,

                    epochs=epochs,

                    unique_string=unique_string,

                )

            else:

                fit_mtr_pytorch_model_per_task(

                    model=model,

                    train_dataset=train_dataset,

                    valid_dataset=valid_dataset,

                    epochs=epochs,

                    unique_string=unique_string,

                )

            # evaluate the model using the test set

            test_set_df = evaluate_mtr_pytorch_model(

                model=model, test_dataset=test_dataset, tasks=tasks, unique_string=unique_string

            )

            return model, train_dataset, valid_dataset, test_dataset, test_set_df
