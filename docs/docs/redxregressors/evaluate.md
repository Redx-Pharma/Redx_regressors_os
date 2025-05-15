# Module redxregressors.evaluate

Module for evaluating and plotting the results of models

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module for evaluating and plotting the results of models

        """

        import logging

        from typing import Union, Tuple, List, Optional, Any

        import pandas as pd

        import numpy as np

        from sklearn.metrics import (

            mean_squared_error,

            r2_score,

            mean_absolute_error,

            mean_absolute_percentage_error,

            root_mean_squared_error,

        )

        import matplotlib.pyplot as plt

        from matplotlib.pyplot import cm

        import scipy

        import os

        import plotly.express as px

        import seaborn as sns

        log = logging.getLogger(__name__)



        def rmse(exp, pred) -> np.ndarray[Any, np.dtype[Any]]:

            """

            Function to calculate the Root Mean Squared Error between two arrays

            Args:

                exp (np.ndarray): the expected values

                pred (np.ndarray): the predicted values

            Returns:

                np.ndarray: the root mean squared error

            """

            return np.sqrt(mean_squared_error(exp, pred))



        def calculate_regression_metrics(

            exp_array: np.ndarray, pred_array: np.ndarray, return_tuple: bool = False

        ) -> Union[dict, Tuple[float, float, float, float, float]]:

            """

            Function to calculate the regression metrics between two arrays

            Args:

                exp_array (np.ndarray): the expected values

                pred_array (np.ndarray): the predicted values

                return_tuple (bool): whether to return a tuple or a dictionary

            Returns:

                Union[dict, Tuple[float, float, float, float, float]]: the regression metrics

            """

            mse = mean_squared_error(exp_array, pred_array)

            rmse = np.sqrt(mean_squared_error(exp_array, pred_array))

            mae = mean_absolute_error(exp_array, pred_array)

            r2 = r2_score(exp_array, pred_array)

            mape = mean_absolute_percentage_error(exp_array, pred_array)

            if return_tuple is False:

                return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

            else:

                return mse, rmse, mae, r2, mape



        def get_regression_metric_table(

            df: Union[pd.DataFrame, None] = None,

            exp_column: str = "y",

            pred_column: str = "y_pred",

            exp_array: Union[np.ndarray, None] = None,

            pred_array: Union[np.ndarray, None] = None,

            filename: Optional[str] = None,

        ) -> pd.DataFrame:

            """

            Function to calculate the regression metrics between two arrays and return a dataframe. If a filename is provided, the dataframe is saved to a csv file.

            Args:

                df (Union[pd.DataFrame, None]): the dataframe containing the experimental and predicted columns

                exp_column (str): the column name of the experimental values

                pred_column (str): the column name of the predicted values

                exp_array (Union[np.ndarray, None]): the experimental values

                pred_array (Union[np.ndarray, None]): the predicted values

                filename (Optional[str]): the filename to save the dataframe to

            Returns:

                pd.DataFrame: the regression metrics dataframe

            """

            if df is not None:

                # Copy to avoid changing the original dataframe

                exp_array = df[exp_column].values.copy()

                pred_array = df[pred_column].values.copy()

            reg_metrics = calculate_regression_metrics(exp_array, pred_array)

            reg_metrics_df = pd.DataFrame(reg_metrics, index=[0])

            if filename is not None:

                reg_metrics_df.to_csv(filename, index=False)

            return reg_metrics_df



        def plot_png_parity_plot(

            df: Union[pd.DataFrame, None] = None,

            exp_column: str = "y",

            pred_column: str = "y_pred",

            label_column: Optional[str] = None,

            exp_array: Union[np.ndarray, None] = None,

            pred_array: Union[np.ndarray, None] = None,

            label_list: Optional[List[str]] = None,

            task_name: str = "model predictions",

            offset: float = 0.1,

            no_save: bool = False,

        ) -> plt.Axes | None:

            """

            Function to plot a parity plot between the experimental and predicted values and save it to a png file

            Args:

                df (Union[pd.DataFrame, None]): the dataframe containing the experimental and predicted columns

                exp_column (str): the column name of the experimental values

                pred_column (str): the column name of the predicted values

                label_column (Optional[str]): the column name of the labels

                exp_array (Union[np.ndarray, None]): the experimental values

                pred_array (Union[np.ndarray, None]): the predicted values

                label_list (Optional[List[str]]): the list of labels

                task_name (str): the name of the task

                offset (float): the offset for the labels

                no_save (bool): whether to save the plot

            Returns:

                None | plt.Axes: the plot

            """

            if df is not None:

                # Copy to avoid changing the original dataframe

                exp_array = df[exp_column].values.copy()

                pred_array = df[pred_column].values.copy()

                if label_column is not None:

                    label_list = df[label_column].values.copy()

            _, rmse, mae, r2, _ = calculate_regression_metrics(

                exp_array, pred_array, return_tuple=True

            )

            fig = plt.figure(figsize=(10, 10))

            xymin = np.floor(min(exp_array.tolist() + pred_array.tolist()))

            xymax = np.ceil(max(exp_array.tolist() + pred_array.tolist()))

            log.debug(f"min {xymin}, max {xymax}")

            ticks = np.arange(xymin, xymax + 1, 1.0)

            # plt.scatter(test.y.ravel(), test.y.ravel(), label=f"Prefect Prediction", c="#89a0b0")

            plt.scatter(

                exp_array,

                pred_array,

                label=f"{task_name} $R^{2}$ {r2:.2f} RMSE {rmse:.2f} MAE {mae:.2f}",

                c="b",

            )

            plt.plot([xymin, xymax], [xymin, xymax], "k--", label="x = y")

            plt.grid()

            plt.legend()

            plt.xlabel(f"{task_name} Experimental", fontsize=25)

            plt.ylabel(f"{task_name} Prediction", fontsize=25)

            plt.title(f"{task_name} Vs. Prediction", fontsize=27)

            ax = plt.gca()

            ax.tick_params(axis="both", which="major", labelsize=17)

            ax.set_yticks(ticks)

            ax.set_xticks(ticks)

            if label_list is not None:

                try:

                    from adjustText import adjust_text

                    ax_tmp = [

                        ax.annotate(

                            labels,

                            (exp_array[ith], pred_array[ith]),

                            xytext=(exp_array[ith] + offset, pred_array[ith] + offset),

                            arrowprops=dict(arrowstyle="->", color="k"),

                        )

                        for ith, labels in enumerate(label_list)

                    ]

                    adjust_text(ax_tmp)

                except ImportError:

                    log.warning(

                        "Please install adjustText to avoid overlapping labels in the plot"

                    )

                    for ith, labels in enumerate(label_list):

                        ax.annotate(

                            labels,

                            (exp_array[ith], pred_array[ith]),

                            xytext=(exp_array[ith] + offset, pred_array[ith] + offset),

                            arrowprops=dict(arrowstyle="->", color="k"),

                        )

            plt.tight_layout()

            if no_save is False:

                plt.savefig(

                    f"evaluation_{'_'.join(task_name.replace(f'{os.sep}', '_').split())}.png"

                )

                plt.close(fig)

            else:

                return plt.gca()



        def plot_png_parity_plot_coloured_by_catagorical_column(

            df: Union[pd.DataFrame, None] = None,

            exp_column: str = "y",

            pred_column: str = "y_pred",

            label_column: Optional[List[str]] = None,

            cat_column: str = "series",

            exp_array: Union[np.ndarray, None] = None,

            pred_array: Union[np.ndarray, None] = None,

            cat_array: Union[np.ndarray, None] = None,

            label_list: Optional[List[str]] = None,

            task_name: str = "model predictions",

            offset: float = 0.1,

            no_save: bool = False,

        ) -> plt.Axes | None:

            """

            Function to plot a parity plot between the experimental and predicted values and save it to a png file

            Args:

                df (Union[pd.DataFrame, None]): the dataframe containing the experimental and predicted columns

                exp_column (str): the column name of the experimental values

                pred_column (str): the column name of the predicted values

                label_column (Optional[str]): the column name of the labels

                exp_array (Union[np.ndarray, None]): the experimental values

                pred_array (Union[np.ndarray, None]): the predicted values

                cat_column (str): the column name of the catagorical values

                label_list (Optional[List[str]]): the list of labels

                task_name (str): the name of the task

                offset (float): the offset for the labels

                no_save (bool): whether to save the plot

            Returns:

                plt.Axes: the plot | None

            """

            if df is not None:

                # Copy to avoid changing the original dataframe

                exp_array = df[exp_column].values.copy()

                pred_array = df[pred_column].values.copy()

                cat_array = df[cat_column].values.copy()

                if label_column is not None:

                    label_list = df[label_column].values.copy()

            cat_values = set(cat_array)

            fig = plt.figure(figsize=(10, 10))

            xymin = np.floor(min(exp_array.tolist() + pred_array.tolist()))

            xymax = np.ceil(max(exp_array.tolist() + pred_array.tolist()))

            log.debug(f"min {xymin}, max {xymax}")

            ticks = np.arange(xymin, xymax + 1, 1.0)

            colours = iter(cm.rainbow(np.linspace(0, 1, len(cat_values))))

            for cat in cat_values:

                indexes = np.where(cat_array == cat)

                exp = exp_array[indexes]

                pred = pred_array[indexes]

                _, rmse, mae, r2, _ = calculate_regression_metrics(exp, pred, return_tuple=True)

                plt.scatter(

                    exp,

                    pred,

                    label=f"{task_name} {cat} $R^{2}$ {r2:.2f} RMSE {rmse:.2f} MAE {mae:.2f}",

                    c=next(colours),

                )

            plt.plot([xymin, xymax], [xymin, xymax], "k--", label="x = y")

            plt.grid()

            plt.legend()

            plt.xlabel(f"{task_name} Experimental", fontsize=25)

            plt.ylabel(f"{task_name} Prediction", fontsize=25)

            plt.title(f"{task_name} Vs. Prediction", fontsize=27)

            ax = plt.gca()

            ax.tick_params(axis="both", which="major", labelsize=17)

            if label_list is not None:

                try:

                    from adjustText import adjust_text

                    ax_tmp = [

                        ax.annotate(

                            labels,

                            (exp_array[ith], pred_array[ith]),

                            xytext=(exp_array[ith] + offset, pred_array[ith] + offset),

                            arrowprops=dict(arrowstyle="->", color="k"),

                        )

                        for ith, labels in enumerate(label_list)

                    ]

                    adjust_text(ax_tmp)

                except ImportError:

                    log.warning(

                        "Please install adjustText to avoid overlapping labels in the plot"

                    )

                    for ith, labels in enumerate(label_list):

                        ax.annotate(

                            labels,

                            (exp_array[ith], pred_array[ith]),

                            xytext=(exp_array[ith] + offset, pred_array[ith] + offset),

                            arrowprops=dict(arrowstyle="->", color="k"),

                        )

            ax.set_yticks(ticks)

            ax.set_xticks(ticks)

            plt.tight_layout()

            if no_save is False:

                plt.savefig(

                    f"evaluation_{'_'.join(task_name.replace(f'{os.sep}', '_').split())}.png"

                )

                plt.close(fig)

            else:

                return plt.gca()



        def interactive_parity_plot(

            df: Union[pd.DataFrame, None] = None,

            exp_column: str = "y",

            pred_column: str = "y_pred",

            structure_column: str = "smiles",

            label_column: Optional[List[str]] = None,

        ):

            """

            Function to generate an interactive parity plot using Plotly Express

            Args:

                df (Union[pd.DataFrame, None]): the dataframe containing the experimental and predicted columns

                exp_column (str): the column name of the experimental values

                pred_column (str): the column name of the predicted values

                structure_column (str): the column name of the SMILES

                label_column (Optional[List[str]]): the list of labels

            Returns:

                plotly.graph_objects.Figure: the interactive parity plot

            """

            # generate a scatter plot

            fig = px.scatter(

                df,

                x=exp_column,

                y=pred_column,

                trendline="ols",

                hover_data=[structure_column, label_column],

            )

            fig.show()

            return fig

            # add molecules to the plotly graph - returns a Dash app

            # app = molplotly.add_molecules(fig=fig,

            #                             df=df,

            #                             smiles_col=structure_column,

            #                             title_col=label_column,

            #                             )

            # # run Dash app inline in notebook (or in an external server)

            # app.run(jupyter_mode='inline', debug=True, jupyter_height=1000, jupyter_width=1000) # port=8700,



        def parity_plot(

            y_test,

            y_pred,

            xymin: Optional[float] = None,

            xymax: Optional[float] = None,

            style: str = "seaborn-v0_8-dark-palette",

            size: Tuple[float, float] = (10.0, 10.0),

            title_fontsize: int = 27,

            filename: Optional[str] = "pariry_plot.png",

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

            """

            # get the min and max values for the x and y axis if not input

            if xymin is None:

                xymin = np.floor(min(y_test.tolist() + y_pred.tolist()))

            if xymax is None:

                xymax = np.ceil(max(y_test.tolist() + y_pred.tolist()))

            log.debug(f"min {xymin}, max {xymax}")

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

                ax.tick_params(axis="both", which="major", labelsize=max(title_fontsize - 2, 7))

                ax.set_yticks(ticks)

                ax.set_xticks(ticks)

                plt.tight_layout()

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig



        def plot_residuals(

            y_test,

            y_pred,

            style="seaborn-v0_8-dark-palette",

            size=(10, 10),

            title_fontsize: int = 27,

            filename: Optional[str] = "residuals_plot.png",

        ) -> plt.Figure:

            """

            Function to plot the residuals of the test set predictions

            Args:

                y_test (np.ndarray): The test set target values

                y_pred (np.ndarray): The predicted values

                size (Tuple[int, int]): The size of the plot

                title_fontsize (int): The fontsize of the title

            Returns:

                plt.figure: The residuals plot

            """

            with plt.style.context(style=style):

                fig, ax = plt.subplots(figsize=size)

                df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})

                sns.residplot(

                    df,

                    x="y_pred",

                    y="y_test",

                    lowess=True,

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

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig



        def plot_prediction_error(

            y_test,

            y_pred,

            style="seaborn-v0_8-dark-palette",

            size=(10, 10),

            title_fontsize: int = 27,

            filename: Optional[str] = "prediction_error_plot.png",

        ) -> plt.Figure:

            """

            Function to plot the prediction error plot of the test set predictions

            Args:

                y_test (np.ndarray): The test set target values

                y_pred (np.ndarray): The predicted values

                style (str): The style of the plot

                size (Tuple[int, int]): The size of the plot

                title_fontsize (int): The fontsize of the title

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

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig



        def plot_qq(

            y_test,

            y_pred,

            style="seaborn-v0_8-dark-palette",

            size=(10, 10),

            title_fontsize: int = 27,

            filename: Optional[str] = "qq_plot.png",

        ) -> plt.Figure:

            """

            Function to plot the QQ plot of the residuals of the test set predictions

            Args:

                y_test (np.ndarray): The test set target values

                y_pred (np.ndarray): The predicted values

                style (str): The style of the plot

                size (Tuple[int, int]): The size of the plot

                title_fontsize (int): The fontsize of the title

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

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig



        def plot_metric_curves(

            metrics: list,

            metric_labels: list,

            x: Optional[List[int]] = None,

            size: Tuple[int, int] = (10, 10),

            style="seaborn-v0_8-dark-palette",

            title_fontsize: int = 27,

            filename: Optional[str] = None,

        ) -> plt.Figure:

            """

            Function to plot the metric curves of the model training

            Args:

                metrics (list): The list of metrics

                metric_labels (list): The list of metric labels

                size (Tuple[int, int]): The size of the plot

                style (str): The style of the plot

                title_fontsize (int): The fontsize of the title

                filename (Optional[str]): The filename to save the plot to

            Returns:

                plt.Figure: The metric curves plot

            """

            with plt.style.context(style=style):

                fig = plt.figure(figsize=size)

                colours = iter(cm.rainbow(np.linspace(0, 1, len(metrics))))

                if x is None:

                    x = [ith for ith in range(len(metrics[0]))]

                log.debug(f"x: {x}")

                for metric, label in zip(metrics, metric_labels):

                    plt.plot(x, metric, "o-", label=label, linewidth=2, color=next(colours))

                plt.grid()

                plt.legend()

                plt.xlabel("Epochs", fontsize=max(title_fontsize - 2, 10))

                plt.ylabel("Metric Value", fontsize=max(title_fontsize - 2, 10))

                plt.title("Model Training Metric Curves", fontsize=title_fontsize)

                ax = plt.gca()

                ax.tick_params(

                    axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                )

                plt.tight_layout()

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### calculate_regression_metrics

```python3
def calculate_regression_metrics(
    exp_array: numpy.ndarray,
    pred_array: numpy.ndarray,
    return_tuple: bool = False
) -> Union[dict, Tuple[float, float, float, float, float]]
```

Function to calculate the regression metrics between two arrays

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| exp_array | np.ndarray | the expected values | None |
| pred_array | np.ndarray | the predicted values | None |
| return_tuple | bool | whether to return a tuple or a dictionary | None |

**Returns:**

| Type | Description |
|---|---|
| Union[dict, Tuple[float, float, float, float, float]] | the regression metrics |

??? example "View Source"
        def calculate_regression_metrics(

            exp_array: np.ndarray, pred_array: np.ndarray, return_tuple: bool = False

        ) -> Union[dict, Tuple[float, float, float, float, float]]:

            """

            Function to calculate the regression metrics between two arrays

            Args:

                exp_array (np.ndarray): the expected values

                pred_array (np.ndarray): the predicted values

                return_tuple (bool): whether to return a tuple or a dictionary

            Returns:

                Union[dict, Tuple[float, float, float, float, float]]: the regression metrics

            """

            mse = mean_squared_error(exp_array, pred_array)

            rmse = np.sqrt(mean_squared_error(exp_array, pred_array))

            mae = mean_absolute_error(exp_array, pred_array)

            r2 = r2_score(exp_array, pred_array)

            mape = mean_absolute_percentage_error(exp_array, pred_array)

            if return_tuple is False:

                return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

            else:

                return mse, rmse, mae, r2, mape


### get_regression_metric_table

```python3
def get_regression_metric_table(
    df: Optional[pandas.core.frame.DataFrame] = None,
    exp_column: str = 'y',
    pred_column: str = 'y_pred',
    exp_array: Optional[numpy.ndarray] = None,
    pred_array: Optional[numpy.ndarray] = None,
    filename: Optional[str] = None
) -> pandas.core.frame.DataFrame
```

Function to calculate the regression metrics between two arrays and return a dataframe. If a filename is provided, the dataframe is saved to a csv file.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | Union[pd.DataFrame, None] | the dataframe containing the experimental and predicted columns | None |
| exp_column | str | the column name of the experimental values | None |
| pred_column | str | the column name of the predicted values | None |
| exp_array | Union[np.ndarray, None] | the experimental values | None |
| pred_array | Union[np.ndarray, None] | the predicted values | None |
| filename | Optional[str] | the filename to save the dataframe to | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the regression metrics dataframe |

??? example "View Source"
        def get_regression_metric_table(

            df: Union[pd.DataFrame, None] = None,

            exp_column: str = "y",

            pred_column: str = "y_pred",

            exp_array: Union[np.ndarray, None] = None,

            pred_array: Union[np.ndarray, None] = None,

            filename: Optional[str] = None,

        ) -> pd.DataFrame:

            """

            Function to calculate the regression metrics between two arrays and return a dataframe. If a filename is provided, the dataframe is saved to a csv file.

            Args:

                df (Union[pd.DataFrame, None]): the dataframe containing the experimental and predicted columns

                exp_column (str): the column name of the experimental values

                pred_column (str): the column name of the predicted values

                exp_array (Union[np.ndarray, None]): the experimental values

                pred_array (Union[np.ndarray, None]): the predicted values

                filename (Optional[str]): the filename to save the dataframe to

            Returns:

                pd.DataFrame: the regression metrics dataframe

            """

            if df is not None:

                # Copy to avoid changing the original dataframe

                exp_array = df[exp_column].values.copy()

                pred_array = df[pred_column].values.copy()

            reg_metrics = calculate_regression_metrics(exp_array, pred_array)

            reg_metrics_df = pd.DataFrame(reg_metrics, index=[0])

            if filename is not None:

                reg_metrics_df.to_csv(filename, index=False)

            return reg_metrics_df


### interactive_parity_plot

```python3
def interactive_parity_plot(
    df: Optional[pandas.core.frame.DataFrame] = None,
    exp_column: str = 'y',
    pred_column: str = 'y_pred',
    structure_column: str = 'smiles',
    label_column: Optional[List[str]] = None
)
```

Function to generate an interactive parity plot using Plotly Express

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | Union[pd.DataFrame, None] | the dataframe containing the experimental and predicted columns | None |
| exp_column | str | the column name of the experimental values | None |
| pred_column | str | the column name of the predicted values | None |
| structure_column | str | the column name of the SMILES | None |
| label_column | Optional[List[str]] | the list of labels | None |

**Returns:**

| Type | Description |
|---|---|
| plotly.graph_objects.Figure | the interactive parity plot |

??? example "View Source"
        def interactive_parity_plot(

            df: Union[pd.DataFrame, None] = None,

            exp_column: str = "y",

            pred_column: str = "y_pred",

            structure_column: str = "smiles",

            label_column: Optional[List[str]] = None,

        ):

            """

            Function to generate an interactive parity plot using Plotly Express

            Args:

                df (Union[pd.DataFrame, None]): the dataframe containing the experimental and predicted columns

                exp_column (str): the column name of the experimental values

                pred_column (str): the column name of the predicted values

                structure_column (str): the column name of the SMILES

                label_column (Optional[List[str]]): the list of labels

            Returns:

                plotly.graph_objects.Figure: the interactive parity plot

            """

            # generate a scatter plot

            fig = px.scatter(

                df,

                x=exp_column,

                y=pred_column,

                trendline="ols",

                hover_data=[structure_column, label_column],

            )

            fig.show()

            return fig

            # add molecules to the plotly graph - returns a Dash app

            # app = molplotly.add_molecules(fig=fig,

            #                             df=df,

            #                             smiles_col=structure_column,

            #                             title_col=label_column,

            #                             )

            # # run Dash app inline in notebook (or in an external server)

            # app.run(jupyter_mode='inline', debug=True, jupyter_height=1000, jupyter_width=1000) # port=8700,


### parity_plot

```python3
def parity_plot(
    y_test,
    y_pred,
    xymin: Optional[float] = None,
    xymax: Optional[float] = None,
    style: str = 'seaborn-v0_8-dark-palette',
    size: Tuple[float, float] = (10.0, 10.0),
    title_fontsize: int = 27,
    filename: Optional[str] = 'pariry_plot.png'
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

??? example "View Source"
        def parity_plot(

            y_test,

            y_pred,

            xymin: Optional[float] = None,

            xymax: Optional[float] = None,

            style: str = "seaborn-v0_8-dark-palette",

            size: Tuple[float, float] = (10.0, 10.0),

            title_fontsize: int = 27,

            filename: Optional[str] = "pariry_plot.png",

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

            """

            # get the min and max values for the x and y axis if not input

            if xymin is None:

                xymin = np.floor(min(y_test.tolist() + y_pred.tolist()))

            if xymax is None:

                xymax = np.ceil(max(y_test.tolist() + y_pred.tolist()))

            log.debug(f"min {xymin}, max {xymax}")

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

                ax.tick_params(axis="both", which="major", labelsize=max(title_fontsize - 2, 7))

                ax.set_yticks(ticks)

                ax.set_xticks(ticks)

                plt.tight_layout()

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig


### plot_metric_curves

```python3
def plot_metric_curves(
    metrics: list,
    metric_labels: list,
    x: Optional[List[int]] = None,
    size: Tuple[int, int] = (10, 10),
    style='seaborn-v0_8-dark-palette',
    title_fontsize: int = 27,
    filename: Optional[str] = None
) -> matplotlib.figure.Figure
```

Function to plot the metric curves of the model training

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| metrics | list | The list of metrics | None |
| metric_labels | list | The list of metric labels | None |
| size | Tuple[int, int] | The size of the plot | None |
| style | str | The style of the plot | None |
| title_fontsize | int | The fontsize of the title | None |
| filename | Optional[str] | The filename to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The metric curves plot |

??? example "View Source"
        def plot_metric_curves(

            metrics: list,

            metric_labels: list,

            x: Optional[List[int]] = None,

            size: Tuple[int, int] = (10, 10),

            style="seaborn-v0_8-dark-palette",

            title_fontsize: int = 27,

            filename: Optional[str] = None,

        ) -> plt.Figure:

            """

            Function to plot the metric curves of the model training

            Args:

                metrics (list): The list of metrics

                metric_labels (list): The list of metric labels

                size (Tuple[int, int]): The size of the plot

                style (str): The style of the plot

                title_fontsize (int): The fontsize of the title

                filename (Optional[str]): The filename to save the plot to

            Returns:

                plt.Figure: The metric curves plot

            """

            with plt.style.context(style=style):

                fig = plt.figure(figsize=size)

                colours = iter(cm.rainbow(np.linspace(0, 1, len(metrics))))

                if x is None:

                    x = [ith for ith in range(len(metrics[0]))]

                log.debug(f"x: {x}")

                for metric, label in zip(metrics, metric_labels):

                    plt.plot(x, metric, "o-", label=label, linewidth=2, color=next(colours))

                plt.grid()

                plt.legend()

                plt.xlabel("Epochs", fontsize=max(title_fontsize - 2, 10))

                plt.ylabel("Metric Value", fontsize=max(title_fontsize - 2, 10))

                plt.title("Model Training Metric Curves", fontsize=title_fontsize)

                ax = plt.gca()

                ax.tick_params(

                    axis="both", which="major", labelsize=max(title_fontsize - 10, 7)

                )

                plt.tight_layout()

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig


### plot_png_parity_plot

```python3
def plot_png_parity_plot(
    df: Optional[pandas.core.frame.DataFrame] = None,
    exp_column: str = 'y',
    pred_column: str = 'y_pred',
    label_column: Optional[str] = None,
    exp_array: Optional[numpy.ndarray] = None,
    pred_array: Optional[numpy.ndarray] = None,
    label_list: Optional[List[str]] = None,
    task_name: str = 'model predictions',
    offset: float = 0.1,
    no_save: bool = False
) -> matplotlib.axes._axes.Axes | None
```

Function to plot a parity plot between the experimental and predicted values and save it to a png file

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | Union[pd.DataFrame, None] | the dataframe containing the experimental and predicted columns | None |
| exp_column | str | the column name of the experimental values | None |
| pred_column | str | the column name of the predicted values | None |
| label_column | Optional[str] | the column name of the labels | None |
| exp_array | Union[np.ndarray, None] | the experimental values | None |
| pred_array | Union[np.ndarray, None] | the predicted values | None |
| label_list | Optional[List[str]] | the list of labels | None |
| task_name | str | the name of the task | None |
| offset | float | the offset for the labels | None |
| no_save | bool | whether to save the plot | None |

**Returns:**

| Type | Description |
|---|---|
| None | None | plt.Axes: the plot |

??? example "View Source"
        def plot_png_parity_plot(

            df: Union[pd.DataFrame, None] = None,

            exp_column: str = "y",

            pred_column: str = "y_pred",

            label_column: Optional[str] = None,

            exp_array: Union[np.ndarray, None] = None,

            pred_array: Union[np.ndarray, None] = None,

            label_list: Optional[List[str]] = None,

            task_name: str = "model predictions",

            offset: float = 0.1,

            no_save: bool = False,

        ) -> plt.Axes | None:

            """

            Function to plot a parity plot between the experimental and predicted values and save it to a png file

            Args:

                df (Union[pd.DataFrame, None]): the dataframe containing the experimental and predicted columns

                exp_column (str): the column name of the experimental values

                pred_column (str): the column name of the predicted values

                label_column (Optional[str]): the column name of the labels

                exp_array (Union[np.ndarray, None]): the experimental values

                pred_array (Union[np.ndarray, None]): the predicted values

                label_list (Optional[List[str]]): the list of labels

                task_name (str): the name of the task

                offset (float): the offset for the labels

                no_save (bool): whether to save the plot

            Returns:

                None | plt.Axes: the plot

            """

            if df is not None:

                # Copy to avoid changing the original dataframe

                exp_array = df[exp_column].values.copy()

                pred_array = df[pred_column].values.copy()

                if label_column is not None:

                    label_list = df[label_column].values.copy()

            _, rmse, mae, r2, _ = calculate_regression_metrics(

                exp_array, pred_array, return_tuple=True

            )

            fig = plt.figure(figsize=(10, 10))

            xymin = np.floor(min(exp_array.tolist() + pred_array.tolist()))

            xymax = np.ceil(max(exp_array.tolist() + pred_array.tolist()))

            log.debug(f"min {xymin}, max {xymax}")

            ticks = np.arange(xymin, xymax + 1, 1.0)

            # plt.scatter(test.y.ravel(), test.y.ravel(), label=f"Prefect Prediction", c="#89a0b0")

            plt.scatter(

                exp_array,

                pred_array,

                label=f"{task_name} $R^{2}$ {r2:.2f} RMSE {rmse:.2f} MAE {mae:.2f}",

                c="b",

            )

            plt.plot([xymin, xymax], [xymin, xymax], "k--", label="x = y")

            plt.grid()

            plt.legend()

            plt.xlabel(f"{task_name} Experimental", fontsize=25)

            plt.ylabel(f"{task_name} Prediction", fontsize=25)

            plt.title(f"{task_name} Vs. Prediction", fontsize=27)

            ax = plt.gca()

            ax.tick_params(axis="both", which="major", labelsize=17)

            ax.set_yticks(ticks)

            ax.set_xticks(ticks)

            if label_list is not None:

                try:

                    from adjustText import adjust_text

                    ax_tmp = [

                        ax.annotate(

                            labels,

                            (exp_array[ith], pred_array[ith]),

                            xytext=(exp_array[ith] + offset, pred_array[ith] + offset),

                            arrowprops=dict(arrowstyle="->", color="k"),

                        )

                        for ith, labels in enumerate(label_list)

                    ]

                    adjust_text(ax_tmp)

                except ImportError:

                    log.warning(

                        "Please install adjustText to avoid overlapping labels in the plot"

                    )

                    for ith, labels in enumerate(label_list):

                        ax.annotate(

                            labels,

                            (exp_array[ith], pred_array[ith]),

                            xytext=(exp_array[ith] + offset, pred_array[ith] + offset),

                            arrowprops=dict(arrowstyle="->", color="k"),

                        )

            plt.tight_layout()

            if no_save is False:

                plt.savefig(

                    f"evaluation_{'_'.join(task_name.replace(f'{os.sep}', '_').split())}.png"

                )

                plt.close(fig)

            else:

                return plt.gca()


### plot_png_parity_plot_coloured_by_catagorical_column

```python3
def plot_png_parity_plot_coloured_by_catagorical_column(
    df: Optional[pandas.core.frame.DataFrame] = None,
    exp_column: str = 'y',
    pred_column: str = 'y_pred',
    label_column: Optional[List[str]] = None,
    cat_column: str = 'series',
    exp_array: Optional[numpy.ndarray] = None,
    pred_array: Optional[numpy.ndarray] = None,
    cat_array: Optional[numpy.ndarray] = None,
    label_list: Optional[List[str]] = None,
    task_name: str = 'model predictions',
    offset: float = 0.1,
    no_save: bool = False
) -> matplotlib.axes._axes.Axes | None
```

Function to plot a parity plot between the experimental and predicted values and save it to a png file

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | Union[pd.DataFrame, None] | the dataframe containing the experimental and predicted columns | None |
| exp_column | str | the column name of the experimental values | None |
| pred_column | str | the column name of the predicted values | None |
| label_column | Optional[str] | the column name of the labels | None |
| exp_array | Union[np.ndarray, None] | the experimental values | None |
| pred_array | Union[np.ndarray, None] | the predicted values | None |
| cat_column | str | the column name of the catagorical values | None |
| label_list | Optional[List[str]] | the list of labels | None |
| task_name | str | the name of the task | None |
| offset | float | the offset for the labels | None |
| no_save | bool | whether to save the plot | None |

**Returns:**

| Type | Description |
|---|---|
| plt.Axes | the plot | None |

??? example "View Source"
        def plot_png_parity_plot_coloured_by_catagorical_column(

            df: Union[pd.DataFrame, None] = None,

            exp_column: str = "y",

            pred_column: str = "y_pred",

            label_column: Optional[List[str]] = None,

            cat_column: str = "series",

            exp_array: Union[np.ndarray, None] = None,

            pred_array: Union[np.ndarray, None] = None,

            cat_array: Union[np.ndarray, None] = None,

            label_list: Optional[List[str]] = None,

            task_name: str = "model predictions",

            offset: float = 0.1,

            no_save: bool = False,

        ) -> plt.Axes | None:

            """

            Function to plot a parity plot between the experimental and predicted values and save it to a png file

            Args:

                df (Union[pd.DataFrame, None]): the dataframe containing the experimental and predicted columns

                exp_column (str): the column name of the experimental values

                pred_column (str): the column name of the predicted values

                label_column (Optional[str]): the column name of the labels

                exp_array (Union[np.ndarray, None]): the experimental values

                pred_array (Union[np.ndarray, None]): the predicted values

                cat_column (str): the column name of the catagorical values

                label_list (Optional[List[str]]): the list of labels

                task_name (str): the name of the task

                offset (float): the offset for the labels

                no_save (bool): whether to save the plot

            Returns:

                plt.Axes: the plot | None

            """

            if df is not None:

                # Copy to avoid changing the original dataframe

                exp_array = df[exp_column].values.copy()

                pred_array = df[pred_column].values.copy()

                cat_array = df[cat_column].values.copy()

                if label_column is not None:

                    label_list = df[label_column].values.copy()

            cat_values = set(cat_array)

            fig = plt.figure(figsize=(10, 10))

            xymin = np.floor(min(exp_array.tolist() + pred_array.tolist()))

            xymax = np.ceil(max(exp_array.tolist() + pred_array.tolist()))

            log.debug(f"min {xymin}, max {xymax}")

            ticks = np.arange(xymin, xymax + 1, 1.0)

            colours = iter(cm.rainbow(np.linspace(0, 1, len(cat_values))))

            for cat in cat_values:

                indexes = np.where(cat_array == cat)

                exp = exp_array[indexes]

                pred = pred_array[indexes]

                _, rmse, mae, r2, _ = calculate_regression_metrics(exp, pred, return_tuple=True)

                plt.scatter(

                    exp,

                    pred,

                    label=f"{task_name} {cat} $R^{2}$ {r2:.2f} RMSE {rmse:.2f} MAE {mae:.2f}",

                    c=next(colours),

                )

            plt.plot([xymin, xymax], [xymin, xymax], "k--", label="x = y")

            plt.grid()

            plt.legend()

            plt.xlabel(f"{task_name} Experimental", fontsize=25)

            plt.ylabel(f"{task_name} Prediction", fontsize=25)

            plt.title(f"{task_name} Vs. Prediction", fontsize=27)

            ax = plt.gca()

            ax.tick_params(axis="both", which="major", labelsize=17)

            if label_list is not None:

                try:

                    from adjustText import adjust_text

                    ax_tmp = [

                        ax.annotate(

                            labels,

                            (exp_array[ith], pred_array[ith]),

                            xytext=(exp_array[ith] + offset, pred_array[ith] + offset),

                            arrowprops=dict(arrowstyle="->", color="k"),

                        )

                        for ith, labels in enumerate(label_list)

                    ]

                    adjust_text(ax_tmp)

                except ImportError:

                    log.warning(

                        "Please install adjustText to avoid overlapping labels in the plot"

                    )

                    for ith, labels in enumerate(label_list):

                        ax.annotate(

                            labels,

                            (exp_array[ith], pred_array[ith]),

                            xytext=(exp_array[ith] + offset, pred_array[ith] + offset),

                            arrowprops=dict(arrowstyle="->", color="k"),

                        )

            ax.set_yticks(ticks)

            ax.set_xticks(ticks)

            plt.tight_layout()

            if no_save is False:

                plt.savefig(

                    f"evaluation_{'_'.join(task_name.replace(f'{os.sep}', '_').split())}.png"

                )

                plt.close(fig)

            else:

                return plt.gca()


### plot_prediction_error

```python3
def plot_prediction_error(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    filename: Optional[str] = 'prediction_error_plot.png'
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

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The prediction error plot |

??? example "View Source"
        def plot_prediction_error(

            y_test,

            y_pred,

            style="seaborn-v0_8-dark-palette",

            size=(10, 10),

            title_fontsize: int = 27,

            filename: Optional[str] = "prediction_error_plot.png",

        ) -> plt.Figure:

            """

            Function to plot the prediction error plot of the test set predictions

            Args:

                y_test (np.ndarray): The test set target values

                y_pred (np.ndarray): The predicted values

                style (str): The style of the plot

                size (Tuple[int, int]): The size of the plot

                title_fontsize (int): The fontsize of the title

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

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig


### plot_qq

```python3
def plot_qq(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    filename: Optional[str] = 'qq_plot.png'
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

**Returns:**

| Type | Description |
|---|---|
| plt.Figure | The QQ plot |

??? example "View Source"
        def plot_qq(

            y_test,

            y_pred,

            style="seaborn-v0_8-dark-palette",

            size=(10, 10),

            title_fontsize: int = 27,

            filename: Optional[str] = "qq_plot.png",

        ) -> plt.Figure:

            """

            Function to plot the QQ plot of the residuals of the test set predictions

            Args:

                y_test (np.ndarray): The test set target values

                y_pred (np.ndarray): The predicted values

                style (str): The style of the plot

                size (Tuple[int, int]): The size of the plot

                title_fontsize (int): The fontsize of the title

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

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig


### plot_residuals

```python3
def plot_residuals(
    y_test,
    y_pred,
    style='seaborn-v0_8-dark-palette',
    size=(10, 10),
    title_fontsize: int = 27,
    filename: Optional[str] = 'residuals_plot.png'
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

**Returns:**

| Type | Description |
|---|---|
| plt.figure | The residuals plot |

??? example "View Source"
        def plot_residuals(

            y_test,

            y_pred,

            style="seaborn-v0_8-dark-palette",

            size=(10, 10),

            title_fontsize: int = 27,

            filename: Optional[str] = "residuals_plot.png",

        ) -> plt.Figure:

            """

            Function to plot the residuals of the test set predictions

            Args:

                y_test (np.ndarray): The test set target values

                y_pred (np.ndarray): The predicted values

                size (Tuple[int, int]): The size of the plot

                title_fontsize (int): The fontsize of the title

            Returns:

                plt.figure: The residuals plot

            """

            with plt.style.context(style=style):

                fig, ax = plt.subplots(figsize=size)

                df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})

                sns.residplot(

                    df,

                    x="y_pred",

                    y="y_test",

                    lowess=True,

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

                if filename is not None:

                    plt.savefig(filename)

            plt.close(fig)

            return fig


### rmse

```python3
def rmse(
    exp,
    pred
) -> numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]
```

Function to calculate the Root Mean Squared Error between two arrays

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| exp | np.ndarray | the expected values | None |
| pred | np.ndarray | the predicted values | None |

**Returns:**

| Type | Description |
|---|---|
| np.ndarray | the root mean squared error |

??? example "View Source"
        def rmse(exp, pred) -> np.ndarray[Any, np.dtype[Any]]:

            """

            Function to calculate the Root Mean Squared Error between two arrays

            Args:

                exp (np.ndarray): the expected values

                pred (np.ndarray): the predicted values

            Returns:

                np.ndarray: the root mean squared error

            """

            return np.sqrt(mean_squared_error(exp, pred))
