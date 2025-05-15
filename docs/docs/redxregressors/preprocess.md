# Module redxregressors.preprocess

Module for pre-processing data and filtering features

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module for pre-processing data and filtering features

        """

        import logging

        from typing import Union, Optional, List, Any

        import pandas as pd

        import numpy as np

        import matplotlib.pyplot as plt

        import scipy.stats as stats

        import seaborn as sns

        from itertools import product, combinations_with_replacement

        from tqdm import tqdm

        log = logging.getLogger(__name__)



        def check_for_non_normality(

            df, normal_p_threshold: float = 0.1, min_observations_number: int = 25

        ) -> List[Any]:

            """

            Function to check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

            Returns:

                list[Any]: a list of features that are likely not normal

            """

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normal = []

            if len(df.index) >= min_observations_number:

                log.info(

                    f"Checking for non-normality using normtest of the feature distributions with p-value threshold {normal_p_threshold}"

                )

                not_normal_p_vals = stats.normaltest(df)

                for ith, (f, p) in enumerate(zip(df.columns, not_normal_p_vals.pvalue)):

                    if p < normal_p_threshold:

                        log.warning(

                            f"Feature {f} is not normally distributed with p-value {p} (statistic (chi2): {not_normal_p_vals.statistic[ith]})"

                        )

                        likley_no_normal.append(f)

            # If there are too few observations we will use a Monte Carlo test to check for non-normality. Too few examples limits the kurtosis

            # tests used in normaltest you need at least 20 points for this test based on scipys warnings

            else:

                log.warning(

                    f"Number of observations is less than {min_observations_number} will use Monte Carlo test to check for non-normality this can be slow and inaccurate"

                )

                not_normal_p_vals = stats.monte_carlo_test(

                    df,

                    stats.norm.rvs,

                    lambda x: stats.normaltest(x).statistic,

                    alternative="greater",

                )

                for ith, (f, p) in enumerate(zip(df.columns, not_normal_p_vals.pvalue)):

                    if p < normal_p_threshold:

                        log.warning(

                            f"Feature {f} is not normally distributed with p-value {p} (statistic (chi2): {not_normal_p_vals.statistic[ith]})"

                        )

                        likley_no_normal.append(f)

            return likley_no_normal



        def plot_heatmap(

            df: pd.DataFrame,

            filename: Optional[str] = None,

            mask: Optional[Union[pd.DataFrame, np.ndarray]] = None,

            title: str = "Heatmap",

            annotation: bool = False,

            cmap: str = "coolwarm",

            image_frac: float = 0.4,

            vmin=-1.0,

            vmax=1.0,

        ) -> plt.Figure:

            """

            Function to plot a heatmap from a correlation dataframe

            Args:

                df (pd.DataFrame): the correlation dataframe

                filename (Optional[str]): the filename to save the plot to

                annotation (bool): whether to annotate the heatmap

                cmap (str): the colour map to use

                image_frac (float): the fraction of the image size

            Returns:

                plt.figure.Figure: the figure object

            """

            hw = min(int(np.ceil(len(df.index) * image_frac)), 400)

            fig, ax = plt.subplots(figsize=(hw, hw))

            if mask is None:

                hmap = sns.heatmap(

                    df,

                    vmin=vmin,

                    vmax=vmax,

                    annot=annotation,

                    cmap=cmap,

                    xticklabels=True,

                    yticklabels=True,

                    fmt=".1g",

                    ax=ax,

                )

            else:

                hmap = sns.heatmap(

                    df,

                    mask=mask,

                    vmin=vmin,

                    vmax=vmax,

                    annot=annotation,

                    cmap=cmap,

                    xticklabels=True,

                    yticklabels=True,

                    fmt=".1g",

                    ax=ax,

                )

            ax.tick_params(axis="both", which="major", labelsize=10)

            hmap.set_title(title, fontdict={"fontsize": 27}, pad=10)

            plt.tight_layout()

            if filename is not None:

                if not filename.endswith(".png"):

                    filename = f"{filename}.png"

                plt.savefig(filename)

            plt.close(fig)

            return fig



        def filter_based_on_relationship_matrix(

            df: pd.DataFrame,

            relationship_matrix: pd.DataFrame,

            threshold: float = 0.5,

            verbose: bool = False,

            greater_than: bool = False,

            less_than: bool = False,

        ) -> pd.DataFrame:

            """

            Function to filter the dataframe based on a relationship matrix. This is useful for filtering out features that are highly correlated with one another.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                relationship_matrix (pd.DataFrame): the relationship matrix

                threshold (float): the threshold to filter the features

                verbose (bool): whether to print the features that are correlated

                greater_than (bool): whether to filter the features that are greater than the threshold

                less_than (bool): whether to filter the features that are less than the threshold

            Returns:

                pd.DataFrame: the filtered dataframe

            """

            if greater_than is False and less_than is False:

                raise ValueError(

                    "You must specify either greater_than or less_than as True, usually this will be greater_than"

                )

            # Get the features that are correlated with one another ignoring the diagonal

            correlated_features = set()

            for col in relationship_matrix.columns:

                if verbose is True:

                    for ent in relationship_matrix[relationship_matrix[col] > threshold].index:

                        if ent != col:

                            log.info(f"Feature {col} is correlated with {ent}")

                # Option for greater than threshold or less than threshold

                if greater_than is True:

                    # Update the correlated features set with the features that are correlated with the current feature. Note we exclude the diagonal element i.e. the feature with itself with ent != col

                    correlated_features.update(

                        set(

                            [

                                ent

                                for ent in relationship_matrix[

                                    relationship_matrix[col] > threshold

                                ].index

                                if ent != col

                            ]

                        )

                    )

                elif less_than is True:

                    # Update the correlated features set with the features that are correlated with the current feature. Note we exclude the diagonal element i.e. the feature with itself with ent != col

                    correlated_features.update(

                        set(

                            [

                                ent

                                for ent in relationship_matrix[

                                    relationship_matrix[col] < threshold

                                ].index

                                if ent != col

                            ]

                        )

                    )

            # Remove the correlated features

            features_to_remove = list(correlated_features)

            log.info(

                f"Removing {len(features_to_remove)} features leaving {df.shape[1] - len(features_to_remove)}. Original had rows: {df.shape[0]} and columns: {df.shape[1]}"

            )

            log.info(f"Removing the following features: {features_to_remove}")

            df = df.drop(features_to_remove, axis=1)

            return df



        def filter_based_on_relationship_vector(

            df: pd.DataFrame,

            relationship_vector: pd.Series,

            threshold: float = 0.5,

            greater_than: bool = False,

            less_than: bool = False,

        ) -> pd.DataFrame:

            """

            A function to filter the dataframe based on a relationship vector. This is useful for filtering out features that are highly correlated to a target.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                relationship_vector (pdSeries): the relationship vector

                threshold (float): the threshold to filter the features

                greater_than (bool): whether to filter the features that are greater than the threshold

                less_than (bool): whether to filter the features that are less than the threshold

            Returns:

                pd.DataFrame: the filtered dataframe

            """

            if greater_than is False and less_than is False:

                raise ValueError(

                    "You must specify either greater_than or less_than as True, usually this will be less_than"

                )

            correlated_features = []

            log.info(f"relationship_vector: {type(relationship_vector)}")

            for indx, val in relationship_vector.items():

                log.info(f"Feature {indx} has a correlation of {val} with the target")

                # Option for greater than threshold or less than threshold

                if greater_than is True:

                    if val > threshold:

                        # Update the correlated features set with the features that are correlated with the target for removal

                        correlated_features.append(indx)

                elif less_than is True:

                    if val < threshold:

                        # Update the correlated features set with the features that are not strongly correlated with the target for removal

                        correlated_features.append(indx)

            features_to_remove = list(set(correlated_features))

            log.info(

                f"Removing {len(features_to_remove)} features leaving {df.shape[1] - len(features_to_remove)}. Original had rows: {df.shape[0]} and columns: {df.shape[1]}"

            )

            log.info(f"Removing the following features: {features_to_remove}")

            df = df.drop(features_to_remove, axis=1)

            return df



        def remove_highly_correlated_continous_features(

            df: pd.DataFrame,

            image_frac: float = 0.2,

            annotation: bool = False,

            correlation_method: str = "pearson",

            filename: str = "feature_correlation",

            threshold: Union[float, None] = 0.5,

            normal_p_threshold: float = 0.1,

            drop_non_normal_features: bool = False,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to remove highly correlated features from the dataframe. Note we extract the numerical features only and calculate the correlation we do not include catagorical features.

            This function also plots the correlation matrix as a heatmap.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                image_frac (float): the fraction of the image size

                annotation (bool): whether to annotate the heatmap

                correlation_method (str): the correlation method one of pearson, kendall or spearman

                filename (str): the filename to save the plot to

                threshold (float): the correlation metric threshold to filter the features or None to return the correlation matrix

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

                drop_non_normal_features (bool): whether to drop the features that are not normally distributed

                no_plot (bool): whether to plot the correlation as a heatmap

            Returns:

                pd.DataFrame: the reduced dataframe

            """

            # Extract the numerical features only

            numerical_features_continous = df[

                df.select_dtypes(include=["floating"]).columns.tolist()

            ].copy()

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normally_distributed_features = check_for_non_normality(

                numerical_features_continous, normal_p_threshold=normal_p_threshold

            )

            if drop_non_normal_features is True:

                log.info(

                    f"Dropping {len(likley_no_normally_distributed_features)} features that are likely not normally distributed"

                )

                numerical_features_continous = numerical_features_continous.drop(

                    likley_no_normally_distributed_features, axis=1

                )

            # Calculate the correlation

            corr = numerical_features_continous.corr(method=correlation_method)

            corr.to_csv(f"{filename}_{correlation_method}.csv", index=False)

            # plot the correlation as a heatmap

            if no_plot is False:

                log.info(

                    f"Plotting the correlation matrix as a heatmap saved to {filename}_{correlation_method}.png"

                )

                plot_heatmap(

                    corr,

                    filename=f"{filename}_{correlation_method}.png",

                    title="Continous Feature Correlation Heatmap",

                    annotation=annotation,

                    image_frac=image_frac,

                    vmin=-1.0,

                    vmax=1.0,

                )

            # filter the features based on the correlation matrix

            if threshold is not None:

                features_with_low_correlation = filter_based_on_relationship_matrix(

                    df, corr, threshold, greater_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on correlations less than threshold {threshold}"

                )

                return features_with_low_correlation

            else:

                log.info("Returning correlation matrix without filtering")

                return corr



        def remove_low_correlation_continous_features_to_target(

            df: pd.DataFrame,

            target: str,

            image_frac: float = 0.2,

            correlation_method: str = "pearson",

            filename: str = "feature_target_correlation",

            threshold: Union[float, None] = 0.2,

            normal_p_threshold: float = 0.1,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to plot the correlation between the features and the target. Note we extract the numerical features only and calculate the correlation we do not include catagorical features.

            Args:

                df (pd.DataFrame): the dataframe containing the features and target

                target (str): the target column name

                image_frac (float): the fraction of the image size

                correlation_method (str): the correlation method one of pearson, kendall or spearman

                filename (str): the filename to save the plot to

                threshold (float): the correlation metric threshold to filter the features or None to return the correlation matrix

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

                no_plot (bool): whether to plot the correlation as a bar chart

            Returns:

                pd.DataFrame: the reduced dataframe

            """

            # Extract the numerical features only

            numerical_features_continous = df[

                df.select_dtypes(include=["floating"]).columns.tolist()

            ].copy()

            log.debug(

                f"Minimums: {numerical_features_continous.min()} Maximums: {numerical_features_continous.max()}"

            )

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normally_distributed_features = check_for_non_normality(

                numerical_features_continous, normal_p_threshold=normal_p_threshold

            )

            if len(likley_no_normally_distributed_features) > 0:

                log.warning(

                    f"The following features are likely not normally distributed and as such may not be statistically valid: {likley_no_normally_distributed_features}"

                )

            # Calculate the correlation

            corr = numerical_features_continous.corr(method=correlation_method)[

                target

            ].sort_values(ascending=False)

            corr = corr.drop(target)

            corr.to_csv(

                f"{filename}_{correlation_method}.csv", index=True, index_label="Features"

            )

            log.info(type(corr))

            if len(corr[corr.isna()]) > 0:

                log.warning(

                    f"The following are NA and will be filled with zeros: {corr[corr.isna()]}"

                )

                corr = corr.fillna(0.0).sort_values(ascending=False)

            log.debug(f"Correlation series after filling na wih 0.0 {corr}")

            if no_plot is False:

                # get image size height and width

                hw = min(

                    int(np.ceil(len(numerical_features_continous.index) * image_frac)), 400

                )

                # plot the correlation as a bar chart

                fig, ax = plt.subplots(figsize=(hw, hw / 4.0))

                corr.plot(kind="bar", ax=ax)

                ax.set_title(

                    f"Numerical Feature Target {correlation_method.capitalize()} Correlation",

                    fontsize=27,

                )

                ax.set_ylabel(f"{correlation_method.capitalize()} Correlation", fontsize=25)

                ax.set_xlabel("Feature", fontsize=25)

                ax.tick_params(axis="both", which="major", labelsize=15)

                plt.tight_layout()

                plt.savefig(f"{filename}_{correlation_method}.png")

                plt.close(fig)

            if threshold is not None:

                features_with_high_correlation = filter_based_on_relationship_vector(

                    df, abs(corr), threshold, less_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on correlations less than threshold {threshold}"

                )

                return features_with_high_correlation

            else:

                log.info("Returning correlation series without filtering")

                return corr



        def remove_significantly_related_categorical_features(

            df: pd.DataFrame,

            include_int: bool = True,

            image_frac: float = 0.7,

            annotation: bool = False,

            significant: float = 0.05,

            filename: str = "chi2_pvalue.csv",

            filter: bool = True,

            include_diagonal: bool = False,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to plot the Chi squared p-values between the catagorical features and reduce the feature matrix to include only those with a non-significant relationship between them.

            A feature with a significant (<0.05 p-value) is likley to have a dependency on one another.  Note we extract the catagorical features only and calculate the correlation we do not include continous features.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                include_int (bool): whether to include integer features

                image_frac (float): the fraction of the image size

                annotation (bool): whether to annotate the heatmap

                significant (float): the significant level for the p-value

                filename (str): the filename to save the p-value matrix to

                filter (bool): whether to filter the features based on the p-value

                include_diagonal (bool): whether to include the diagonal in the plot

                no_plot (bool): whether to plot the correlation as a heatmap

            Returns:

                pd.DataFrame: the reduced dataframe

            """

            # Extract the catagorical features only

            if include_int is True:

                inc_types = ["object", "integer"]

            else:

                inc_types = ["object"]

            catagorical_features = df[

                df.select_dtypes(include=inc_types).columns.tolist()

            ].copy()

            feats = catagorical_features.columns.tolist()

            combinations = list(combinations_with_replacement(feats, 2))

            result = []

            for ent in tqdm(

                combinations, desc="Calculating Chi^2 p-value for combinations ....."

            ):

                chi2 = stats.chi2_contingency(

                    pd.crosstab(catagorical_features[ent[0]], catagorical_features[ent[1]])

                )

                log.debug(chi2)

                log.debug(f"Chi^2 p-value for {ent[0]} and {ent[1]} is {chi2.pvalue}")

                result.append((ent[0], ent[1], chi2.pvalue))

                # Bonferroni correction? Not sure if it is appropiate here so will leave for now

                # number_of_tests = len(pd.DataFrame(np.sort(pd.crosstab(catagorical_features[ent[0]], catagorical_features[ent[1]]), axis=1),columns =df.columns ).drop_duplicates().index)

                # log.debug(f"Number of tests: {number_of_tests}")

            log.debug(f"Result list: {result}")

            # Calculate the correlation

            corr = pd.DataFrame(result, columns=["feature 1", "feature 2", "chi2 p-value"])

            corr = corr.pivot(index="feature 1", columns="feature 2", values="chi2 p-value")

            corr.fillna(0.0, inplace=True)

            corr = corr.T

            log.debug(f"pvalue matrix: {corr}")

            log.debug(f"Min: {corr.min().min()} Max: {corr.max().max()}")

            if include_diagonal is True:

                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

            else:

                mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

            log.debug(f"Mask for the lower triangle: {mask}")

            corr.to_csv(filename, index=False)

            if no_plot is False:

                # filter the features based on the correlation matrix

                hw = min(int(np.ceil(len(corr.index) * image_frac)), 400)

                fig, ax = plt.subplots(figsize=(hw, hw))

                hmap = sns.heatmap(

                    corr,

                    mask=mask,

                    vmin=0.0,

                    vmax=max(1.0, corr.max().max()),

                    annot=annotation,

                    xticklabels=True,

                    yticklabels=True,

                    cmap="coolwarm_r",

                    fmt=".1g",

                    ax=ax,

                )

                cbar = hmap.collections[0].colorbar

                cbar.ax.tick_params(labelsize=25)

                ax.tick_params(axis="both", which="major", labelsize=15)

                hmap.set_title(

                    f"Catagorical Feature $Chi^{2}$ P-Value", fontdict={"fontsize": 27}, pad=10

                )

                plt.xlabel("")

                plt.ylabel("")

                plt.tight_layout()

                plt.savefig(f"{filename}.png")

                plt.close(fig)

            log.info(

                "Catagorical feature chi^2 p-value heatmap has been generated. Where the p-value is < 0.05 the features are likely to be dependent on one another."

            )

            corr_sq_matrix = np.tril(corr.values)

            corr_sq_matrix = (

                corr_sq_matrix + corr_sq_matrix.T - np.diag(np.diag(corr_sq_matrix))

            )

            corr_sq_matrix_df = pd.DataFrame(

                corr_sq_matrix, index=corr.index, columns=corr.columns

            )

            if filter is True:

                features_with_low_correlation = filter_based_on_relationship_matrix(

                    df, corr_sq_matrix_df, significant, less_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on p-values from the chi2 test less than threshold {significant}"

                )

                return features_with_low_correlation

            else:

                log.info("Returning p-value matrix without filtering")

                return corr_sq_matrix_df



        def remove_high_correlation_binary_categorical_and_continous_features(

            df: Optional[pd.DataFrame] = None,

            continuous_df: Optional[pd.DataFrame] = None,

            categorical_df: Optional[pd.DataFrame] = None,

            threshold: float = 0.5,

            image_frac: float = 0.7,

            annotation: bool = False,

            include_int: bool = True,

            filename: str = "continous_to_catagorical_correlation",

            normal_p_threshold: float = 0.1,

            drop_non_normal_features: bool = False,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to plot the correlation between the continous features and the binary catagorical features. If a single dataframe is input we attempt to extract the continous and catagorical features using the data type.

            The continous and categorical features can be passed in explicitly in there own dataframes. The correlation between the binary catagorical features and the continous features is caluclated using the point biserial correlation coefficient.

            Args:

                df (Optional[pd.DataFrame]): the dataframe containing the features

                continuous_df (Optional[pd.DataFrame]): the dataframe containing the continous features

                categorical_df (Optional[pd.DataFrame]): the dataframe containing the catagorical features

                threshold (float): the correlation metric threshold to filter the features

                image_frac (float): the fraction of the image size

                annotation (bool): whether to annotate the heatmap

                include_int (bool): whether to include integer features

                filename (str): the filename to save the plot to

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

                drop_non_normal_features (bool): whether to drop features that are likely not normally distributed

                no_plot (bool): whether to plot the correlation as a heatmap

            Returns:

                pd.DataFrame: the reduced dataframe

            Raises:

                ValueError: if the dataframe or the continuous and categorical dataframes are not provided

                RuntimeError: if the categorical columns are not binary

            """

            if df is not None:

                continuous_df = df.select_dtypes(include=["floating"]).copy()

                if include_int is False:

                    categorical_df = df.select_dtypes(include=["object"]).copy()

                else:

                    categorical_df = df.select_dtypes(include=["object", "integer"]).copy()

            if continuous_df is None or categorical_df is None:

                raise ValueError(

                    "You must provide a dataframe or the continuous and categorical dataframes"

                )

            if any(ent > 2 for ent in categorical_df.nunique(axis=0)):

                log.warning(

                    "There are categorical columns with more than 2 unique values, this is not a binary problem and the statistics will not be valid"

                )

                raise RuntimeError(

                    "Not all categorical columns are binary the statistical test is not valid"

                )

            else:

                log.info(

                    "All categorical columns are binary the statistical test should be valid"

                )

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normally_distributed_features = check_for_non_normality(

                continuous_df, normal_p_threshold=normal_p_threshold

            )

            if drop_non_normal_features is True:

                log.info(

                    f"Dropping {len(likley_no_normally_distributed_features)} features that are likely not normally distributed  as they may not be statistically valid in the correlation calculation"

                )

                continuous_df = continuous_df.drop(

                    likley_no_normally_distributed_features, axis=1

                )

            corr = pd.DataFrame(

                np.zeros((categorical_df.shape[1], continuous_df.shape[1])),

                columns=continuous_df.columns,

                index=categorical_df.columns,

            )

            combintaton = product(continuous_df.columns, categorical_df.columns, repeat=1)

            for ith, (row, col) in enumerate(combintaton):

                log.debug(

                    f"Calculating the {ith} point biserial correlation coefficient {col} and {row}"

                )

                if np.all(categorical_df[col] == categorical_df[col].iloc[0]):

                    log.warning(f"Column {col} is constant setting the correlation to 0.0")

                    corr.loc[col, row] = 0.0

                    continue

                if np.all(continuous_df[row] == continuous_df[row].iloc[0]):

                    log.warning(f"Row {row} is constant setting the correlation to 0.0")

                    corr.loc[col, row] = 0.0

                    continue

                r = stats.pointbiserialr(categorical_df[col], continuous_df[row])

                corr.loc[col, row] = r.statistic

            corr.to_csv(f"{filename}.csv", index=True, index_label="continous")

            if no_plot is False:

                # plot the correlation as a heatmap

                log.info(

                    f"Plotting the correlation matrix as a heatmap saved to {filename}_pointbiserial.png"

                )

                plot_heatmap(

                    corr,

                    filename=f"{filename}_pointbiserial.png",

                    annotation=annotation,

                    title="Binary Categorical To Continous Correlation Heatmap",

                    image_frac=image_frac,

                    vmin=-1.0,

                    vmax=1.0,

                )

            # filter the features based on the correlation matrix

            if threshold is not None:

                if df is None:

                    try:

                        df = pd.concat([continuous_df, categorical_df], axis=1)

                    except ValueError:

                        log.error(

                            "Could not concatenate the dataframes therefore cannot filter will return the correlation matrix without filtering"

                        )

                        return corr

                features_with_low_correlation = filter_based_on_relationship_matrix(

                    df, corr, threshold, greater_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on remobing correlations larger than threshold {threshold}"

                )

                return features_with_low_correlation

            else:

                log.info("Returning correlation matrix without filtering")

                return corr



        def remove_low_correlation_binary_categorical_and_continous_target(

            df: Optional[pd.DataFrame] = None,

            continuous_df: Optional[pd.DataFrame] = None,

            categorical_df: Optional[pd.DataFrame] = None,

            threshold: float = 0.5,

            image_frac: float = 0.7,

            annotation: bool = False,

            include_int: bool = True,

            filename: str = "continous_to_catagorical_correlation",

            normal_p_threshold: float = 0.1,

            drop_non_normal_features: bool = False,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to plot the correlation between the continous features and the binary catagorical features. If a single dataframe is input we attempt to extract the continous and catagorical features using the data type.

            The continous and categorical features can be passed in explicitly in there own dataframes. The correlation between the binary catagorical features and the continous features is caluclated using the point biserial correlation coefficient.

            Args:

                df (Optional[pd.DataFrame]): the dataframe containing the features

                continuous_df (Optional[pd.DataFrame]): the dataframe containing the continous features

                categorical_df (Optional[pd.DataFrame]): the dataframe containing the catagorical features

                threshold (float): the correlation metric threshold to filter the features

                image_frac (float): the fraction of the image size

                annotation (bool): whether to annotate the heatmap

                include_int (bool): whether to include integer features

                filename (str): the filename to save the plot to

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

                drop_non_normal_features (bool): whether to drop features that are likely not normally distributed

                no_plot (bool): whether to plot the correlation as a heatmap

            Returns:

                pd.DataFrame: the reduced dataframe

            Raises:

                ValueError: if the dataframe or the continuous and categorical dataframes are not provided

                RuntimeError: if the categorical columns are not binary

            """

            if df is not None:

                continuous_df = df.select_dtypes(include=["floating"]).copy()

                if include_int is False:

                    categorical_df = df.select_dtypes(include=["object"]).copy()

                else:

                    categorical_df = df.select_dtypes(include=["object", "integer"]).copy()

            if continuous_df is None or categorical_df is None:

                raise ValueError(

                    "You must provide a dataframe or the continuous and categorical dataframes"

                )

            if any(ent > 2 for ent in categorical_df.nunique(axis=0)):

                log.warning(

                    "There are categorical columns with more than 2 unique values, this is not a binary problem and the statistics will not be valid"

                )

                raise RuntimeError(

                    "Not all categorical columns are binary the statistical test is not valid"

                )

            else:

                log.info(

                    "All categorical columns are binary the statistical test should be valid"

                )

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normally_distributed_features = check_for_non_normality(

                continuous_df, normal_p_threshold=normal_p_threshold

            )

            if drop_non_normal_features is True:

                log.info(

                    f"Dropping {len(likley_no_normally_distributed_features)} features that are likely not normally distributed as they may not be statistically valid in the correlation calculation"

                )

                continuous_df = continuous_df.drop(

                    likley_no_normally_distributed_features, axis=1

                )

            corr = pd.DataFrame(

                np.zeros((categorical_df.shape[1], continuous_df.shape[1])),

                columns=continuous_df.columns,

                index=categorical_df.columns,

            )

            combintaton = product(continuous_df.columns, categorical_df.columns, repeat=1)

            for ith, (row, col) in enumerate(combintaton):

                log.debug(

                    f"Calculating the {ith} point biserial correlation coefficient {col} and {row}"

                )

                if np.all(categorical_df[col] == categorical_df[col].iloc[0]):

                    log.warning(f"Column {col} is constant setting the correlation to 0.0")

                    corr.loc[col, row] = 0.0

                    continue

                if np.all(continuous_df[row] == continuous_df[row].iloc[0]):

                    log.warning(f"Row {row} is constant setting the correlation to 0.0")

                    corr.loc[col, row] = 0.0

                    continue

                r = stats.pointbiserialr(categorical_df[col], continuous_df[row])

                log.info(

                    f"Correlation between {col} and {row} is {r.statistic} with p-value {r.pvalue}"

                )

                corr.loc[col, row] = r.statistic

            # corr.fillna(0.0, inplace=True)

            corr.to_csv(f"{filename}.csv", index=True, index_label="continous")

            if no_plot is False:

                # plot the correlation as a heatmap

                hw = min(int(np.ceil(len(corr.index) * image_frac)), 400)

                fig, ax = plt.subplots(figsize=(hw, hw))

                hmap = sns.heatmap(

                    corr,

                    vmin=-1.0,

                    vmax=1.0,

                    annot=annotation,

                    cmap="coolwarm",

                    xticklabels=True,

                    yticklabels=True,

                    fmt=".1g",

                    ax=ax,

                )

                ax.tick_params(axis="both", which="major", labelsize=10)

                hmap.set_title(

                    "Continous Feature Correlation Heatmap", fontdict={"fontsize": 27}, pad=10

                )

                plt.tight_layout()

                plt.savefig(f"{filename}_pointbiserial.png")

                plt.close(fig)

            if threshold is not None:

                if df is None:

                    try:

                        df = pd.concat([continuous_df, categorical_df], axis=1)

                    except ValueError:

                        log.error(

                            "Could not concatenate the dataframes therefore cannot filter will return the correlation matrix without filtering"

                        )

                        return corr

                features_with_low_correlation = filter_based_on_relationship_matrix(

                    df, abs(corr), threshold, less_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on removing correlations less than threshold {threshold}"

                )

                return features_with_low_correlation

            else:

                log.info("Returning correlation matrix without filtering")

                return corr



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### check_for_non_normality

```python3
def check_for_non_normality(
    df,
    normal_p_threshold: float = 0.1,
    min_observations_number: int = 25
) -> List[Any]
```

Function to check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | the dataframe containing the features | None |
| normal_p_threshold | float | the p-value below which we accepted the alternative hypothesis that the feature is not normal | None |

**Returns:**

| Type | Description |
|---|---|
| list[Any] | a list of features that are likely not normal |

??? example "View Source"
        def check_for_non_normality(

            df, normal_p_threshold: float = 0.1, min_observations_number: int = 25

        ) -> List[Any]:

            """

            Function to check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

            Returns:

                list[Any]: a list of features that are likely not normal

            """

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normal = []

            if len(df.index) >= min_observations_number:

                log.info(

                    f"Checking for non-normality using normtest of the feature distributions with p-value threshold {normal_p_threshold}"

                )

                not_normal_p_vals = stats.normaltest(df)

                for ith, (f, p) in enumerate(zip(df.columns, not_normal_p_vals.pvalue)):

                    if p < normal_p_threshold:

                        log.warning(

                            f"Feature {f} is not normally distributed with p-value {p} (statistic (chi2): {not_normal_p_vals.statistic[ith]})"

                        )

                        likley_no_normal.append(f)

            # If there are too few observations we will use a Monte Carlo test to check for non-normality. Too few examples limits the kurtosis

            # tests used in normaltest you need at least 20 points for this test based on scipys warnings

            else:

                log.warning(

                    f"Number of observations is less than {min_observations_number} will use Monte Carlo test to check for non-normality this can be slow and inaccurate"

                )

                not_normal_p_vals = stats.monte_carlo_test(

                    df,

                    stats.norm.rvs,

                    lambda x: stats.normaltest(x).statistic,

                    alternative="greater",

                )

                for ith, (f, p) in enumerate(zip(df.columns, not_normal_p_vals.pvalue)):

                    if p < normal_p_threshold:

                        log.warning(

                            f"Feature {f} is not normally distributed with p-value {p} (statistic (chi2): {not_normal_p_vals.statistic[ith]})"

                        )

                        likley_no_normal.append(f)

            return likley_no_normal


### filter_based_on_relationship_matrix

```python3
def filter_based_on_relationship_matrix(
    df: pandas.core.frame.DataFrame,
    relationship_matrix: pandas.core.frame.DataFrame,
    threshold: float = 0.5,
    verbose: bool = False,
    greater_than: bool = False,
    less_than: bool = False
) -> pandas.core.frame.DataFrame
```

Function to filter the dataframe based on a relationship matrix. This is useful for filtering out features that are highly correlated with one another.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | the dataframe containing the features | None |
| relationship_matrix | pd.DataFrame | the relationship matrix | None |
| threshold | float | the threshold to filter the features | None |
| verbose | bool | whether to print the features that are correlated | None |
| greater_than | bool | whether to filter the features that are greater than the threshold | None |
| less_than | bool | whether to filter the features that are less than the threshold | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the filtered dataframe |

??? example "View Source"
        def filter_based_on_relationship_matrix(

            df: pd.DataFrame,

            relationship_matrix: pd.DataFrame,

            threshold: float = 0.5,

            verbose: bool = False,

            greater_than: bool = False,

            less_than: bool = False,

        ) -> pd.DataFrame:

            """

            Function to filter the dataframe based on a relationship matrix. This is useful for filtering out features that are highly correlated with one another.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                relationship_matrix (pd.DataFrame): the relationship matrix

                threshold (float): the threshold to filter the features

                verbose (bool): whether to print the features that are correlated

                greater_than (bool): whether to filter the features that are greater than the threshold

                less_than (bool): whether to filter the features that are less than the threshold

            Returns:

                pd.DataFrame: the filtered dataframe

            """

            if greater_than is False and less_than is False:

                raise ValueError(

                    "You must specify either greater_than or less_than as True, usually this will be greater_than"

                )

            # Get the features that are correlated with one another ignoring the diagonal

            correlated_features = set()

            for col in relationship_matrix.columns:

                if verbose is True:

                    for ent in relationship_matrix[relationship_matrix[col] > threshold].index:

                        if ent != col:

                            log.info(f"Feature {col} is correlated with {ent}")

                # Option for greater than threshold or less than threshold

                if greater_than is True:

                    # Update the correlated features set with the features that are correlated with the current feature. Note we exclude the diagonal element i.e. the feature with itself with ent != col

                    correlated_features.update(

                        set(

                            [

                                ent

                                for ent in relationship_matrix[

                                    relationship_matrix[col] > threshold

                                ].index

                                if ent != col

                            ]

                        )

                    )

                elif less_than is True:

                    # Update the correlated features set with the features that are correlated with the current feature. Note we exclude the diagonal element i.e. the feature with itself with ent != col

                    correlated_features.update(

                        set(

                            [

                                ent

                                for ent in relationship_matrix[

                                    relationship_matrix[col] < threshold

                                ].index

                                if ent != col

                            ]

                        )

                    )

            # Remove the correlated features

            features_to_remove = list(correlated_features)

            log.info(

                f"Removing {len(features_to_remove)} features leaving {df.shape[1] - len(features_to_remove)}. Original had rows: {df.shape[0]} and columns: {df.shape[1]}"

            )

            log.info(f"Removing the following features: {features_to_remove}")

            df = df.drop(features_to_remove, axis=1)

            return df


### filter_based_on_relationship_vector

```python3
def filter_based_on_relationship_vector(
    df: pandas.core.frame.DataFrame,
    relationship_vector: pandas.core.series.Series,
    threshold: float = 0.5,
    greater_than: bool = False,
    less_than: bool = False
) -> pandas.core.frame.DataFrame
```

A function to filter the dataframe based on a relationship vector. This is useful for filtering out features that are highly correlated to a target.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | the dataframe containing the features | None |
| relationship_vector | pdSeries | the relationship vector | None |
| threshold | float | the threshold to filter the features | None |
| greater_than | bool | whether to filter the features that are greater than the threshold | None |
| less_than | bool | whether to filter the features that are less than the threshold | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the filtered dataframe |

??? example "View Source"
        def filter_based_on_relationship_vector(

            df: pd.DataFrame,

            relationship_vector: pd.Series,

            threshold: float = 0.5,

            greater_than: bool = False,

            less_than: bool = False,

        ) -> pd.DataFrame:

            """

            A function to filter the dataframe based on a relationship vector. This is useful for filtering out features that are highly correlated to a target.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                relationship_vector (pdSeries): the relationship vector

                threshold (float): the threshold to filter the features

                greater_than (bool): whether to filter the features that are greater than the threshold

                less_than (bool): whether to filter the features that are less than the threshold

            Returns:

                pd.DataFrame: the filtered dataframe

            """

            if greater_than is False and less_than is False:

                raise ValueError(

                    "You must specify either greater_than or less_than as True, usually this will be less_than"

                )

            correlated_features = []

            log.info(f"relationship_vector: {type(relationship_vector)}")

            for indx, val in relationship_vector.items():

                log.info(f"Feature {indx} has a correlation of {val} with the target")

                # Option for greater than threshold or less than threshold

                if greater_than is True:

                    if val > threshold:

                        # Update the correlated features set with the features that are correlated with the target for removal

                        correlated_features.append(indx)

                elif less_than is True:

                    if val < threshold:

                        # Update the correlated features set with the features that are not strongly correlated with the target for removal

                        correlated_features.append(indx)

            features_to_remove = list(set(correlated_features))

            log.info(

                f"Removing {len(features_to_remove)} features leaving {df.shape[1] - len(features_to_remove)}. Original had rows: {df.shape[0]} and columns: {df.shape[1]}"

            )

            log.info(f"Removing the following features: {features_to_remove}")

            df = df.drop(features_to_remove, axis=1)

            return df


### plot_heatmap

```python3
def plot_heatmap(
    df: pandas.core.frame.DataFrame,
    filename: Optional[str] = None,
    mask: Union[pandas.core.frame.DataFrame, numpy.ndarray, NoneType] = None,
    title: str = 'Heatmap',
    annotation: bool = False,
    cmap: str = 'coolwarm',
    image_frac: float = 0.4,
    vmin=-1.0,
    vmax=1.0
) -> matplotlib.figure.Figure
```

Function to plot a heatmap from a correlation dataframe

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | the correlation dataframe | None |
| filename | Optional[str] | the filename to save the plot to | None |
| annotation | bool | whether to annotate the heatmap | None |
| cmap | str | the colour map to use | None |
| image_frac | float | the fraction of the image size | None |

**Returns:**

| Type | Description |
|---|---|
| plt.figure.Figure | the figure object |

??? example "View Source"
        def plot_heatmap(

            df: pd.DataFrame,

            filename: Optional[str] = None,

            mask: Optional[Union[pd.DataFrame, np.ndarray]] = None,

            title: str = "Heatmap",

            annotation: bool = False,

            cmap: str = "coolwarm",

            image_frac: float = 0.4,

            vmin=-1.0,

            vmax=1.0,

        ) -> plt.Figure:

            """

            Function to plot a heatmap from a correlation dataframe

            Args:

                df (pd.DataFrame): the correlation dataframe

                filename (Optional[str]): the filename to save the plot to

                annotation (bool): whether to annotate the heatmap

                cmap (str): the colour map to use

                image_frac (float): the fraction of the image size

            Returns:

                plt.figure.Figure: the figure object

            """

            hw = min(int(np.ceil(len(df.index) * image_frac)), 400)

            fig, ax = plt.subplots(figsize=(hw, hw))

            if mask is None:

                hmap = sns.heatmap(

                    df,

                    vmin=vmin,

                    vmax=vmax,

                    annot=annotation,

                    cmap=cmap,

                    xticklabels=True,

                    yticklabels=True,

                    fmt=".1g",

                    ax=ax,

                )

            else:

                hmap = sns.heatmap(

                    df,

                    mask=mask,

                    vmin=vmin,

                    vmax=vmax,

                    annot=annotation,

                    cmap=cmap,

                    xticklabels=True,

                    yticklabels=True,

                    fmt=".1g",

                    ax=ax,

                )

            ax.tick_params(axis="both", which="major", labelsize=10)

            hmap.set_title(title, fontdict={"fontsize": 27}, pad=10)

            plt.tight_layout()

            if filename is not None:

                if not filename.endswith(".png"):

                    filename = f"{filename}.png"

                plt.savefig(filename)

            plt.close(fig)

            return fig


### remove_high_correlation_binary_categorical_and_continous_features

```python3
def remove_high_correlation_binary_categorical_and_continous_features(
    df: Optional[pandas.core.frame.DataFrame] = None,
    continuous_df: Optional[pandas.core.frame.DataFrame] = None,
    categorical_df: Optional[pandas.core.frame.DataFrame] = None,
    threshold: float = 0.5,
    image_frac: float = 0.7,
    annotation: bool = False,
    include_int: bool = True,
    filename: str = 'continous_to_catagorical_correlation',
    normal_p_threshold: float = 0.1,
    drop_non_normal_features: bool = False,
    no_plot: bool = False
) -> pandas.core.frame.DataFrame
```

Function to plot the correlation between the continous features and the binary catagorical features. If a single dataframe is input we attempt to extract the continous and catagorical features using the data type.

The continous and categorical features can be passed in explicitly in there own dataframes. The correlation between the binary catagorical features and the continous features is caluclated using the point biserial correlation coefficient.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | Optional[pd.DataFrame] | the dataframe containing the features | None |
| continuous_df | Optional[pd.DataFrame] | the dataframe containing the continous features | None |
| categorical_df | Optional[pd.DataFrame] | the dataframe containing the catagorical features | None |
| threshold | float | the correlation metric threshold to filter the features | None |
| image_frac | float | the fraction of the image size | None |
| annotation | bool | whether to annotate the heatmap | None |
| include_int | bool | whether to include integer features | None |
| filename | str | the filename to save the plot to | None |
| normal_p_threshold | float | the p-value below which we accepted the alternative hypothesis that the feature is not normal | None |
| drop_non_normal_features | bool | whether to drop features that are likely not normally distributed | None |
| no_plot | bool | whether to plot the correlation as a heatmap | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the reduced dataframe |

**Raises:**

| Type | Description |
|---|---|
| ValueError | if the dataframe or the continuous and categorical dataframes are not provided |
| RuntimeError | if the categorical columns are not binary |

??? example "View Source"
        def remove_high_correlation_binary_categorical_and_continous_features(

            df: Optional[pd.DataFrame] = None,

            continuous_df: Optional[pd.DataFrame] = None,

            categorical_df: Optional[pd.DataFrame] = None,

            threshold: float = 0.5,

            image_frac: float = 0.7,

            annotation: bool = False,

            include_int: bool = True,

            filename: str = "continous_to_catagorical_correlation",

            normal_p_threshold: float = 0.1,

            drop_non_normal_features: bool = False,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to plot the correlation between the continous features and the binary catagorical features. If a single dataframe is input we attempt to extract the continous and catagorical features using the data type.

            The continous and categorical features can be passed in explicitly in there own dataframes. The correlation between the binary catagorical features and the continous features is caluclated using the point biserial correlation coefficient.

            Args:

                df (Optional[pd.DataFrame]): the dataframe containing the features

                continuous_df (Optional[pd.DataFrame]): the dataframe containing the continous features

                categorical_df (Optional[pd.DataFrame]): the dataframe containing the catagorical features

                threshold (float): the correlation metric threshold to filter the features

                image_frac (float): the fraction of the image size

                annotation (bool): whether to annotate the heatmap

                include_int (bool): whether to include integer features

                filename (str): the filename to save the plot to

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

                drop_non_normal_features (bool): whether to drop features that are likely not normally distributed

                no_plot (bool): whether to plot the correlation as a heatmap

            Returns:

                pd.DataFrame: the reduced dataframe

            Raises:

                ValueError: if the dataframe or the continuous and categorical dataframes are not provided

                RuntimeError: if the categorical columns are not binary

            """

            if df is not None:

                continuous_df = df.select_dtypes(include=["floating"]).copy()

                if include_int is False:

                    categorical_df = df.select_dtypes(include=["object"]).copy()

                else:

                    categorical_df = df.select_dtypes(include=["object", "integer"]).copy()

            if continuous_df is None or categorical_df is None:

                raise ValueError(

                    "You must provide a dataframe or the continuous and categorical dataframes"

                )

            if any(ent > 2 for ent in categorical_df.nunique(axis=0)):

                log.warning(

                    "There are categorical columns with more than 2 unique values, this is not a binary problem and the statistics will not be valid"

                )

                raise RuntimeError(

                    "Not all categorical columns are binary the statistical test is not valid"

                )

            else:

                log.info(

                    "All categorical columns are binary the statistical test should be valid"

                )

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normally_distributed_features = check_for_non_normality(

                continuous_df, normal_p_threshold=normal_p_threshold

            )

            if drop_non_normal_features is True:

                log.info(

                    f"Dropping {len(likley_no_normally_distributed_features)} features that are likely not normally distributed  as they may not be statistically valid in the correlation calculation"

                )

                continuous_df = continuous_df.drop(

                    likley_no_normally_distributed_features, axis=1

                )

            corr = pd.DataFrame(

                np.zeros((categorical_df.shape[1], continuous_df.shape[1])),

                columns=continuous_df.columns,

                index=categorical_df.columns,

            )

            combintaton = product(continuous_df.columns, categorical_df.columns, repeat=1)

            for ith, (row, col) in enumerate(combintaton):

                log.debug(

                    f"Calculating the {ith} point biserial correlation coefficient {col} and {row}"

                )

                if np.all(categorical_df[col] == categorical_df[col].iloc[0]):

                    log.warning(f"Column {col} is constant setting the correlation to 0.0")

                    corr.loc[col, row] = 0.0

                    continue

                if np.all(continuous_df[row] == continuous_df[row].iloc[0]):

                    log.warning(f"Row {row} is constant setting the correlation to 0.0")

                    corr.loc[col, row] = 0.0

                    continue

                r = stats.pointbiserialr(categorical_df[col], continuous_df[row])

                corr.loc[col, row] = r.statistic

            corr.to_csv(f"{filename}.csv", index=True, index_label="continous")

            if no_plot is False:

                # plot the correlation as a heatmap

                log.info(

                    f"Plotting the correlation matrix as a heatmap saved to {filename}_pointbiserial.png"

                )

                plot_heatmap(

                    corr,

                    filename=f"{filename}_pointbiserial.png",

                    annotation=annotation,

                    title="Binary Categorical To Continous Correlation Heatmap",

                    image_frac=image_frac,

                    vmin=-1.0,

                    vmax=1.0,

                )

            # filter the features based on the correlation matrix

            if threshold is not None:

                if df is None:

                    try:

                        df = pd.concat([continuous_df, categorical_df], axis=1)

                    except ValueError:

                        log.error(

                            "Could not concatenate the dataframes therefore cannot filter will return the correlation matrix without filtering"

                        )

                        return corr

                features_with_low_correlation = filter_based_on_relationship_matrix(

                    df, corr, threshold, greater_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on remobing correlations larger than threshold {threshold}"

                )

                return features_with_low_correlation

            else:

                log.info("Returning correlation matrix without filtering")

                return corr


### remove_highly_correlated_continous_features

```python3
def remove_highly_correlated_continous_features(
    df: pandas.core.frame.DataFrame,
    image_frac: float = 0.2,
    annotation: bool = False,
    correlation_method: str = 'pearson',
    filename: str = 'feature_correlation',
    threshold: Optional[float] = 0.5,
    normal_p_threshold: float = 0.1,
    drop_non_normal_features: bool = False,
    no_plot: bool = False
) -> pandas.core.frame.DataFrame
```

Function to remove highly correlated features from the dataframe. Note we extract the numerical features only and calculate the correlation we do not include catagorical features.

This function also plots the correlation matrix as a heatmap.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | the dataframe containing the features | None |
| image_frac | float | the fraction of the image size | None |
| annotation | bool | whether to annotate the heatmap | None |
| correlation_method | str | the correlation method one of pearson, kendall or spearman | None |
| filename | str | the filename to save the plot to | None |
| threshold | float | the correlation metric threshold to filter the features or None to return the correlation matrix | None |
| normal_p_threshold | float | the p-value below which we accepted the alternative hypothesis that the feature is not normal | None |
| drop_non_normal_features | bool | whether to drop the features that are not normally distributed | None |
| no_plot | bool | whether to plot the correlation as a heatmap | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the reduced dataframe |

??? example "View Source"
        def remove_highly_correlated_continous_features(

            df: pd.DataFrame,

            image_frac: float = 0.2,

            annotation: bool = False,

            correlation_method: str = "pearson",

            filename: str = "feature_correlation",

            threshold: Union[float, None] = 0.5,

            normal_p_threshold: float = 0.1,

            drop_non_normal_features: bool = False,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to remove highly correlated features from the dataframe. Note we extract the numerical features only and calculate the correlation we do not include catagorical features.

            This function also plots the correlation matrix as a heatmap.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                image_frac (float): the fraction of the image size

                annotation (bool): whether to annotate the heatmap

                correlation_method (str): the correlation method one of pearson, kendall or spearman

                filename (str): the filename to save the plot to

                threshold (float): the correlation metric threshold to filter the features or None to return the correlation matrix

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

                drop_non_normal_features (bool): whether to drop the features that are not normally distributed

                no_plot (bool): whether to plot the correlation as a heatmap

            Returns:

                pd.DataFrame: the reduced dataframe

            """

            # Extract the numerical features only

            numerical_features_continous = df[

                df.select_dtypes(include=["floating"]).columns.tolist()

            ].copy()

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normally_distributed_features = check_for_non_normality(

                numerical_features_continous, normal_p_threshold=normal_p_threshold

            )

            if drop_non_normal_features is True:

                log.info(

                    f"Dropping {len(likley_no_normally_distributed_features)} features that are likely not normally distributed"

                )

                numerical_features_continous = numerical_features_continous.drop(

                    likley_no_normally_distributed_features, axis=1

                )

            # Calculate the correlation

            corr = numerical_features_continous.corr(method=correlation_method)

            corr.to_csv(f"{filename}_{correlation_method}.csv", index=False)

            # plot the correlation as a heatmap

            if no_plot is False:

                log.info(

                    f"Plotting the correlation matrix as a heatmap saved to {filename}_{correlation_method}.png"

                )

                plot_heatmap(

                    corr,

                    filename=f"{filename}_{correlation_method}.png",

                    title="Continous Feature Correlation Heatmap",

                    annotation=annotation,

                    image_frac=image_frac,

                    vmin=-1.0,

                    vmax=1.0,

                )

            # filter the features based on the correlation matrix

            if threshold is not None:

                features_with_low_correlation = filter_based_on_relationship_matrix(

                    df, corr, threshold, greater_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on correlations less than threshold {threshold}"

                )

                return features_with_low_correlation

            else:

                log.info("Returning correlation matrix without filtering")

                return corr


### remove_low_correlation_binary_categorical_and_continous_target

```python3
def remove_low_correlation_binary_categorical_and_continous_target(
    df: Optional[pandas.core.frame.DataFrame] = None,
    continuous_df: Optional[pandas.core.frame.DataFrame] = None,
    categorical_df: Optional[pandas.core.frame.DataFrame] = None,
    threshold: float = 0.5,
    image_frac: float = 0.7,
    annotation: bool = False,
    include_int: bool = True,
    filename: str = 'continous_to_catagorical_correlation',
    normal_p_threshold: float = 0.1,
    drop_non_normal_features: bool = False,
    no_plot: bool = False
) -> pandas.core.frame.DataFrame
```

Function to plot the correlation between the continous features and the binary catagorical features. If a single dataframe is input we attempt to extract the continous and catagorical features using the data type.

The continous and categorical features can be passed in explicitly in there own dataframes. The correlation between the binary catagorical features and the continous features is caluclated using the point biserial correlation coefficient.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | Optional[pd.DataFrame] | the dataframe containing the features | None |
| continuous_df | Optional[pd.DataFrame] | the dataframe containing the continous features | None |
| categorical_df | Optional[pd.DataFrame] | the dataframe containing the catagorical features | None |
| threshold | float | the correlation metric threshold to filter the features | None |
| image_frac | float | the fraction of the image size | None |
| annotation | bool | whether to annotate the heatmap | None |
| include_int | bool | whether to include integer features | None |
| filename | str | the filename to save the plot to | None |
| normal_p_threshold | float | the p-value below which we accepted the alternative hypothesis that the feature is not normal | None |
| drop_non_normal_features | bool | whether to drop features that are likely not normally distributed | None |
| no_plot | bool | whether to plot the correlation as a heatmap | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the reduced dataframe |

**Raises:**

| Type | Description |
|---|---|
| ValueError | if the dataframe or the continuous and categorical dataframes are not provided |
| RuntimeError | if the categorical columns are not binary |

??? example "View Source"
        def remove_low_correlation_binary_categorical_and_continous_target(

            df: Optional[pd.DataFrame] = None,

            continuous_df: Optional[pd.DataFrame] = None,

            categorical_df: Optional[pd.DataFrame] = None,

            threshold: float = 0.5,

            image_frac: float = 0.7,

            annotation: bool = False,

            include_int: bool = True,

            filename: str = "continous_to_catagorical_correlation",

            normal_p_threshold: float = 0.1,

            drop_non_normal_features: bool = False,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to plot the correlation between the continous features and the binary catagorical features. If a single dataframe is input we attempt to extract the continous and catagorical features using the data type.

            The continous and categorical features can be passed in explicitly in there own dataframes. The correlation between the binary catagorical features and the continous features is caluclated using the point biserial correlation coefficient.

            Args:

                df (Optional[pd.DataFrame]): the dataframe containing the features

                continuous_df (Optional[pd.DataFrame]): the dataframe containing the continous features

                categorical_df (Optional[pd.DataFrame]): the dataframe containing the catagorical features

                threshold (float): the correlation metric threshold to filter the features

                image_frac (float): the fraction of the image size

                annotation (bool): whether to annotate the heatmap

                include_int (bool): whether to include integer features

                filename (str): the filename to save the plot to

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

                drop_non_normal_features (bool): whether to drop features that are likely not normally distributed

                no_plot (bool): whether to plot the correlation as a heatmap

            Returns:

                pd.DataFrame: the reduced dataframe

            Raises:

                ValueError: if the dataframe or the continuous and categorical dataframes are not provided

                RuntimeError: if the categorical columns are not binary

            """

            if df is not None:

                continuous_df = df.select_dtypes(include=["floating"]).copy()

                if include_int is False:

                    categorical_df = df.select_dtypes(include=["object"]).copy()

                else:

                    categorical_df = df.select_dtypes(include=["object", "integer"]).copy()

            if continuous_df is None or categorical_df is None:

                raise ValueError(

                    "You must provide a dataframe or the continuous and categorical dataframes"

                )

            if any(ent > 2 for ent in categorical_df.nunique(axis=0)):

                log.warning(

                    "There are categorical columns with more than 2 unique values, this is not a binary problem and the statistics will not be valid"

                )

                raise RuntimeError(

                    "Not all categorical columns are binary the statistical test is not valid"

                )

            else:

                log.info(

                    "All categorical columns are binary the statistical test should be valid"

                )

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normally_distributed_features = check_for_non_normality(

                continuous_df, normal_p_threshold=normal_p_threshold

            )

            if drop_non_normal_features is True:

                log.info(

                    f"Dropping {len(likley_no_normally_distributed_features)} features that are likely not normally distributed as they may not be statistically valid in the correlation calculation"

                )

                continuous_df = continuous_df.drop(

                    likley_no_normally_distributed_features, axis=1

                )

            corr = pd.DataFrame(

                np.zeros((categorical_df.shape[1], continuous_df.shape[1])),

                columns=continuous_df.columns,

                index=categorical_df.columns,

            )

            combintaton = product(continuous_df.columns, categorical_df.columns, repeat=1)

            for ith, (row, col) in enumerate(combintaton):

                log.debug(

                    f"Calculating the {ith} point biserial correlation coefficient {col} and {row}"

                )

                if np.all(categorical_df[col] == categorical_df[col].iloc[0]):

                    log.warning(f"Column {col} is constant setting the correlation to 0.0")

                    corr.loc[col, row] = 0.0

                    continue

                if np.all(continuous_df[row] == continuous_df[row].iloc[0]):

                    log.warning(f"Row {row} is constant setting the correlation to 0.0")

                    corr.loc[col, row] = 0.0

                    continue

                r = stats.pointbiserialr(categorical_df[col], continuous_df[row])

                log.info(

                    f"Correlation between {col} and {row} is {r.statistic} with p-value {r.pvalue}"

                )

                corr.loc[col, row] = r.statistic

            # corr.fillna(0.0, inplace=True)

            corr.to_csv(f"{filename}.csv", index=True, index_label="continous")

            if no_plot is False:

                # plot the correlation as a heatmap

                hw = min(int(np.ceil(len(corr.index) * image_frac)), 400)

                fig, ax = plt.subplots(figsize=(hw, hw))

                hmap = sns.heatmap(

                    corr,

                    vmin=-1.0,

                    vmax=1.0,

                    annot=annotation,

                    cmap="coolwarm",

                    xticklabels=True,

                    yticklabels=True,

                    fmt=".1g",

                    ax=ax,

                )

                ax.tick_params(axis="both", which="major", labelsize=10)

                hmap.set_title(

                    "Continous Feature Correlation Heatmap", fontdict={"fontsize": 27}, pad=10

                )

                plt.tight_layout()

                plt.savefig(f"{filename}_pointbiserial.png")

                plt.close(fig)

            if threshold is not None:

                if df is None:

                    try:

                        df = pd.concat([continuous_df, categorical_df], axis=1)

                    except ValueError:

                        log.error(

                            "Could not concatenate the dataframes therefore cannot filter will return the correlation matrix without filtering"

                        )

                        return corr

                features_with_low_correlation = filter_based_on_relationship_matrix(

                    df, abs(corr), threshold, less_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on removing correlations less than threshold {threshold}"

                )

                return features_with_low_correlation

            else:

                log.info("Returning correlation matrix without filtering")

                return corr


### remove_low_correlation_continous_features_to_target

```python3
def remove_low_correlation_continous_features_to_target(
    df: pandas.core.frame.DataFrame,
    target: str,
    image_frac: float = 0.2,
    correlation_method: str = 'pearson',
    filename: str = 'feature_target_correlation',
    threshold: Optional[float] = 0.2,
    normal_p_threshold: float = 0.1,
    no_plot: bool = False
) -> pandas.core.frame.DataFrame
```

Function to plot the correlation between the features and the target. Note we extract the numerical features only and calculate the correlation we do not include catagorical features.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | the dataframe containing the features and target | None |
| target | str | the target column name | None |
| image_frac | float | the fraction of the image size | None |
| correlation_method | str | the correlation method one of pearson, kendall or spearman | None |
| filename | str | the filename to save the plot to | None |
| threshold | float | the correlation metric threshold to filter the features or None to return the correlation matrix | None |
| normal_p_threshold | float | the p-value below which we accepted the alternative hypothesis that the feature is not normal | None |
| no_plot | bool | whether to plot the correlation as a bar chart | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the reduced dataframe |

??? example "View Source"
        def remove_low_correlation_continous_features_to_target(

            df: pd.DataFrame,

            target: str,

            image_frac: float = 0.2,

            correlation_method: str = "pearson",

            filename: str = "feature_target_correlation",

            threshold: Union[float, None] = 0.2,

            normal_p_threshold: float = 0.1,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to plot the correlation between the features and the target. Note we extract the numerical features only and calculate the correlation we do not include catagorical features.

            Args:

                df (pd.DataFrame): the dataframe containing the features and target

                target (str): the target column name

                image_frac (float): the fraction of the image size

                correlation_method (str): the correlation method one of pearson, kendall or spearman

                filename (str): the filename to save the plot to

                threshold (float): the correlation metric threshold to filter the features or None to return the correlation matrix

                normal_p_threshold (float): the p-value below which we accepted the alternative hypothesis that the feature is not normal

                no_plot (bool): whether to plot the correlation as a bar chart

            Returns:

                pd.DataFrame: the reduced dataframe

            """

            # Extract the numerical features only

            numerical_features_continous = df[

                df.select_dtypes(include=["floating"]).columns.tolist()

            ].copy()

            log.debug(

                f"Minimums: {numerical_features_continous.min()} Maximums: {numerical_features_continous.max()}"

            )

            # check for normality of the feature distributions. Note this is a test for non-normality not confirmation of normality.

            # We assume if the p-value is less than 0.1 the feature is not normal with 90% confidence.

            likley_no_normally_distributed_features = check_for_non_normality(

                numerical_features_continous, normal_p_threshold=normal_p_threshold

            )

            if len(likley_no_normally_distributed_features) > 0:

                log.warning(

                    f"The following features are likely not normally distributed and as such may not be statistically valid: {likley_no_normally_distributed_features}"

                )

            # Calculate the correlation

            corr = numerical_features_continous.corr(method=correlation_method)[

                target

            ].sort_values(ascending=False)

            corr = corr.drop(target)

            corr.to_csv(

                f"{filename}_{correlation_method}.csv", index=True, index_label="Features"

            )

            log.info(type(corr))

            if len(corr[corr.isna()]) > 0:

                log.warning(

                    f"The following are NA and will be filled with zeros: {corr[corr.isna()]}"

                )

                corr = corr.fillna(0.0).sort_values(ascending=False)

            log.debug(f"Correlation series after filling na wih 0.0 {corr}")

            if no_plot is False:

                # get image size height and width

                hw = min(

                    int(np.ceil(len(numerical_features_continous.index) * image_frac)), 400

                )

                # plot the correlation as a bar chart

                fig, ax = plt.subplots(figsize=(hw, hw / 4.0))

                corr.plot(kind="bar", ax=ax)

                ax.set_title(

                    f"Numerical Feature Target {correlation_method.capitalize()} Correlation",

                    fontsize=27,

                )

                ax.set_ylabel(f"{correlation_method.capitalize()} Correlation", fontsize=25)

                ax.set_xlabel("Feature", fontsize=25)

                ax.tick_params(axis="both", which="major", labelsize=15)

                plt.tight_layout()

                plt.savefig(f"{filename}_{correlation_method}.png")

                plt.close(fig)

            if threshold is not None:

                features_with_high_correlation = filter_based_on_relationship_vector(

                    df, abs(corr), threshold, less_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on correlations less than threshold {threshold}"

                )

                return features_with_high_correlation

            else:

                log.info("Returning correlation series without filtering")

                return corr


### remove_significantly_related_categorical_features

```python3
def remove_significantly_related_categorical_features(
    df: pandas.core.frame.DataFrame,
    include_int: bool = True,
    image_frac: float = 0.7,
    annotation: bool = False,
    significant: float = 0.05,
    filename: str = 'chi2_pvalue.csv',
    filter: bool = True,
    include_diagonal: bool = False,
    no_plot: bool = False
) -> pandas.core.frame.DataFrame
```

Function to plot the Chi squared p-values between the catagorical features and reduce the feature matrix to include only those with a non-significant relationship between them.

A feature with a significant (<0.05 p-value) is likley to have a dependency on one another.  Note we extract the catagorical features only and calculate the correlation we do not include continous features.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | the dataframe containing the features | None |
| include_int | bool | whether to include integer features | None |
| image_frac | float | the fraction of the image size | None |
| annotation | bool | whether to annotate the heatmap | None |
| significant | float | the significant level for the p-value | None |
| filename | str | the filename to save the p-value matrix to | None |
| filter | bool | whether to filter the features based on the p-value | None |
| include_diagonal | bool | whether to include the diagonal in the plot | None |
| no_plot | bool | whether to plot the correlation as a heatmap | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | the reduced dataframe |

??? example "View Source"
        def remove_significantly_related_categorical_features(

            df: pd.DataFrame,

            include_int: bool = True,

            image_frac: float = 0.7,

            annotation: bool = False,

            significant: float = 0.05,

            filename: str = "chi2_pvalue.csv",

            filter: bool = True,

            include_diagonal: bool = False,

            no_plot: bool = False,

        ) -> pd.DataFrame:

            """

            Function to plot the Chi squared p-values between the catagorical features and reduce the feature matrix to include only those with a non-significant relationship between them.

            A feature with a significant (<0.05 p-value) is likley to have a dependency on one another.  Note we extract the catagorical features only and calculate the correlation we do not include continous features.

            Args:

                df (pd.DataFrame): the dataframe containing the features

                include_int (bool): whether to include integer features

                image_frac (float): the fraction of the image size

                annotation (bool): whether to annotate the heatmap

                significant (float): the significant level for the p-value

                filename (str): the filename to save the p-value matrix to

                filter (bool): whether to filter the features based on the p-value

                include_diagonal (bool): whether to include the diagonal in the plot

                no_plot (bool): whether to plot the correlation as a heatmap

            Returns:

                pd.DataFrame: the reduced dataframe

            """

            # Extract the catagorical features only

            if include_int is True:

                inc_types = ["object", "integer"]

            else:

                inc_types = ["object"]

            catagorical_features = df[

                df.select_dtypes(include=inc_types).columns.tolist()

            ].copy()

            feats = catagorical_features.columns.tolist()

            combinations = list(combinations_with_replacement(feats, 2))

            result = []

            for ent in tqdm(

                combinations, desc="Calculating Chi^2 p-value for combinations ....."

            ):

                chi2 = stats.chi2_contingency(

                    pd.crosstab(catagorical_features[ent[0]], catagorical_features[ent[1]])

                )

                log.debug(chi2)

                log.debug(f"Chi^2 p-value for {ent[0]} and {ent[1]} is {chi2.pvalue}")

                result.append((ent[0], ent[1], chi2.pvalue))

                # Bonferroni correction? Not sure if it is appropiate here so will leave for now

                # number_of_tests = len(pd.DataFrame(np.sort(pd.crosstab(catagorical_features[ent[0]], catagorical_features[ent[1]]), axis=1),columns =df.columns ).drop_duplicates().index)

                # log.debug(f"Number of tests: {number_of_tests}")

            log.debug(f"Result list: {result}")

            # Calculate the correlation

            corr = pd.DataFrame(result, columns=["feature 1", "feature 2", "chi2 p-value"])

            corr = corr.pivot(index="feature 1", columns="feature 2", values="chi2 p-value")

            corr.fillna(0.0, inplace=True)

            corr = corr.T

            log.debug(f"pvalue matrix: {corr}")

            log.debug(f"Min: {corr.min().min()} Max: {corr.max().max()}")

            if include_diagonal is True:

                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

            else:

                mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

            log.debug(f"Mask for the lower triangle: {mask}")

            corr.to_csv(filename, index=False)

            if no_plot is False:

                # filter the features based on the correlation matrix

                hw = min(int(np.ceil(len(corr.index) * image_frac)), 400)

                fig, ax = plt.subplots(figsize=(hw, hw))

                hmap = sns.heatmap(

                    corr,

                    mask=mask,

                    vmin=0.0,

                    vmax=max(1.0, corr.max().max()),

                    annot=annotation,

                    xticklabels=True,

                    yticklabels=True,

                    cmap="coolwarm_r",

                    fmt=".1g",

                    ax=ax,

                )

                cbar = hmap.collections[0].colorbar

                cbar.ax.tick_params(labelsize=25)

                ax.tick_params(axis="both", which="major", labelsize=15)

                hmap.set_title(

                    f"Catagorical Feature $Chi^{2}$ P-Value", fontdict={"fontsize": 27}, pad=10

                )

                plt.xlabel("")

                plt.ylabel("")

                plt.tight_layout()

                plt.savefig(f"{filename}.png")

                plt.close(fig)

            log.info(

                "Catagorical feature chi^2 p-value heatmap has been generated. Where the p-value is < 0.05 the features are likely to be dependent on one another."

            )

            corr_sq_matrix = np.tril(corr.values)

            corr_sq_matrix = (

                corr_sq_matrix + corr_sq_matrix.T - np.diag(np.diag(corr_sq_matrix))

            )

            corr_sq_matrix_df = pd.DataFrame(

                corr_sq_matrix, index=corr.index, columns=corr.columns

            )

            if filter is True:

                features_with_low_correlation = filter_based_on_relationship_matrix(

                    df, corr_sq_matrix_df, significant, less_than=True

                )

                log.info(

                    f"Returning filtered dataframe based on p-values from the chi2 test less than threshold {significant}"

                )

                return features_with_low_correlation

            else:

                log.info("Returning p-value matrix without filtering")

                return corr_sq_matrix_df
