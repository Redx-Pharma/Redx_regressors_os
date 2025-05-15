#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit tests for preprocssing
"""

import pytest
import logging
import pandas as pd
from unittest.mock import patch
from redxregressors.preprocess import (
    filter_based_on_relationship_matrix,
    filter_based_on_relationship_vector,
    remove_highly_correlated_continous_features,
    remove_low_correlation_continous_features_to_target,
    remove_significantly_related_categorical_features,
    remove_high_correlation_binary_categorical_and_continous_features,
    remove_low_correlation_binary_categorical_and_continous_target,
)

log = logging.getLogger(__name__)


def test_filter_based_on_relationship_matrix():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [4, 5, 6]})
    relationship_matrix = pd.DataFrame(
        {"A": [1.0, 0.9, 0.1], "B": [0.9, 1.0, 0.2], "C": [0.1, 0.2, 1.0]},
        index=["A", "B", "C"],
    )

    filtered_df = filter_based_on_relationship_matrix(
        df, relationship_matrix, threshold=0.8, greater_than=True
    )
    assert "B" not in filtered_df.columns

    with pytest.raises(ValueError):
        filter_based_on_relationship_matrix(df, relationship_matrix)


def test_filter_based_on_relationship_vector():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [4, 5, 6]})
    relationship_vector = pd.Series([0.9, 0.1, 0.2], index=["A", "B", "C"])

    filtered_df = filter_based_on_relationship_vector(
        df, relationship_vector, threshold=0.8, greater_than=True
    )
    assert "A" not in filtered_df.columns

    with pytest.raises(ValueError):
        filter_based_on_relationship_vector(df, relationship_vector)


@patch("redxregressors.preprocess.plt.savefig")
def test_plot_continous_features_to_features_correlation(mock_savefig):
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "C": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        }
    )

    filtered_df = remove_highly_correlated_continous_features(df, threshold=0.8)
    assert "B" not in filtered_df.columns
    mock_savefig.assert_called_once()


@patch("redxregressors.preprocess.plt.savefig")
def test_plot_continous_features_continous_target_correlation(mock_savefig):
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "C": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        }
    )
    target = "C"

    filtered_df = remove_low_correlation_continous_features_to_target(
        df, target, threshold=0.2
    )
    assert "A" in filtered_df.columns
    mock_savefig.assert_called_once()


@patch("redxregressors.preprocess.plt.savefig")
def test_plot_catagorical_features_to_catagorical_features_correlation(mock_savefig):
    df = pd.DataFrame(
        {"A": [1] * 25 + [0] * 75, "B": [1] * 90 + [0] * 10, "C": [1] * 25 + [0] * 75}
    )

    filtered_df = remove_significantly_related_categorical_features(
        df, significant=0.05
    )
    assert "B" in filtered_df.columns
    mock_savefig.assert_called_once()


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [1.0, 2.0, 3.0, 4.0],
            "C": [1, 0, 1, 0],
            "D": [1, 1, 1, 1],
        }
    )
    continuous_df = df[["A", "B"]]
    categorical_df = df[["C", "D"]]
    return df, continuous_df, categorical_df


def test_remove_high_correlation_binary_categorical_and_continous_features(sample_data):
    df = pd.DataFrame(
        {
            "A": [1.0, 3.1, 1.02, 3.05, 1.04, 1.07, 3.07, 3.04],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "C": [1, 0, 1, 0, 0, 0, 1, 1],
            "D": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    continuous_df = df[["A", "B"]]
    categorical_df = df[["C", "D"]]

    expected_df = pd.DataFrame(
        {
            "A": {0: 1.0, 1: 3.1, 2: 1.02, 3: 3.05, 4: 1.04, 5: 1.07, 6: 3.07, 7: 3.04},
            "B": {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0, 6: 7.0, 7: 8.0},
            "D": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
        }
    )

    with patch("redxregressors.preprocess.plot_heatmap") as mock_plot_heatmap, patch(
        "redxregressors.preprocess.filter_based_on_relationship_matrix", return_value=df
    ) as mock_filter:
        result = remove_high_correlation_binary_categorical_and_continous_features(
            df=df, threshold=0.5
        )

        assert isinstance(result, pd.DataFrame)
        mock_plot_heatmap.assert_called_once()
        mock_filter.assert_called_once()

    with pytest.raises(ValueError):
        remove_high_correlation_binary_categorical_and_continous_features()

    with pytest.raises(RuntimeError):
        df["E"] = [1, 2, 3, 4, 2, 3, 4, 5]
        remove_high_correlation_binary_categorical_and_continous_features(df=df)

    result = remove_high_correlation_binary_categorical_and_continous_features(
        threshold=0.1,
        continuous_df=continuous_df,
        categorical_df=categorical_df,
        annotation=True,
    )

    pd.testing.assert_frame_equal(result, expected_df)


def test_remove_low_correlation_binary_categorical_and_continous_target(sample_data):
    df = pd.DataFrame(
        {
            "A": [1.0, 1.2, 1.5, 2.2, 2.7, 2.9, 3.5, 3.2],
            "B": [0, 0, 0, 0, 1, 1, 1, 1],
            "C": [0, 0, 0, 1, 0, 1, 1, 1],
            "D": [1, 0, 0, 1, 0, 1, 0, 1],
        }
    )
    continuous_df = df[["A"]]
    categorical_df = df[["B", "C", "D"]]

    expected_df = pd.DataFrame(
        {
            "A": [1.0, 1.2, 1.5, 2.2, 2.7, 2.9, 3.5, 3.2],
            "B": [0, 0, 0, 0, 1, 1, 1, 1],
            "C": [0, 0, 0, 1, 0, 1, 1, 1],
        }
    )

    with patch("redxregressors.preprocess.sns.heatmap") as mock_heatmap:
        result = remove_low_correlation_binary_categorical_and_continous_target(
            df=df, threshold=0.5
        )

        assert isinstance(result, pd.DataFrame)
        mock_heatmap.assert_called_once()

    with pytest.raises(ValueError):
        remove_low_correlation_binary_categorical_and_continous_target()

    with pytest.raises(RuntimeError):
        df["E"] = [1, 2, 3, 4, 2, 3, 4, 5]
        remove_low_correlation_binary_categorical_and_continous_target(df=df)

    result = remove_low_correlation_binary_categorical_and_continous_target(
        threshold=0.5,
        continuous_df=continuous_df,
        categorical_df=categorical_df,
        annotation=True,
    )

    log.error(result)
    pd.testing.assert_frame_equal(result, expected_df)


if __name__ == "__main__":
    pytest.main()
