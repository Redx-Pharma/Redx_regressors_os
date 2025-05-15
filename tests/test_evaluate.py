#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit tests for evaluation
"""

import pytest
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from redxregressors.evaluate import (
    rmse,
    calculate_regression_metrics,
    get_regression_metric_table,
)

log = logging.getLogger(__name__)


def test_rmse():
    exp = np.array([1, 2, 3])
    pred = np.array([1, 2, 3])
    expected_output = 0.0
    assert rmse(exp, pred) == expected_output

    exp = np.array([1, 2, 3])
    pred = np.array([4, 5, 6])
    expected_output = np.sqrt(mean_squared_error(exp, pred))
    assert rmse(exp, pred) == expected_output


def test_calculate_regression_metrics():
    exp_array = np.array([1, 2, 3])
    pred_array = np.array([1, 2, 3])
    expected_output = {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 1.0, "mape": 0.0}
    assert calculate_regression_metrics(exp_array, pred_array) == expected_output

    exp_array = np.array([1, 2, 3])
    pred_array = np.array([4, 5, 6])
    expected_output = {
        "mse": mean_squared_error(exp_array, pred_array),
        "rmse": np.sqrt(mean_squared_error(exp_array, pred_array)),
        "mae": mean_absolute_error(exp_array, pred_array),
        "r2": r2_score(exp_array, pred_array),
        "mape": mean_absolute_percentage_error(exp_array, pred_array),
    }
    assert calculate_regression_metrics(exp_array, pred_array) == expected_output


def test_get_regression_metric_table():
    df = pd.DataFrame({"y": [1, 2, 3], "y_pred": [1, 2, 3]})
    expected_output = pd.DataFrame(
        {"mse": [0.0], "rmse": [0.0], "mae": [0.0], "r2": [1.0], "mape": [0.0]}
    )
    pd.testing.assert_frame_equal(get_regression_metric_table(df), expected_output)


if __name__ == "__main__":
    pytest.main()
