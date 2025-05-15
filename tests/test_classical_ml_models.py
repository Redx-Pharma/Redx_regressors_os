#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit test
"""

import logging

import pandas as pd
import pytest

from redxregressors import classical_ml_models

log = logging.getLogger(__name__)


number_of_defualt_models = 17
number_of_defualt_linear_models = 3
number_of_defualt_kernel_models = 4
number_of_defualt_bayesian_models = 3
number_of_defualt_ensemble_models = 6
number_of_defualt_neural_network_models = 1


@pytest.fixture
def pandas_dataframe() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    test_file = pd.DataFrame(
        [
            ["c1ccccc1", "benzene", 0.5, 1.0, 0.25, 1.2],
            ["C1CCCC1C(N)C", "1-cyclopentylethanamine", 0.9, 0.1, 1.2, 0.9],
            ["C1CCCC1C(=O)C", "1-cyclopentylethanone", 0.75, 0.05, 1.2, 0.9],
            ["C1CCCC1C(O)C", "1-cyclopentylethanol", 0.95, 0.12, 1.2, 0.9],
            ["C1CCCCC1C(N)C", "1-cyclohexylethanamine", 0.95, 0.15, 1.22, 0.95],
            ["C1CCCCC1C(=O)C", "1-cyclohexylethanone", 0.79, 0.02, 1.24, 0.97],
            ["C1CCCCC1C(O)C", "1-cyclohexylethanol", 1.1, 1.2, 1.4, 0.95],
            ["NCc1ccccc1", "benzylamine", 1.2, 0.02, 2.2, 0.75],
            ["C", "methane", -1.2, 0.01, 0.02, -10.0],
            ["CC", "ethane", -1.0, 0.2, 0.07, -10.2],
            ["CCC", "propane", -1.0, -0.4, 0.1, -10.7],
            ["CCCC", "butane", -0.7, -0.9, 0.2, -11.0],
        ],
        columns=["smiles", "names", "bind_target_0", "bind_target_1", "tox", "sol"],
    )
    return test_file


def test_get_all_default_models():
    """
    Test the codes can get all of the default models
    """
    all_default_models = classical_ml_models.get_models()

    assert len(all_default_models) == number_of_defualt_models


def test_get_linear_default_models():
    """
    Test the codes can get all of the linear models
    """
    models = classical_ml_models.linear_models()

    assert len(models) == number_of_defualt_linear_models


def test_get_kernel_default_models():
    """
    Test the codes can get all of the kernel models
    """
    models = classical_ml_models.kernel_models()

    assert len(models) == number_of_defualt_kernel_models


def test_get_bayesian_default_models():
    """
    Test the codes can get all of the Bayesian models
    """
    models = classical_ml_models.bayesian_models()

    assert len(models) == number_of_defualt_bayesian_models


def test_get_ensemble_default_models():
    """
    Test the codes can get all of the ensemble models
    """
    models = classical_ml_models.ensemble_models()

    assert len(models) == number_of_defualt_ensemble_models


def test_get_neural_network_default_models():
    """
    Test the codes can get all of the Bayesian models
    """
    models = classical_ml_models.neural_network_models()

    assert len(models) == number_of_defualt_neural_network_models
