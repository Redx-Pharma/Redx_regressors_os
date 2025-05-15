#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit tests for featurization
"""

import pytest
import pandas as pd
import numpy as np
from rdkit.DataStructs import cDataStructs
from redxregressors import ml_featurization
import logging

log = logging.getLogger(__name__)


def test_list_of_bitvects_to_numpy_arrays():
    bitvect = cDataStructs.CreateFromBitString("1011")
    result = ml_featurization.list_of_bitvects_to_numpy_arrays([bitvect])
    expected = np.array([[1, 0, 1, 1]], dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_list_of_bitvects_to_list_of_lists():
    bitvect = cDataStructs.CreateFromBitString("1011")
    result = ml_featurization.list_of_bitvects_to_list_of_lists([bitvect])
    expected = [[1, 0, 1, 1]]
    assert result == expected


def test_bitstring_to_bit_vect():
    bstring = "10101010001101"
    result = ml_featurization.bitstring_to_bit_vect(bstring)
    assert isinstance(result, cDataStructs.ExplicitBitVect)


def test_df_rows_to_list_of_bit_vect():
    df = pd.DataFrame([[1, 0, 1, 0, 1, 1, 1, 1]])
    result = ml_featurization.df_rows_to_list_of_bit_vect(df)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], cDataStructs.ExplicitBitVect)


def test_validate_smiles_and_get_ecfp_bitvect():
    result = ml_featurization.validate_smiles_and_get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024
    )
    assert isinstance(result, list)
    assert isinstance(result[0], cDataStructs.ExplicitBitVect)


def test_validate_smiles_and_get_ecfp_numpy():
    result = ml_featurization.validate_smiles_and_get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_np=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


def test_validate_smiles_and_get_ecfp_dataframe():
    out_df = ml_featurization.validate_smiles_and_get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "ecfp_bit_0"
    assert int(out_df.loc[0, "ecfp_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_bit_175"]) == 1


def test_validate_smiles_and_get_ecfp_dataframe_with_input_df():
    data_df = pd.DataFrame([["toluene", "c1ccccc1C"]], columns=["id", "smiles"])
    out_df = ml_featurization.validate_smiles_and_get_ecfp(
        data_df=data_df, smiles_column="smiles", hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "id"
    assert int(out_df.loc[0, "ecfp_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_bit_175"]) == 1


def test_get_ecfp_bitvect():
    result = ml_featurization.get_ecfp(smiles=["c1ccccc1C"], hash_length=1024)
    assert isinstance(result, tuple)
    assert isinstance(result[0], cDataStructs.ExplicitBitVect)


def test_get_ecfp_numpy():
    result = ml_featurization.get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_np=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


def test_get_ecfp_dataframe():
    out_df = ml_featurization.get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "ecfp_bit_0"
    assert int(out_df.loc[0, "ecfp_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_bit_175"]) == 1


def test_get_ecfp_dataframe_with_input_df():
    data_df = pd.DataFrame([["toluene", "c1ccccc1C"]], columns=["id", "smiles"])
    out_df = ml_featurization.get_ecfp(
        data_df=data_df, smiles_column="smiles", hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "id"
    assert int(out_df.loc[0, "ecfp_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_bit_175"]) == 1


def test_get_count_ecfp_bitvect():
    result = ml_featurization.get_count_ecfp(smiles=["c1ccccc1C"], hash_length=1024)
    assert isinstance(result, tuple)
    assert isinstance(result[0], cDataStructs.UIntSparseIntVect)


def test_get_count_ecfp_numpy():
    result = ml_featurization.get_count_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_np=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


def test_get_count_ecfp_dataframe():
    out_df = ml_featurization.get_count_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "ecfp_count_bit_0"
    assert int(out_df.loc[0, "ecfp_count_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_count_bit_175"]) == 2


def test_get_count_ecfp_dataframe_with_input_df():
    data_df = pd.DataFrame([["toluene", "c1ccccc1C"]], columns=["id", "smiles"])
    out_df = ml_featurization.get_count_ecfp(
        data_df=data_df, smiles_column="smiles", hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "id"
    assert int(out_df.loc[0, "ecfp_count_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_count_bit_175"]) == 2


def test_get_maccs_bitvect():
    result = ml_featurization.get_maccs(smiles=["c1ccccc1C"])
    assert isinstance(result, list)
    assert isinstance(result[0], cDataStructs.ExplicitBitVect)


def test_get_maccs_bitvect_on_bits():
    vecs = ml_featurization.get_maccs(smiles=["c1ccccc1C"])  ##### CHECK THIS ONE
    assert tuple(vecs[0].GetOnBits()) == (160, 162, 163, 165)


def test_get_maccs_dataframe():
    out_df = ml_featurization.get_maccs(smiles=["c1ccccc1C"], return_df=True)
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "maccs_bit_0"
    assert int(out_df.loc[0, "maccs_bit_0"]) == 0
    assert int(out_df.loc[0, "maccs_bit_163"]) == 1


def test_get_maccs_dataframe_with_input_df():
    data_df = pd.DataFrame([["toluene", "c1ccccc1C"]], columns=["id", "smiles"])
    out_df = ml_featurization.get_maccs(
        data_df=data_df, smiles_column="smiles", return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "id"
    assert int(out_df.loc[0, "maccs_bit_0"]) == 0
    assert int(out_df.loc[0, "maccs_bit_163"]) == 1


def test_get_rdkit_descriptors_dict():
    des = ml_featurization.get_rdkit_descriptors(smiles=["c1ccccc1C"])
    assert isinstance(des[0], dict)


def test_get_rdkit_descriptors_dataframe():
    out_df = ml_featurization.get_rdkit_descriptors(
        smiles=["c1ccccc1C"], return_df=True
    )
    assert len(out_df.columns) == 217
    assert round(float(out_df.loc[0, "rdkit_descriptor_Chi4n"]), 4) == 0.5344
    assert round(float(out_df.loc[0, "rdkit_descriptor_BCUT2D_MWLOW"]), 4) == 10.2446


def test_get_nlp_smiles_rep_return_df():
    # Sample data
    data = {"smiles": ["c1ccccc1", "CCN"]}
    df = pd.DataFrame(data)

    # Call the function
    result = ml_featurization.get_nlp_smiles_rep(
        data_df=df, smiles_column="smiles", return_df=True
    )

    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert (
        "embedding_Saideepthi55-sentencetransformer_ftmodel_on_chemical_dataset_0"
        in result.columns
    )
    assert np.isclose(
        result.loc[
            0,
            "embedding_Saideepthi55-sentencetransformer_ftmodel_on_chemical_dataset_0",
        ],
        -0.07106573,
    )
    # feature and smiles column
    assert len(result.columns) == 769


def test_get_nlp_smiles_rep_return_np():
    # Sample data
    smiles = ["c1ccccc1", "CCN"]

    # Call the function
    result = ml_featurization.get_nlp_smiles_rep(smiles=smiles, return_np=True)

    # Assertions
    assert isinstance(result, np.ndarray)
    assert np.isclose(result[0, 0], -0.07106573)
    assert result.shape[0] == len(smiles)
    assert result.shape[1] == 768


def test_get_nlp_smiles_rep_invalid_input():
    # Sample data
    data = {"smiles": ["CCO", "CCN", "CCC"]}
    df = pd.DataFrame(data)

    # Call the function with invalid input
    with pytest.raises(RuntimeError):
        ml_featurization.get_nlp_smiles_rep(
            data_df=df, smiles_column="smiles", return_df=True, return_np=True
        )

    with pytest.raises(RuntimeError):
        ml_featurization.get_nlp_smiles_rep(
            data_df=None, smiles_column=None, smiles=None
        )

    data = {"smiles": ["CCO", "CCN", "NOT_A_VALID_SMILES"]}
    df = pd.DataFrame(data)
    with pytest.raises(RuntimeError):
        ml_featurization.get_nlp_smiles_rep(data_df=data, smiles_column="smiles")


# T5 model too large for GitHub runners with the library
# def test_get_t5_smiles_rep_return_df():
#     # Sample data
#     data = {"smiles": ["c1ccccc1", "CCN"]}
#     df = pd.DataFrame(data)

#     # Call the function
#     result = ml_featurization.get_t5_smiles_rep(
#         data_df=df, smiles_column="smiles", return_df=True
#     )

#     # Assertions
#     assert isinstance(result, pd.DataFrame)
#     assert "embedding_laituan245-molt5-large-smiles2caption_0" in result.columns
#     assert np.isclose(
#         result.loc[0, "embedding_laituan245-molt5-large-smiles2caption_0"], 0.006745348
#     )
#     # feature and smiles column
#     assert len(result.columns) == 1025


# def test_get_t5_smiles_rep_return_np():
#     # Sample data
#     smiles = ["c1ccccc1", "CCN"]

#     # Call the function
#     result = ml_featurization.get_t5_smiles_rep(smiles=smiles, return_np=True)

#     # Assertions
#     assert isinstance(result, np.ndarray)
#     assert np.isclose(result[0, 0], 0.006745348)
#     assert result.shape[0] == len(smiles)
#     assert result.shape[1] == 1024


# def test_get_t5_smiles_rep_invalid_input():
#     # Sample data
#     data = {"smiles": ["CCO", "CCN", "CCC"]}
#     df = pd.DataFrame(data)

#     # Call the function with invalid input
#     with pytest.raises(RuntimeError):
#         ml_featurization.get_t5_smiles_rep(
#             data_df=df, smiles_column="smiles", return_df=True, return_np=True
#         )

#     with pytest.raises(RuntimeError):
#         ml_featurization.get_t5_smiles_rep(
#             data_df=None, smiles_column=None, smiles=None
#         )

#     data = {"smiles": ["CCO", "CCN", "NOT_A_VALID_SMILES"]}
#     df = pd.DataFrame(data)
#     with pytest.raises(RuntimeError):
#         ml_featurization.get_t5_smiles_rep(data_df=data, smiles_column="smiles")


# Run the tests
if __name__ == "__main__":
    pytest.main()
