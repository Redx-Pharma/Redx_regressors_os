import logging
import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors
from redxregressors import applicability_domain, ml_featurization

log = logging.getLogger(__name__)


def test_tanimoto_ad() -> None:
    training_smiles = ["CCO", "CCN", "CCC"]
    test_smiles = ["CCO", "CCCN", "CCCC"]
    tanimoto_threshold = 0.7
    expected_output = np.array([True, False, False])
    assert np.array_equal(
        applicability_domain.tanimoto_ad(
            training_smiles, test_smiles, tanimoto_threshold
        ),
        expected_output,
    )

    # Test case 2: All test molecules are similar to training molecules
    training_smiles = ["CCO", "CCN", "CCC"]
    test_smiles = ["CCO", "CCN", "CCC"]
    tanimoto_threshold = 0.7
    expected_output = np.array([True, True, True])
    assert np.array_equal(
        applicability_domain.tanimoto_ad(
            training_smiles, test_smiles, tanimoto_threshold
        ),
        expected_output,
    )

    # Test case 3: No test molecules are similar to training molecules
    training_smiles = ["CCO", "CCN", "CCC"]
    test_smiles = ["C#N", "C#C", "c1ccccc1"]
    tanimoto_threshold = 0.7
    expected_output = np.array([False, False, False])
    assert np.array_equal(
        applicability_domain.tanimoto_ad(
            training_smiles, test_smiles, tanimoto_threshold
        ),
        expected_output,
    )

    # Test case 4: Using minimum_fraction_similar parameter with one similar in minimum fraction
    training_smiles = ["CCO", "CCN", "CCC"]
    test_smiles = ["CCO", "CCCN", "CCCC"]
    tanimoto_threshold = 0.7
    minimum_fraction_similar = 0.3
    print(
        applicability_domain.tanimoto_ad(
            training_smiles,
            test_smiles,
            tanimoto_threshold,
            minimum_fraction_similar=minimum_fraction_similar,
        )
    )
    expected_output = np.array([True, False, False])
    assert np.array_equal(
        applicability_domain.tanimoto_ad(
            training_smiles,
            test_smiles,
            tanimoto_threshold,
            minimum_fraction_similar=minimum_fraction_similar,
        ),
        expected_output,
    )

    # Test case 5: Using minimum_fraction_similar parameter with no similar in minimim fraction
    training_smiles = ["CCO", "CCN", "CCC"]
    test_smiles = ["CCO", "CCCN", "CCCC"]
    tanimoto_threshold = 0.7
    minimum_fraction_similar = 0.5
    print(
        applicability_domain.tanimoto_ad(
            training_smiles,
            test_smiles,
            tanimoto_threshold,
            minimum_fraction_similar=minimum_fraction_similar,
        )
    )
    expected_output = np.array([False, False, False])
    assert np.array_equal(
        applicability_domain.tanimoto_ad(
            training_smiles,
            test_smiles,
            tanimoto_threshold,
            minimum_fraction_similar=minimum_fraction_similar,
        ),
        expected_output,
    )


def test_get_tanimoto_ad_model() -> None:
    train_smiles = ["c1ccccc1", "c1ccccc1C", "c1ccccc1N", "c1ccccc1F", "c1ccccc1Cl"]
    test_smiles = ["c1ccccc1Br", "c1ccccc1I", "C1CCCCC1C", "CCCCN", "C(CCN)CC(=O)O"]

    # Runs an internal test for the model producing equivalent results to explicit calculation if the NearstNeighbors model is returned this is passed
    m = applicability_domain.get_tanimoto_ad_model(
        train_smiles, test_smiles=test_smiles, hash_length=1024, radius=2
    )
    assert isinstance(m, NearestNeighbors)

    # misspecified algorithm should be kd_tree not kdtree
    with pytest.raises(ValueError):
        _ = applicability_domain.get_tanimoto_ad_model(
            train_smiles,
            test_smiles=test_smiles,
            hash_length=1024,
            radius=2,
            algorithm="kdtree",
        )


def test_get_ad_model() -> None:
    train_smiles = ["c1ccccc1", "c1ccccc1C", "c1ccccc1N", "c1ccccc1F", "c1ccccc1Cl"]
    test_smiles = ["c1ccccc1Br", "c1ccccc1I", "C1CCCCC1C", "CCCCN", "C(CCN)CC(=O)O"]
    test_features = ml_featurization.get_ecfp(
        smiles=test_smiles, hash_length=1024, return_df=True
    )
    exp_jaccard_distance = [0.625, 0.625, 0.95238095, 0.95238095, 0.92]
    exp_euclidean_distances = [
        2.82842712,
        2.82842712,
        3.74165739,
        3.74165739,
        4.12310563,
    ]
    exp_cosine_distances = [0.45454545, 0.45454545, 0.90909091, 0.90909091, 0.84924433]

    # Test case 1: Jaccard distance
    m = applicability_domain.get_ad_model(
        train_smiles, metric="jaccard", hash_length=1024, radius=2
    )
    assert all(
        ent < 1e-5
        for ent in np.abs(
            m.kneighbors(test_features.astype("boolean"), n_neighbors=1)[0].flatten()
            - exp_jaccard_distance
        )
    )

    # Test case 2: Euclidean distance
    m = applicability_domain.get_ad_model(
        train_smiles, metric="euclidean", hash_length=1024, radius=2
    )
    assert all(
        ent < 1e-5
        for ent in np.abs(
            m.kneighbors(test_features, n_neighbors=1)[0].flatten()
            - exp_euclidean_distances
        )
    )

    # Test case 3: Cosine distance
    m = applicability_domain.get_ad_model(
        train_smiles, metric="cosine", hash_length=1024, radius=2
    )
    assert all(
        ent < 1e-5
        for ent in np.abs(
            m.kneighbors(test_features, n_neighbors=1)[0].flatten()
            - exp_cosine_distances
        )
    )


def test_calculate_similarity_ad_from_model():
    train_smiles = [
        "c1ccccc1",
        "c1ccccc1C",
        "c1ccccc1N",
        "c1ccccc1F",
        "c1ccccc1Cl",
        "c1ccccc1CC",
        "c1ccccc1CCI",
    ]
    test_smiles = [
        "c1ccccc1CF",
        "c1ccccc1CF",
        "c1ccccc1Br",
        "c1ccccc1I",
        "C1CCCCC1C",
        "CCCCN",
        "C(CCN)CC(=O)O",
    ]

    m = applicability_domain.get_tanimoto_ad_model(
        train_smiles, test_smiles=test_smiles, hash_length=1024, radius=2
    )
    sims, _ = applicability_domain.calculate_similarity_ad_from_model(
        m, test_smiles, similarity_threshold=0.3, n_neighbours_within_threshold=1
    )
    # log.error(sims)
    # log.error(1.0 - np.array([0.0, 0.0, 0.625, 0.625, 0.95238095, 0.95238095, 0.92]))
    assert np.array_equal(
        sims.flatten(), np.array([True, True, True, True, False, False, False])
    )

    sims, _ = applicability_domain.calculate_similarity_ad_from_model(
        m, test_smiles, similarity_threshold=0.4, n_neighbours_within_threshold=2
    )
    log.error(sims)
    assert np.array_equal(
        sims.flatten(), np.array([True, True, False, False, False, False, False])
    )

    m = applicability_domain.get_ad_model(
        train_smiles, metric="cosine", hash_length=1024, radius=2
    )
    sims, _ = applicability_domain.calculate_similarity_ad_from_model(
        m, test_smiles, similarity_threshold=0.5, n_neighbours_within_threshold=1
    )
    log.error(sims)
    assert np.array_equal(
        sims.flatten(), np.array([True, True, True, True, False, False, False])
    )
