# Module redxregressors.applicability_domain

Module for calculating a models applicability domain. Please note there is no one way to calculate this and it is a choice how we define this.

The methods here are based on the Tanimoto similarity and the Jaccard distance of the ECFP fingerprints of the training set and the test set.
The domain is therefore defined as the set of molecules that are within a certain similarity threshold of the training set.

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module for calculating a models applicability domain. Please note there is no one way to calculate this and it is a choice how we define this.

        The methods here are based on the Tanimoto similarity and the Jaccard distance of the ECFP fingerprints of the training set and the test set.

        The domain is therefore defined as the set of molecules that are within a certain similarity threshold of the training set.

        """

        import logging

        import os

        from typing import Union, List, Optional, Literal

        import numpy as np

        from numpy.typing import ArrayLike

        import pandas as pd

        from rdkit import Chem

        from rdkit.Chem import AllChem

        from rdkit.Chem import DataStructs

        from sklearn.neighbors import NearestNeighbors

        from redxregressors import ml_featurization

        from scipy.spatial.distance import jaccard

        log = logging.getLogger(__name__)



        def tanimoto_ad(

            training_smiles: List[str],

            test_smiles: List[str],

            tanimoto_threshold: float = 0.7,

            radius: int = 2,

            hash_length: int = 1024,

            minimum_fraction_similar: Union[float, None] = None,

            **kwargs,

        ) -> ArrayLike:

            """

            Function to calculate the applicability domain of a model using Tanimoto similarity

            Args:

                training_smiles (Iterable): An iterable of SMILES strings for the training set

                test_smiles (Iterable): An iterable of SMILES strings for the test set

                tanimoto_threshold (float): The threshold for the Tanimoto similarity

                radius (int): The radius for the Morgan fingerprint

                hash_length (int): The length of the hash for the fingerprint

            Returns:

                ArrayLike: A boolean array of whether the test set is within the threshold

            """

            # Convert SMILES to RDKit molecules

            training_mols = [Chem.MolFromSmiles(smile) for smile in training_smiles]

            test_mols = [Chem.MolFromSmiles(smile) for smile in test_smiles]

            ecpf_generator = AllChem.GetMorganGenerator(

                radius=radius, fpSize=hash_length, **kwargs

            )

            # Convert molecules to fingerprints

            training_fps = ecpf_generator.GetFingerprints(training_mols, numThreads=4)

            test_fps = ecpf_generator.GetFingerprints(test_mols, numThreads=4)

            # Calculate bulk Tanimoto similarity

            similarity_matrix = np.zeros((len(test_fps), len(training_fps)))

            for ith, test_fp in enumerate(test_fps):

                similarities = DataStructs.BulkTanimotoSimilarity(test_fp, training_fps)

                similarity_matrix[ith, :] = similarities

            log.debug(f"Tanimoto similarity matrix: {similarity_matrix}")

            # Determine if each test molecule is within the threshold

            if minimum_fraction_similar is None:

                within_threshold = np.any(similarity_matrix >= tanimoto_threshold, axis=1)

            else:

                fraction_similar = np.sum(

                    similarity_matrix >= tanimoto_threshold, axis=1

                ) / len(training_fps)

                within_threshold = fraction_similar >= minimum_fraction_similar

                log.debug(f"In applicability domain Fraction similar: {fraction_similar}")

            log.debug(f"In applicability domain within threshold: {within_threshold}")

            return within_threshold



        def get_tanimoto_ad_model(

            training_smiles: List[str],

            test_smiles: Optional[List[str]] = None,

            radius: int = 2,

            hash_length: int = 1024,

            algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "brute",

            **kwargs,

        ) -> NearestNeighbors:

            """

            Function to calculate the applicability domain of a model using Tanimoto/Jaccard distance

            Args:

                training_smiles (List[str]): the list of training smiles

                test_smiles (List[str]): the list of test smiles

                radius (int): the radius of the ECFP

                hash_length (int): the length of the hash for the ECFP

                algorithm (str): the algorithm to use for the NearestNeighbors

                **kwargs: the keyword arguments for the NearestNeighbors

            Returns:

                NearestNeighbors: the NearestNeighbors model

            """

            # Convert SMILES to RDKit molecules

            ecfp_reps = ml_featurization.get_ecfp(

                smiles=training_smiles, hash_length=hash_length, radius=radius, return_df=True

            )

            # fit the model, note we use the brute force method by default this should be fine for small datasets and gives the extact answer

            nn_model = NearestNeighbors(

                n_neighbors=5, metric="jaccard", algorithm=algorithm, **kwargs

            ).fit(ecfp_reps.astype("boolean"))

            if test_smiles is not None:

                test_ecfp_reps = ml_featurization.get_ecfp(

                    smiles=test_smiles, hash_length=hash_length, radius=radius, return_df=True

                )

                model_distances, model_closest_indexes = nn_model.kneighbors(

                    test_ecfp_reps.astype("boolean"), n_neighbors=1

                )

                closest_index = np.ones(test_ecfp_reps.shape[0], dtype=int)

                closest_distance = np.ones(test_ecfp_reps.shape[0])

                for ith, (_, test_ecfp) in enumerate(test_ecfp_reps.iterrows()):

                    for jth, (_, train_ecfp) in enumerate(ecfp_reps.iterrows()):

                        jaccard_dist = jaccard(

                            test_ecfp.values.astype(bool),

                            train_ecfp.values.astype(bool),

                        )

                        if jaccard_dist < closest_distance[ith]:

                            closest_distance[ith] = jaccard_dist

                            closest_index[ith] = int(jth)

                dont_match = 0

                for ith, (md, mi, cd, ci) in enumerate(

                    zip(model_distances, model_closest_indexes, closest_distance, closest_index)

                ):

                    log.debug(

                        f"Model distances index {ith}: {md} Model indexes: {mi} Closest index: {ci} Closest distance: {cd}"

                    )

                    if abs(md[0] - cd) > 1e-5 or mi[0] != ci:

                        log.warning(

                            "Model distances and closest index do not match the brute force method"

                        )

                        log.warning(

                            f"Model distances index {ith}: {md[0]} Model indexes: {mi[0]} Closest index: {ci} Closest distance: {cd}"

                        )

                        dont_match += 1

                if dont_match == 0:

                    log.info(

                        "Model distances and closest index from test set match the brute force and explicit methods"

                    )

                else:

                    # log.error(f"model_distances: {model_distances} {model_distances.dtype} {os.linesep}model_closest_indexes: {model_closest_indexes} {model_closest_indexes.dtype} {os.linesep}closest_distance: {closest_distance} {closest_distance.dtype}{os.linesep}closest_index: {closest_index}{closest_index.dtype}")

                    err_df = pd.DataFrame(

                        np.array(

                            [

                                [ent[0] for ent in model_distances],

                                [ent[0] for ent in model_closest_indexes],

                                closest_distance,

                                closest_index,

                            ]

                        ).T,

                        columns=[

                            "Model Dist",

                            "Model Inx",

                            "Exact Dist",

                            "Exact Inx",

                        ],

                    )

                    log.error(f"{os.linesep}{err_df}")

                    raise ValueError(

                        "Model distances and closest index from test set do not match the brute force and explicit methods"

                    )

            return nn_model



        def get_ad_model(

            training_smiles: List[str],

            metric: str = "jaccard",

            radius: int = 2,

            hash_length: int = 1024,

            algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "brute",

            **kwargs,

        ) -> NearestNeighbors:

            """

            Function to calculate the applicability domain of a model using Tanimoto/Jaccard distance

            Args:

                training_smiles (Iterable): the list of training smiles

                metric (str): the distance metric to use for the NearestNeighbors

                radius (int): the radius of the ECFP

                hash_length (int): the length of the hash for the ECFP

                algorithm (str): the algorithm to use for the NearestNeighbors

                **kwargs: the keyword arguments for the NearestNeighbors

            Returns:

                NearestNeighbors: the NearestNeighbors model

            """

            # Convert SMILES to RDKit molecules

            ecfp_reps = ml_featurization.get_ecfp(

                smiles=training_smiles, hash_length=hash_length, radius=radius, return_df=True

            )

            if metric == "jaccard":

                ecfp_reps = ecfp_reps.astype("boolean")

            # fit the model, note we use the brute force method by default this should be fine for small datasets and gives the extact answer

            nn_model = NearestNeighbors(

                n_neighbors=5, metric=metric, algorithm=algorithm, **kwargs

            ).fit(ecfp_reps)

            return nn_model



        def calculate_similarity_ad_from_model(

            nn_model: NearestNeighbors,

            test_smiles: List[str],

            radius: int = 2,

            hash_length: int = 1024,

            similarity_threshold: float = 0.7,

            n_neighbours_within_threshold: int = 1,

            **kwargs,

        ) -> tuple[ArrayLike, ArrayLike]:

            """

            Function to take a trained model and a test set and calculate the applicability domain

            Args:

                nn_model (NearestNeighbors): the fitted NearestNeighbors model

                test_smiles (List[str]): the list of test smiles

                radius (int): the radius of the ECFP

                hash_length (int): the length of the hash for the ECFP

                similarity_threshold (float): the similarity threshold this is the minimum similarity to be considered within the applicability domain

                n_neighbours_within_threshold (int): the number of neighbours within the threshold

                **kwargs: the keyword arguments for the NearestNeighbors

            Returns:

                tuple[ArrayLike, ArrayLike]: a tuple of the boolean array of whether the test set is within the threshold and the indexes of the closest neighbours

            """

            # Convert tests set smiles to ECFP

            test_ecfp_reps = ml_featurization.get_ecfp(

                smiles=test_smiles, hash_length=hash_length, radius=radius, return_df=True

            )

            # Calculate the nearest neighbours using the nerest neighbours model

            model_distances, model_closest_indexes = nn_model.kneighbors(

                test_ecfp_reps.astype(bool), n_neighbors=n_neighbours_within_threshold

            )

            # convert distances to similarity (similarity = 1.0 - distance)

            similarity = 1.0 - model_distances

            log.debug(f"Similarity: {similarity}")

            # return boolean array of whether the test set is within the threshold

            log.debug(f"Similarity threshold: {similarity_threshold}")

            if n_neighbours_within_threshold == 1:

                within_threshold = similarity >= similarity_threshold

                log.debug(f"Within threshold: {within_threshold}")

                return within_threshold, model_closest_indexes

            else:

                within_threshold = similarity >= similarity_threshold

                log.debug(f"Within threshold: {within_threshold}")

                n_within_threshold = (

                    within_threshold.sum(axis=1) >= n_neighbours_within_threshold

                )

                log.debug(f"n_within_threshold: {n_within_threshold}")

                return n_within_threshold, model_closest_indexes



        def calculate_distance_ad_from_model(

            nn_model: NearestNeighbors,

            test_smiles: List[str],

            radius: int = 2,

            hash_length: int = 1024,

            distance_threshold: float = 0.7,

            n_neighbours_within_threshold: int = 1,

            **kwargs,

        ) -> tuple[ArrayLike, ArrayLike]:

            """

            Function to take a trained model and a test set and calculate the applicability domain

            Args:

                nn_model (NearestNeighbors): the fitted NearestNeighbors model

                test_smiles (List[str]): the list of test smiles

                radius (int): the radius of the ECFP

                hash_length (int): the length of the hash for the ECFP

                distance_threshold (float): the distance threshold for the NearestNeighbors model this is the maximum distance to be considered within ththe appicability domain

                n_neighbours_within_threshold (int): the number of neighbours within the threshold

                **kwargs: the keyword arguments for the NearestNeighbors

            Returns:

                tuple[ArrayLike, ArrayLike]: a tuple of the boolean array of whether the test set is within the threshold and the indexes of the closest neighbours

            """

            # Convert tests set smiles to ECFP

            test_ecfp_reps = ml_featurization.get_ecfp(

                smiles=test_smiles, hash_length=hash_length, radius=radius, return_df=True

            )

            # Calculate the nearest neighbours using the nerest neighbours model

            model_distances, model_closest_indexes = nn_model.kneighbors(

                test_ecfp_reps, n_neighbors=n_neighbours_within_threshold

            )

            # return boolean array of whether the test set is within the threshold

            if n_neighbours_within_threshold == 1:

                within_threshold = model_distances <= distance_threshold

                return within_threshold, model_closest_indexes

            else:

                within_threshold = model_distances <= distance_threshold

                n_within_threshold = (

                    within_threshold.sum(axis=1) >= n_neighbours_within_threshold

                )

                return n_within_threshold, model_closest_indexes



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### calculate_distance_ad_from_model

```python3
def calculate_distance_ad_from_model(
    nn_model: sklearn.neighbors._unsupervised.NearestNeighbors,
    test_smiles: List[str],
    radius: int = 2,
    hash_length: int = 1024,
    distance_threshold: float = 0.7,
    n_neighbours_within_threshold: int = 1,
    **kwargs
) -> tuple[typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]], typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]]]
```

Function to take a trained model and a test set and calculate the applicability domain

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| nn_model | NearestNeighbors | the fitted NearestNeighbors model | None |
| test_smiles | List[str] | the list of test smiles | None |
| radius | int | the radius of the ECFP | None |
| hash_length | int | the length of the hash for the ECFP | None |
| distance_threshold | float | the distance threshold for the NearestNeighbors model this is the maximum distance to be considered within ththe appicability domain | None |
| n_neighbours_within_threshold | int | the number of neighbours within the threshold | None |
| **kwargs | None | the keyword arguments for the NearestNeighbors | None |

**Returns:**

| Type | Description |
|---|---|
| tuple[ArrayLike, ArrayLike] | a tuple of the boolean array of whether the test set is within the threshold and the indexes of the closest neighbours |

??? example "View Source"
        def calculate_distance_ad_from_model(

            nn_model: NearestNeighbors,

            test_smiles: List[str],

            radius: int = 2,

            hash_length: int = 1024,

            distance_threshold: float = 0.7,

            n_neighbours_within_threshold: int = 1,

            **kwargs,

        ) -> tuple[ArrayLike, ArrayLike]:

            """

            Function to take a trained model and a test set and calculate the applicability domain

            Args:

                nn_model (NearestNeighbors): the fitted NearestNeighbors model

                test_smiles (List[str]): the list of test smiles

                radius (int): the radius of the ECFP

                hash_length (int): the length of the hash for the ECFP

                distance_threshold (float): the distance threshold for the NearestNeighbors model this is the maximum distance to be considered within ththe appicability domain

                n_neighbours_within_threshold (int): the number of neighbours within the threshold

                **kwargs: the keyword arguments for the NearestNeighbors

            Returns:

                tuple[ArrayLike, ArrayLike]: a tuple of the boolean array of whether the test set is within the threshold and the indexes of the closest neighbours

            """

            # Convert tests set smiles to ECFP

            test_ecfp_reps = ml_featurization.get_ecfp(

                smiles=test_smiles, hash_length=hash_length, radius=radius, return_df=True

            )

            # Calculate the nearest neighbours using the nerest neighbours model

            model_distances, model_closest_indexes = nn_model.kneighbors(

                test_ecfp_reps, n_neighbors=n_neighbours_within_threshold

            )

            # return boolean array of whether the test set is within the threshold

            if n_neighbours_within_threshold == 1:

                within_threshold = model_distances <= distance_threshold

                return within_threshold, model_closest_indexes

            else:

                within_threshold = model_distances <= distance_threshold

                n_within_threshold = (

                    within_threshold.sum(axis=1) >= n_neighbours_within_threshold

                )

                return n_within_threshold, model_closest_indexes


### calculate_similarity_ad_from_model

```python3
def calculate_similarity_ad_from_model(
    nn_model: sklearn.neighbors._unsupervised.NearestNeighbors,
    test_smiles: List[str],
    radius: int = 2,
    hash_length: int = 1024,
    similarity_threshold: float = 0.7,
    n_neighbours_within_threshold: int = 1,
    **kwargs
) -> tuple[typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]], typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]]]
```

Function to take a trained model and a test set and calculate the applicability domain

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| nn_model | NearestNeighbors | the fitted NearestNeighbors model | None |
| test_smiles | List[str] | the list of test smiles | None |
| radius | int | the radius of the ECFP | None |
| hash_length | int | the length of the hash for the ECFP | None |
| similarity_threshold | float | the similarity threshold this is the minimum similarity to be considered within the applicability domain | None |
| n_neighbours_within_threshold | int | the number of neighbours within the threshold | None |
| **kwargs | None | the keyword arguments for the NearestNeighbors | None |

**Returns:**

| Type | Description |
|---|---|
| tuple[ArrayLike, ArrayLike] | a tuple of the boolean array of whether the test set is within the threshold and the indexes of the closest neighbours |

??? example "View Source"
        def calculate_similarity_ad_from_model(

            nn_model: NearestNeighbors,

            test_smiles: List[str],

            radius: int = 2,

            hash_length: int = 1024,

            similarity_threshold: float = 0.7,

            n_neighbours_within_threshold: int = 1,

            **kwargs,

        ) -> tuple[ArrayLike, ArrayLike]:

            """

            Function to take a trained model and a test set and calculate the applicability domain

            Args:

                nn_model (NearestNeighbors): the fitted NearestNeighbors model

                test_smiles (List[str]): the list of test smiles

                radius (int): the radius of the ECFP

                hash_length (int): the length of the hash for the ECFP

                similarity_threshold (float): the similarity threshold this is the minimum similarity to be considered within the applicability domain

                n_neighbours_within_threshold (int): the number of neighbours within the threshold

                **kwargs: the keyword arguments for the NearestNeighbors

            Returns:

                tuple[ArrayLike, ArrayLike]: a tuple of the boolean array of whether the test set is within the threshold and the indexes of the closest neighbours

            """

            # Convert tests set smiles to ECFP

            test_ecfp_reps = ml_featurization.get_ecfp(

                smiles=test_smiles, hash_length=hash_length, radius=radius, return_df=True

            )

            # Calculate the nearest neighbours using the nerest neighbours model

            model_distances, model_closest_indexes = nn_model.kneighbors(

                test_ecfp_reps.astype(bool), n_neighbors=n_neighbours_within_threshold

            )

            # convert distances to similarity (similarity = 1.0 - distance)

            similarity = 1.0 - model_distances

            log.debug(f"Similarity: {similarity}")

            # return boolean array of whether the test set is within the threshold

            log.debug(f"Similarity threshold: {similarity_threshold}")

            if n_neighbours_within_threshold == 1:

                within_threshold = similarity >= similarity_threshold

                log.debug(f"Within threshold: {within_threshold}")

                return within_threshold, model_closest_indexes

            else:

                within_threshold = similarity >= similarity_threshold

                log.debug(f"Within threshold: {within_threshold}")

                n_within_threshold = (

                    within_threshold.sum(axis=1) >= n_neighbours_within_threshold

                )

                log.debug(f"n_within_threshold: {n_within_threshold}")

                return n_within_threshold, model_closest_indexes


### get_ad_model

```python3
def get_ad_model(
    training_smiles: List[str],
    metric: str = 'jaccard',
    radius: int = 2,
    hash_length: int = 1024,
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'brute',
    **kwargs
) -> sklearn.neighbors._unsupervised.NearestNeighbors
```

Function to calculate the applicability domain of a model using Tanimoto/Jaccard distance

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| training_smiles | Iterable | the list of training smiles | None |
| metric | str | the distance metric to use for the NearestNeighbors | None |
| radius | int | the radius of the ECFP | None |
| hash_length | int | the length of the hash for the ECFP | None |
| algorithm | str | the algorithm to use for the NearestNeighbors | None |
| **kwargs | None | the keyword arguments for the NearestNeighbors | None |

**Returns:**

| Type | Description |
|---|---|
| NearestNeighbors | the NearestNeighbors model |

??? example "View Source"
        def get_ad_model(

            training_smiles: List[str],

            metric: str = "jaccard",

            radius: int = 2,

            hash_length: int = 1024,

            algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "brute",

            **kwargs,

        ) -> NearestNeighbors:

            """

            Function to calculate the applicability domain of a model using Tanimoto/Jaccard distance

            Args:

                training_smiles (Iterable): the list of training smiles

                metric (str): the distance metric to use for the NearestNeighbors

                radius (int): the radius of the ECFP

                hash_length (int): the length of the hash for the ECFP

                algorithm (str): the algorithm to use for the NearestNeighbors

                **kwargs: the keyword arguments for the NearestNeighbors

            Returns:

                NearestNeighbors: the NearestNeighbors model

            """

            # Convert SMILES to RDKit molecules

            ecfp_reps = ml_featurization.get_ecfp(

                smiles=training_smiles, hash_length=hash_length, radius=radius, return_df=True

            )

            if metric == "jaccard":

                ecfp_reps = ecfp_reps.astype("boolean")

            # fit the model, note we use the brute force method by default this should be fine for small datasets and gives the extact answer

            nn_model = NearestNeighbors(

                n_neighbors=5, metric=metric, algorithm=algorithm, **kwargs

            ).fit(ecfp_reps)

            return nn_model


### get_tanimoto_ad_model

```python3
def get_tanimoto_ad_model(
    training_smiles: List[str],
    test_smiles: Optional[List[str]] = None,
    radius: int = 2,
    hash_length: int = 1024,
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'brute',
    **kwargs
) -> sklearn.neighbors._unsupervised.NearestNeighbors
```

Function to calculate the applicability domain of a model using Tanimoto/Jaccard distance

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| training_smiles | List[str] | the list of training smiles | None |
| test_smiles | List[str] | the list of test smiles | None |
| radius | int | the radius of the ECFP | None |
| hash_length | int | the length of the hash for the ECFP | None |
| algorithm | str | the algorithm to use for the NearestNeighbors | None |
| **kwargs | None | the keyword arguments for the NearestNeighbors | None |

**Returns:**

| Type | Description |
|---|---|
| NearestNeighbors | the NearestNeighbors model |

??? example "View Source"
        def get_tanimoto_ad_model(

            training_smiles: List[str],

            test_smiles: Optional[List[str]] = None,

            radius: int = 2,

            hash_length: int = 1024,

            algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "brute",

            **kwargs,

        ) -> NearestNeighbors:

            """

            Function to calculate the applicability domain of a model using Tanimoto/Jaccard distance

            Args:

                training_smiles (List[str]): the list of training smiles

                test_smiles (List[str]): the list of test smiles

                radius (int): the radius of the ECFP

                hash_length (int): the length of the hash for the ECFP

                algorithm (str): the algorithm to use for the NearestNeighbors

                **kwargs: the keyword arguments for the NearestNeighbors

            Returns:

                NearestNeighbors: the NearestNeighbors model

            """

            # Convert SMILES to RDKit molecules

            ecfp_reps = ml_featurization.get_ecfp(

                smiles=training_smiles, hash_length=hash_length, radius=radius, return_df=True

            )

            # fit the model, note we use the brute force method by default this should be fine for small datasets and gives the extact answer

            nn_model = NearestNeighbors(

                n_neighbors=5, metric="jaccard", algorithm=algorithm, **kwargs

            ).fit(ecfp_reps.astype("boolean"))

            if test_smiles is not None:

                test_ecfp_reps = ml_featurization.get_ecfp(

                    smiles=test_smiles, hash_length=hash_length, radius=radius, return_df=True

                )

                model_distances, model_closest_indexes = nn_model.kneighbors(

                    test_ecfp_reps.astype("boolean"), n_neighbors=1

                )

                closest_index = np.ones(test_ecfp_reps.shape[0], dtype=int)

                closest_distance = np.ones(test_ecfp_reps.shape[0])

                for ith, (_, test_ecfp) in enumerate(test_ecfp_reps.iterrows()):

                    for jth, (_, train_ecfp) in enumerate(ecfp_reps.iterrows()):

                        jaccard_dist = jaccard(

                            test_ecfp.values.astype(bool),

                            train_ecfp.values.astype(bool),

                        )

                        if jaccard_dist < closest_distance[ith]:

                            closest_distance[ith] = jaccard_dist

                            closest_index[ith] = int(jth)

                dont_match = 0

                for ith, (md, mi, cd, ci) in enumerate(

                    zip(model_distances, model_closest_indexes, closest_distance, closest_index)

                ):

                    log.debug(

                        f"Model distances index {ith}: {md} Model indexes: {mi} Closest index: {ci} Closest distance: {cd}"

                    )

                    if abs(md[0] - cd) > 1e-5 or mi[0] != ci:

                        log.warning(

                            "Model distances and closest index do not match the brute force method"

                        )

                        log.warning(

                            f"Model distances index {ith}: {md[0]} Model indexes: {mi[0]} Closest index: {ci} Closest distance: {cd}"

                        )

                        dont_match += 1

                if dont_match == 0:

                    log.info(

                        "Model distances and closest index from test set match the brute force and explicit methods"

                    )

                else:

                    # log.error(f"model_distances: {model_distances} {model_distances.dtype} {os.linesep}model_closest_indexes: {model_closest_indexes} {model_closest_indexes.dtype} {os.linesep}closest_distance: {closest_distance} {closest_distance.dtype}{os.linesep}closest_index: {closest_index}{closest_index.dtype}")

                    err_df = pd.DataFrame(

                        np.array(

                            [

                                [ent[0] for ent in model_distances],

                                [ent[0] for ent in model_closest_indexes],

                                closest_distance,

                                closest_index,

                            ]

                        ).T,

                        columns=[

                            "Model Dist",

                            "Model Inx",

                            "Exact Dist",

                            "Exact Inx",

                        ],

                    )

                    log.error(f"{os.linesep}{err_df}")

                    raise ValueError(

                        "Model distances and closest index from test set do not match the brute force and explicit methods"

                    )

            return nn_model


### tanimoto_ad

```python3
def tanimoto_ad(
    training_smiles: List[str],
    test_smiles: List[str],
    tanimoto_threshold: float = 0.7,
    radius: int = 2,
    hash_length: int = 1024,
    minimum_fraction_similar: Optional[float] = None,
    **kwargs
) -> Union[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[bool | int | float | complex | str | bytes]]
```

Function to calculate the applicability domain of a model using Tanimoto similarity

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| training_smiles | Iterable | An iterable of SMILES strings for the training set | None |
| test_smiles | Iterable | An iterable of SMILES strings for the test set | None |
| tanimoto_threshold | float | The threshold for the Tanimoto similarity | None |
| radius | int | The radius for the Morgan fingerprint | None |
| hash_length | int | The length of the hash for the fingerprint | None |

**Returns:**

| Type | Description |
|---|---|
| ArrayLike | A boolean array of whether the test set is within the threshold |

??? example "View Source"
        def tanimoto_ad(

            training_smiles: List[str],

            test_smiles: List[str],

            tanimoto_threshold: float = 0.7,

            radius: int = 2,

            hash_length: int = 1024,

            minimum_fraction_similar: Union[float, None] = None,

            **kwargs,

        ) -> ArrayLike:

            """

            Function to calculate the applicability domain of a model using Tanimoto similarity

            Args:

                training_smiles (Iterable): An iterable of SMILES strings for the training set

                test_smiles (Iterable): An iterable of SMILES strings for the test set

                tanimoto_threshold (float): The threshold for the Tanimoto similarity

                radius (int): The radius for the Morgan fingerprint

                hash_length (int): The length of the hash for the fingerprint

            Returns:

                ArrayLike: A boolean array of whether the test set is within the threshold

            """

            # Convert SMILES to RDKit molecules

            training_mols = [Chem.MolFromSmiles(smile) for smile in training_smiles]

            test_mols = [Chem.MolFromSmiles(smile) for smile in test_smiles]

            ecpf_generator = AllChem.GetMorganGenerator(

                radius=radius, fpSize=hash_length, **kwargs

            )

            # Convert molecules to fingerprints

            training_fps = ecpf_generator.GetFingerprints(training_mols, numThreads=4)

            test_fps = ecpf_generator.GetFingerprints(test_mols, numThreads=4)

            # Calculate bulk Tanimoto similarity

            similarity_matrix = np.zeros((len(test_fps), len(training_fps)))

            for ith, test_fp in enumerate(test_fps):

                similarities = DataStructs.BulkTanimotoSimilarity(test_fp, training_fps)

                similarity_matrix[ith, :] = similarities

            log.debug(f"Tanimoto similarity matrix: {similarity_matrix}")

            # Determine if each test molecule is within the threshold

            if minimum_fraction_similar is None:

                within_threshold = np.any(similarity_matrix >= tanimoto_threshold, axis=1)

            else:

                fraction_similar = np.sum(

                    similarity_matrix >= tanimoto_threshold, axis=1

                ) / len(training_fps)

                within_threshold = fraction_similar >= minimum_fraction_similar

                log.debug(f"In applicability domain Fraction similar: {fraction_similar}")

            log.debug(f"In applicability domain within threshold: {within_threshold}")

            return within_threshold
