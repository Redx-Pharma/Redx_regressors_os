# Module redxregressors.ml_featurization

Module for generating features such as chemical fingerprints and descriptors and teh converison of common data types of those representations

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        Module for generating features such as chemical fingerprints and descriptors and teh converison of common data types of those representations

        """

        from typing import List, Union, Tuple, Optional, Self

        import pandas as pd

        import numpy as np

        from rdkit import Chem

        from rdkit.Chem import MACCSkeys, AllChem

        import torch

        from rdkit.DataStructs import cDataStructs

        from rdkit.Chem.Descriptors import CalcMolDescriptors

        from sentence_transformers import SentenceTransformer

        from sklearn.base import BaseEstimator, TransformerMixin

        from torch_geometric.loader import DataLoader

        from torch_geometric.data import Data

        from pysmiles import read_smiles

        from redxregressors import utilities

        import os

        from tqdm import tqdm

        import logging

        from redxregressors.utilities import seed_worker, seed_all

        log = logging.getLogger(__name__)

        torch.use_deterministic_algorithms(True)



        class ECFPConverter(BaseEstimator, TransformerMixin):

            """

            This is a pipeline module class for converting SMILES strings to ECFP fingerprints. It can be directly built into a sklearn pipeline.

            For example:

            from sklearn.pipeline import Pipeline

            from sklearn.ensemble import RandomForestClassifier

            from redxregressors import featurization

            model = Pipeline([('smiles_converter', featurization.ECFPConverter()), ('RF', RandomForestClassifier())])

            """

            def __init__(

                self, hash_length: int = 2048, radius: int = 2, smiles_col: str = "smiles"

            ) -> None:

                log.info("Initialising transformer...")

                self.hash_length = hash_length

                self.radius = radius

                self.smiles_col = smiles_col

            def fit(self, X, y=None):

                return self

            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to ECFP representations .....")

                if len(X.shape) == 1:

                    fps = get_ecfp(

                        smiles=X,

                        hash_length=self.hash_length,

                        radius=self.radius,

                        return_np=True,

                    )

                    return fps

                else:

                    fps = get_ecfp(

                        smiles=X[self.smiles_col],

                        hash_length=self.hash_length,

                        radius=self.radius,

                        return_np=True,

                    )

                    return fps



        class CECFPConverter(BaseEstimator, TransformerMixin):

            """

            This is a pipeline module class for converting SMILES strings to count ECFP fingerprints. It can be directly built into a sklearn pipeline.

            For example:

            from sklearn.pipeline import Pipeline

            from sklearn.preprocessing import StandardScaler

            from sklearn.ensemble import RandomForestClassifier

            from redxregressors import featurization

            model = Pipeline([('smiles_converter', featurization.CECFPConverter()), ('scaler', StandardScaler()), ('RF', RandomForestClassifier())])

            """

            def __init__(

                self, hash_length: int = 2048, radius: int = 2, smiles_col: str = "smiles"

            ) -> None:

                log.info("Initialising transformer...")

                self.hash_length = hash_length

                self.radius = radius

                self.smiles_col = smiles_col

            def fit(self, X, y=None):

                return self

            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to CECFP representations .....")

                if len(X.shape) == 1:

                    fps = get_count_ecfp(

                        smiles=X,

                        hash_length=self.hash_length,

                        radius=self.radius,

                        return_np=True,

                    )

                    return fps

                else:

                    fps = get_count_ecfp(

                        smiles=X[self.smiles_col],

                        hash_length=self.hash_length,

                        radius=self.radius,

                        return_np=True,

                    )

                    return fps



        class MACCSConverter(BaseEstimator, TransformerMixin):

            """

            This is a pipeline module class for converting SMILES strings to MACCS fingerprints. It can be directly built into a sklearn pipeline.

            For example:

            from sklearn.pipeline import Pipeline

            from sklearn.ensemble import RandomForestClassifier

            from redxregressors import featurization

            model = Pipeline([('smiles_converter', featurization.MACCSConverter()), ('RF', RandomForestClassifier())])

            """

            def __init__(self, smiles_col: str = "smiles") -> None:

                log.info("Initialising transformer...")

                self.smiles_col = smiles_col

            def fit(self, X, y=None):

                return self

            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to MACCS representations .....")

                if len(X.shape) == 1:

                    fps = get_maccs(smiles=X, return_np=True)

                    return fps

                else:

                    fps = get_maccs(smiles=X[self.smiles_col], return_np=True)

                    return fps



        class RDKitDescriptorsConverter(BaseEstimator, TransformerMixin):

            """

            This is a pipeline module class for converting SMILES strings to RDKit descriptors. It can be directly built into a sklearn pipeline.

            For example:

            from sklearn.pipeline import Pipeline

            from sklearn.preprocessing import StandardScaler

            from sklearn.ensemble import RandomForestClassifier

            from redxregressors import featurization

            model = Pipeline([('smiles_converter', featurization.RDKitDescriptorsConverter()), ('scaler', StandardScaler()), ('RF', RandomForestClassifier())])

            """

            def __init__(self, smiles_col: str = "smiles") -> None:

                log.info("Initialising transformer...")

                self.smiles_col = smiles_col

            def fit(self, X, y=None):

                return self

            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to RDKit descriptors .....")

                if len(X.shape) == 1:

                    fps = get_rdkit_descriptors(smiles=X, return_np=True)

                    return fps

                else:

                    fps = get_rdkit_descriptors(smiles=X[self.smiles_col], return_np=True)

                    return fps



        class NLPConverter(BaseEstimator, TransformerMixin):

            """

            This is a pipeline module class for converting SMILES strings to sentence embeddings. It can be directly built into a sklearn pipeline.

            For example:

            from sklearn.pipeline import Pipeline

            from sklearn.preprocessing import StandardScaler

            from sklearn.ensemble import RandomForestClassifier

            from redxregressors import featurization

            model = Pipeline([('smiles_converter', featurization.NLPConverter()), ('scaler', StandardScaler()), ('RF', RandomForestClassifier())])

            """

            def __init__(

                self,

                model_name: str = "laituan245/molt5-large-smiles2caption",

                smiles_col: str = "smiles",

            ) -> None:

                log.info("Initialising transformer...")

                self.model_name = model_name

                self.smiles_col = smiles_col

                self.model = SentenceTransformer(self.model_name)

            def fit(self, X, y=None):

                return self

            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to sentence embeddings .....")

                if len(X.shape) == 1:

                    embeddings = get_nlp_smiles_rep(model_name=self.model_name, smiles=X)

                    return embeddings

                else:

                    embeddings = get_nlp_smiles_rep(

                        model_name=self.model_name, smiles=X[self.smiles_col]

                    )

                    return embeddings



        def validate_smile(smile: str, canonicalize: bool = True) -> Union[None, str]:

            """

            Function to validate a single smiles string. This differs from get_valid_smiles as

            it operates on a single smiles string and garuntees a return as if the smiles is invalid it returns None.

            Args:

                smile (str): smiles string to check if it is valid

                canaonicalize (bool): whether to return the input smiles or a canonicalized version for a valid smiles string

            Returns:

                Union[None, str]: None for a invalid smiles string and a smiles string for a valid one

            """

            try:

                m = Chem.MolFromSmiles(smile)

            except Exception:

                log.error("Exception when converting smiles to RDKit molecule object")

                return None

            if m is None:

                log.error(f"SMILES string: {smile} is invalid in RDKit and will be skipped")

                return None

            if canonicalize is True:

                return Chem.MolToSmiles(m)

            else:

                return smile



        def validate_smiles(

            smiles: List[str], return_failed_as_None: bool = True, canonicalize: bool = True

        ) -> List[Union[None, str]]:

            """

            Function to validate a list of m smiles strings. This differs from get_valid_smiles as

            it can guarntee a return, if return_failed_as_None is True (defualt), as if the smiles

            is invalid it returns None.

            Args:

                smile (str): smiles string to check if it is valid

                return_failed_as_None (bool): whether to return None if the smiles string is invalid

                canaonicalize (bool): whether to return the input smiles or a canonicalized version for a valid smiles string

            Returns:

                List[Union[None, str]]: None for a invalid smiles string and a smiles string for a valid one

            """

            if return_failed_as_None is True:

                return [validate_smile(smile, canonicalize=canonicalize) for smile in smiles]

            else:

                tmp = [validate_smile(smile, canonicalize=canonicalize) for smile in smiles]

                return [ent for ent in tmp if ent is not None]



        def list_of_bitvects_to_numpy_arrays(

            bvs: List[cDataStructs.ExplicitBitVect],

        ) -> np.ndarray:

            """

            Function to convert list of explicit bitvectirs from RDKit to numpy arrays. Note that at the time of writing RDKit has functions to do this one at a time but not in batches.

            Args:

                bvs (List[cDataStructs.ExplicitBitVect]): List of bitvects from RDKit

            Returns:

                np.ndarray: Numpy array of the bit vector arrays

            Doctest:

            > list_of_bitvects_to_numpy_arrays([cDataStructs.CreateFromBitString("1011")]) # doctest: +NORMALIZE_WHITESPACE

            array([[1, 0, 1, 1]], dtype=uint8)

            """

            return np.array(

                [[int(ent) for ent in list(ent.ToBitString())] for ent in bvs]

            ).astype("uint8")



        def list_of_bitvects_to_list_of_lists(

            bvs: List[cDataStructs.ExplicitBitVect],

        ) -> List[List[int]]:

            """

            Function to convert list of explicit bitvects from RDKit to list of lists. Note that at the time of writing RDKit has functions to do this one at a time but not in batches.

            Args:

                bvs (List[cDataStructs.ExplicitBitVect]): List of bitvects from RDKit

            Returns:

               List[List[int]]: list of lists of integer binary values

            Doctest:

            > list_of_bitvects_to_list_of_lists([cDataStructs.CreateFromBitString("1011")]) # doctest: +NORMALIZE_WHITESPACE

            [[1, 0, 1, 1]]

            """

            return [[int(ent) for ent in list(ent.ToBitString())] for ent in bvs]



        def bitstring_to_bit_vect(bstring: str) -> cDataStructs.ExplicitBitVect:

            """

            Function to convert a bit string i.e. "100010101" to an RDKit explicit bit vector

            Args:

                bstring (str): bit string i.e. a string made up of 1 and 0 only

            Returns:

                cDataStructs.ExplicitBitVect: RDKit explicit bit vector

            Doctest:

            > bitstring_to_bit_vect('10101010001101') # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            <rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>

            """

            return cDataStructs.CreateFromBitString(bstring)



        def df_rows_to_list_of_bit_vect(df: pd.DataFrame) -> List[cDataStructs.ExplicitBitVect]:

            """

            Convert rows of binary values in a dataframe to a list of RDKit explicit bit vectors

            Args:

                df (pd.DataFrame): _description_

            Returns:

                List[cDataStructs.ExplicitBitVect]: _description_

            Doctest:

            > df_rows_to_list_of_bit_vect(pd.DataFrame([[1, 0, 1, 0, 1, 1, 1, 1]])) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]

            """

            bitvectors = []

            for _, row in df.iterrows():

                log.debug(row)

                bs = "".join([str(ent) for ent in row.to_list()])

                log.debug(bs)

                bitvectors.append(bitstring_to_bit_vect(bs))

            return bitvectors



        def validate_smiles_and_get_ecfp(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            radius: int = 2,

            hash_length: int = 2048,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:

            """

            Function to generate ECFP representations from smiles

            Args:

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.

                hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Example:

            ```python

            > validate_smiles_and_get_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]

            > validate_smiles_and_get_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)

            ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    input_n = len(df.index)

                    smiles = validate_smiles(df[smiles_column].to_list())

                    df["ECFP_smiles_standardized"] = smiles

                    df = df.dropna(axis=0, subset=["ECFP_smiles_standardized"])

                    smiles = df["ECFP_smiles_standardized"].to_list()

                    log.info(

                        f"{(len(smiles) / input_n) * 100.0}% ({len(smiles)} out of {input_n}) of the input smiles were successfully read"

                    )

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                input_n = len(smiles)

                df = data_df

                smiles = validate_smiles(smiles, return_failed_as_None=False)

                log.info(

                    f"{(len(smiles) / input_n) * 100.0}% of the input smiles were successfully read"

                )

            fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

            if return_np is True and return_df is False:

                fps_np = np.array(

                    [

                        fp_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))

                        if smi is not None

                        else None

                        for smi in smiles

                    ]

                )

                return fps_np

            elif return_df is True and return_np is False:

                fps_df = pd.DataFrame(

                    [

                        fp_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))

                        if smi is not None

                        else None

                        for smi in smiles

                    ],

                    columns=[f"ecfp_bit_{ith}" for ith in range(hash_length)],

                )

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                fps = [

                    fp_gen.GetFingerprint(Chem.MolFromSmiles(smi)) if smi is not None else None

                    for smi in smiles

                ]

                return fps



        def get_ecfp(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            radius: int = 2,

            hash_length: int = 2048,

            return_df: bool = False,

            return_np: bool = False,

            n_threads: int = 8,

            **kwargs,

        ) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:

            """

            Function to generate ECFP representations from smiles

            Args:

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.

                hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Examples:

            ```python

            > get_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            (<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>,)

            > get_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)

            ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = df[smiles_column].to_list()

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            log.info(f"Making ECFP fingerprints for {len(smiles)} molecules")

            fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

            if return_np is True and return_df is False:

                fps = fp_gen.GetFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                fps_np = list_of_bitvects_to_numpy_arrays(fps)

                return fps_np

            elif return_df is True and return_np is False:

                fps = fp_gen.GetFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                fps_ll = list_of_bitvects_to_list_of_lists(fps)

                fps_df = pd.DataFrame(

                    fps_ll,

                    columns=[f"ecfp_bit_{ith}" for ith in range(hash_length)],

                )

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                fps = fp_gen.GetFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                return fps



        def get_count_ecfp(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            radius: int = 2,

            hash_length: int = 2048,

            return_df: bool = False,

            return_np: bool = False,

            n_threads: int = 8,

            **kwargs,

        ) -> Union[Tuple[cDataStructs.UIntSparseIntVect], pd.DataFrame, np.ndarray]:

            """

             Function to generate count ECFP representations from smiles

             Args:

                 data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                  If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                 smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                 smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                 radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.

                 hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.

                 return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                 return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

             Raises:

                 RuntimeError: If incompatible inputs are given

             Returns:

                 Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Examples:

            ```python

             > get_count_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

             (<rdkit.DataStructs.cDataStructs.UIntSparseIntVect object at ...>,)

             > get_count_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

             array([[0, 0, 0, ..., 0, 0, 0]])

             ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = df[smiles_column].to_list()

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            log.info(f"Making CECFP fingerprints for {len(smiles)} molecules")

            fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

            if return_np is True and return_df is False:

                fps = fp_gen.GetCountFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                fps_np = np.array([ent.ToList() for ent in fps])

                return fps_np

            elif return_df is True and return_np is False:

                fps = fp_gen.GetCountFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                fps_ll = [ent.ToList() for ent in fps]

                fps_df = pd.DataFrame(

                    fps_ll,

                    columns=[f"ecfp_count_bit_{ith}" for ith in range(hash_length)],

                )

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                fps = fp_gen.GetCountFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                return fps



        def get_maccs(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:

            """

            Function to generate MACCS MDL keys representations from smiles

            Args:

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Doctest:

            ```python

            > get_maccs(smiles=["c1ccccc1C"]) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]

            > get_maccs(smiles=["c1ccccc1C"])[0].GetOnBits()

            (160, 162, 163, 165)

            ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = df[smiles_column].to_list()

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            log.info(f"Making MACCS fingerprints for {len(smiles)} molecules")

            fps = [

                MACCSkeys.GenMACCSKeys(ent)

                for ent in [Chem.MolFromSmiles(smi) for smi in smiles]

            ]

            num_maccs_keys = len(fps[0])

            if return_np is True and return_df is False:

                fps_np = np.array([ent.ToList() for ent in fps])

                return fps_np

            elif return_df is True and return_np is False:

                fps_ll = [ent.ToList() for ent in fps]

                fps_df = pd.DataFrame(

                    fps_ll,

                    columns=[f"maccs_bit_{ith}" for ith in range(num_maccs_keys)],

                )

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                return fps



        def get_rdkit_descriptors(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[List[dict], pd.DataFrame, np.ndarray]:

            """

            Function to generate Rdkit descriptor representations from smiles

            Args:

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Example:

            ```python

            > type(get_rdkit_descriptors(smiles=["c1ccccc1C"])[0])

            <class 'dict'>

            > len(get_rdkit_descriptors(smiles=["c1ccccc1C"], return_df=True).columns)

            210

            ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = df[smiles_column].to_list()

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            log.info(f"Making RDKit descriptors for {len(smiles)} molecules")

            fps = [

                CalcMolDescriptors(ent) for ent in [Chem.MolFromSmiles(smi) for smi in smiles]

            ]

            # make the order consistent sort on the keys in the dictionary important for the numpy return

            fps = [dict(sorted(ent.items())) for ent in fps]

            if return_np is True and return_df is False:

                fps_np = pd.DataFrame(fps).values

                return fps_np

            elif return_df is True and return_np is False:

                fps_df = pd.DataFrame(fps)

                fps_df.columns = [f"rdkit_descriptor_{name}" for name in fps_df.columns]

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                return fps



        def get_t5_smiles_rep(

            model_name: str = "laituan245/molt5-large-smiles2caption",

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[List[dict], pd.DataFrame, np.ndarray]:

            """

            Function to generate the T5 embedding representations from smiles using a transformer model. We use the Apache - 2.0 licened model (accessed 7/10/24)

            from laituan245/molt5-large-smiles2caption https://huggingface.co/laituan245/molt5-large-smiles2caption. This function calls get_nlp_smiles_rep using this model.

            Args:

                model_name (str): the model name to use,

                model_type (str): The model type to use i.e. bert, roberta or gpt2.

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

                combine_strategy (Union[str, int]): How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean"

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            """

            embedding_rep = get_nlp_smiles_rep(

                model_name=model_name,

                data_df=data_df,

                smiles_column=smiles_column,

                smiles=smiles,

                return_df=return_df,

                return_np=return_np,

                **kwargs,

            )

            return embedding_rep



        def get_deberta_smiles_rep(

            model_name: str = "knowledgator/SMILES-DeBERTa-large",

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[List[dict], pd.DataFrame, np.ndarray]:

            """

            Function to generate the DeBERTa embedding representations from smiles using a transformer model. We use the Apache - 2.0 licened model (accessed 7/10/24)

            from knowledgator/SMILES-DeBERTa-large https://huggingface.co/knowledgator/SMILES-DeBERTa-large. This function calls get_nlp_smiles_rep using this model.

            Args:

                model_name (str): the model name to use,

                model_type (str): The model type to use i.e. bert, roberta or gpt2.

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

                combine_strategy (Union[str, int]): How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean"

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            """

            embedding_rep = get_nlp_smiles_rep(

                model_name=model_name,

                data_df=data_df,

                smiles_column=smiles_column,

                smiles=smiles,

                return_df=return_df,

                return_np=return_np,

                **kwargs,

            )

            return embedding_rep



        def get_nlp_smiles_rep(

            model_name: str = "Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset",

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[pd.DataFrame, np.ndarray]:

            """

            Function to generate the NLP embedding representations from smiles using a transformer model

            Args:

                model_name (str): the model name to use, default Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset apache 2.0 license accessed 30/10/24 https://huggingface.co/Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

                combine_strategy (Union[str, int]): How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean"

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = list(df[smiles_column])

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            input_n = len(smiles)

            valid_smiles = validate_smiles(smiles, return_failed_as_None=False)

            if len(valid_smiles) != input_n:

                raise RuntimeError(

                    f"ERROR - only {(len(valid_smiles) / input_n) * 100.0}% of the input smiles were successfully read. Please correct the invalid smiles"

                )

            log.info(

                f"Making NLP embeddings using model {model_name} for {len(smiles)} molecules"

            )

            model = SentenceTransformer(model_name, **kwargs)

            embedding_rep = model.encode(smiles, **kwargs)

            if return_np is True and return_df is False:

                return embedding_rep

            elif return_df is True and return_np is False:

                fps_df = pd.DataFrame(embedding_rep)

                fps_df.columns = [

                    f"embedding_{model_name.replace('/', '-')}_{ith}"

                    for ith in range(fps_df.shape[1])

                ]

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                return embedding_rep



        def from_smiles_without_default_graph_feature_gen(

            smiles: str, validate_and_canonicalize: bool = False

        ) -> Data:

            r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`

            instance.

            Args:

                smiles (str): The SMILES string.

                with_hydrogen (bool, optional): If set to :obj:`True`, will store

                    hydrogens in the molecule graph. (default: :obj:`False`)

                kekulize (bool, optional): If set to :obj:`True`, converts aromatic

                    bonds to single/double bonds. (default: :obj:`False`)

            """

            if validate_and_canonicalize is True:

                smiles = utilities.validate_smile(smiles, canonicalize=True)

            data = Data(

                x=torch.tensor([1]),

                edge_index=torch.tensor([[0, 1], [1, 0]]),

                edge_attr=torch.tensor([[1], [1]]),

            )

            data.smiles = smiles

            return data



        class GetAttentiveFPFeatures:

            """

            Class for making the attentiveFP features. Its is based on the paper https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959 and the PyTorch Geometric documentation

            https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/molecule_net.py accessed 21/10/24. The pytorch geometric code is licensed under

            the MIT license which is given below.

            Copyright (c) 2023 PyG Team <team@pyg.org>

            Permission is hereby granted, free of charge, to any person obtaining a copy

            of this software and associated documentation files (the "Software"), to deal

            in the Software without restriction, including without limitation the rights

            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell

            copies of the Software, and to permit persons to whom the Software is

            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in

            all copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,

            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE

            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER

            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN

            THE SOFTWARE.

            """

            def __init__(self) -> None:

                # these definitionns are based on the paper Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism and

                # the AttentiveFP model in pytorch geometric https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/molecule_net.py. The license

                # for the pytorch geometric code is given above but this class has been re-written for use here.

                self.atomic_symbols = [

                    "B",

                    "C",

                    "N",

                    "O",

                    "F",

                    "Si",

                    "P",

                    "S",

                    "Cl",

                    "As",

                    "Se",

                    "Br",

                    "Te",

                    "I",

                    "At",

                    "metal",

                ]

                self.stereo_types = [

                    Chem.rdchem.BondStereo.STEREONONE,

                    Chem.rdchem.BondStereo.STEREOANY,

                    Chem.rdchem.BondStereo.STEREOZ,

                    Chem.rdchem.BondStereo.STEREOE,

                ]

                self.hybrid_types = [

                    Chem.rdchem.HybridizationType.SP,

                    Chem.rdchem.HybridizationType.SP2,

                    Chem.rdchem.HybridizationType.SP3,

                    Chem.rdchem.HybridizationType.SP3D,

                    Chem.rdchem.HybridizationType.SP3D2,

                    "other",

                ]

                self.bond_types = [

                    Chem.rdchem.BondType.SINGLE,

                    Chem.rdchem.BondType.DOUBLE,

                    Chem.rdchem.BondType.TRIPLE,

                    Chem.rdchem.BondType.AROMATIC,

                ]

            def __call__(self, data) -> Data:

                """

                This functions produces the default AtteniveFP feature for nodes and edges in a graph from a SMILES string.

                See the paper Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism Table 1

                for the features used see the site https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959 for the paper.

                Args:

                    data (Data): The data object to add the features to

                Returns:

                    Data: The data object with the features added

                """

                mol = utilities.validate_smile(data.smiles, canonicalize=False, return_mol=True)

                symbol_vec = torch.zeros((len(self.atomic_symbols)), dtype=torch.float32)

                n_connections = torch.zeros((6), dtype=torch.float32)

                hybridization_vec = torch.zeros((len(self.hybrid_types)), dtype=torch.float32)

                n_hydrogens = torch.zeros((5), dtype=torch.float32)

                chiral_vec = torch.zeros((2), dtype=torch.float32)

                node_features = []

                for atom in mol.GetAtoms():

                    # symbol

                    sym_idx = self.atomic_symbols.index(atom.GetSymbol())

                    symbol_vec[sym_idx] = 1.0

                    # degree

                    n_con_idx = atom.GetDegree()

                    n_connections[n_con_idx] = 1.0

                    # formal charge

                    formal_charge = torch.tensor([atom.GetFormalCharge()], dtype=torch.float32)

                    # radical electrons

                    radical_electrons = torch.tensor(

                        [atom.GetNumRadicalElectrons()], dtype=torch.float32

                    )

                    # hybridization

                    hyb_idx = self.hybrid_types.index(atom.GetHybridization())

                    hybridization_vec[hyb_idx] = 1.0

                    # aromaticity

                    if atom.GetIsAromatic():

                        aromaticity = torch.tensor([1.0], dtype=torch.float32)

                    else:

                        aromaticity = torch.tensor([0.0], dtype=torch.float32)

                    # hydrogens

                    n_hyd_idx = atom.GetTotalNumHs()

                    n_hydrogens[n_hyd_idx] = 1.0

                    # chirality

                    if atom.HasProp("_ChiralityPossible"):

                        chirality = torch.tensor([1.0], dtype=torch.float32)

                    else:

                        chirality = torch.tensor([0.0], dtype=torch.float32)

                    # chirality type

                    if atom.HasProp("_CIPCode"):

                        if atom.GetProp("_CIPCode") == "R":

                            chiral_vec[0] = 1.0

                        elif atom.GetProp("_CIPCode") == "S":

                            chiral_vec[1] = 1.0

                    node_features.append(

                        torch.concatenate(

                            [

                                symbol_vec,

                                n_connections,

                                formal_charge,

                                radical_electrons,

                                hybridization_vec,

                                aromaticity,

                                n_hydrogens,

                                chirality,

                                chiral_vec,

                            ],

                            dim=0,

                        )

                    )

                data.x = torch.stack(node_features, dim=0)

                edge_indxs = []

                edge_attrs = []

                bond_order = torch.zeros((4), dtype=torch.float32)

                stero_bond = torch.zeros((4), dtype=torch.float32)

                for bond in mol.GetBonds():

                    # graph index connections for the edges atom 1 to atom 2 and atom 2 to atom 1

                    edge_indxs.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

                    edge_indxs.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

                    # bond type

                    bo_idx = self.bond_types.index(bond.GetBondType())

                    bond_order[bo_idx] = 1.0

                    # bond is it conjugated

                    if bond.GetIsConjugated():

                        conjugation = torch.tensor([1.0], dtype=torch.float32)

                    else:

                        conjugation = torch.tensor([0.0], dtype=torch.float32)

                    # is the bond in a ring

                    if bond.IsInRing():

                        ring = torch.tensor([1.0], dtype=torch.float32)

                    else:

                        ring = torch.tensor([0.0], dtype=torch.float32)

                    # bond stereo chemistry type if any

                    sb_idx = self.stereo_types.index(bond.GetStereo())

                    stero_bond[sb_idx] = 1.0

                    edge_attr = torch.concatenate(

                        [bond_order, conjugation, ring, stero_bond], dim=0

                    )

                    edge_attrs.append(edge_attr)

                    edge_attrs.append(edge_attr)

                if len(edge_attrs) == 0:

                    data.edge_index = torch.zeros((2, 0), dtype=torch.long)

                    data.edge_attr = torch.zeros((0, 10), dtype=torch.float)

                else:

                    data.edge_index = torch.tensor(edge_indxs).t().contiguous()

                    data.edge_attr = torch.stack(edge_attrs, dim=0)

                return data

            def __repr__(self) -> str:

                return f"{self.__class__.__name__}()"

            def __str__(self) -> str:

                return f"{self.__class__.__name__}() node_features: 39, edge_features: 10 Feature for AttentiveFP model https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959 table 1"



        class GCN_featurize(object):

            """

            Class to featurize molecules for use with a GCN model.

            """

            def __init__(

                self,

                smiles: list[str],

                targets: list[float],

                explicit_h: bool = True,

                include_aromaticity: bool = True,

                include_charge: bool = True,

                elements: Optional[List[str]] = None,

                train_fraction: Optional[float] = 0.8,

                test_fraction: Optional[float] = 0.1,

                validation_fraction: Optional[float] = 0.1,

                batch_size: Optional[int] = 1,

            ) -> Self:

                """

                Class to featurize molecules for use with a GCN model.

                Args:

                    smiles (list[str]): A list of SMILES strings.

                    targets (list[float]): A list of target values.

                    explicit_h (bool, optional): Whether to include explicit hydrogens. Defaults to True.

                    include_aromaticity (bool, optional): Whether to include aromaticity. Defaults to True.

                    include_charge (bool, optional): Whether to include charge. Defaults to True.

                    elements (Optional[List[str]], optional): A list of elements to include. Defaults to None.

                    train_fraction (Optional[float], optional): The fraction of data to use for training. Defaults to 0.8.

                    test_fraction (Optional[float], optional): The fraction of data to use for testing. Defaults to 0.1.

                    validation_fraction (Optional[float], optional): The fraction of data to use for validation. Defaults to 0.1.

                    batch_size (Optional[int], optional): The batch size. Defaults to 1.

                Returns:

                    Self: The GCN_featurize object.

                """

                self.smiles = utilities.validate_smiles(smiles)

                self.targets = targets

                self.explicit_h = explicit_h

                self.include_aromaticity = include_aromaticity

                self.include_charge = include_charge

                self.train_fraction = train_fraction

                self.test_fraction = test_fraction

                self.validation_fraction = validation_fraction

                self.batch_size = batch_size

                if elements is None:

                    if self.explicit_h is True:

                        self.elements = [

                            "H",

                            "B",

                            "C",

                            "N",

                            "O",

                            "F",

                            "P",

                            "S",

                            "Cl",

                            "Br",

                            "I",

                        ]

                    else:

                        self.elements = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

                else:

                    self.elements = elements

                # singe, aromatic, double, triple

                self.bond_types = [1, 1.5, 2, 3]

            def __call__(self) -> DataLoader | Tuple[DataLoader, DataLoader, DataLoader]:

                """

                Featurize the molecules.

                Returns:

                    DataLoader | Tuple[DataLoader, DataLoader, DataLoader]: The data loader(s).

                """

                # seed all random number generators

                seed_all()

                # to hold the data objects

                data = []

                # determine which node featrues to include

                node_vec_length = len(self.elements)

                if self.include_aromaticity is True:

                    node_vec_length += 1

                if self.include_charge is True:

                    node_vec_length += 1

                log.info(f"Node features have length: {node_vec_length}")

                # iterate over the smiles and make the data objects

                for smi, targ_y in tqdm(

                    zip(self.smiles, self.targets), desc="Cycling over smiles ....."

                ):

                    molecule_features = []

                    edge_idxs = []

                    edge_features = []

                    # use the read_smiles function to get the graph with a key node features and edge features

                    G = read_smiles(smi, explicit_hydrogen=self.explicit_h)

                    elements = np.asarray(G.nodes(data="element"))[:, 1]

                    if self.include_aromaticity is True:

                        aromatic = np.asarray(G.nodes(data="aromatic"))[:, 1]

                    if self.include_charge is True:

                        charge = np.asarray(G.nodes(data="charge"))[:, 1]

                    # iterate over the nodes to make the node features. We build a vector for each node and aggregate them into a list as a molecule matrix for the node features

                    for ith, atom in enumerate(elements):

                        node_vec = np.zeros((node_vec_length), dtype=np.float32)

                        node_vec[self.elements.index(atom)] = 1.0

                        if self.include_aromaticity is True:

                            node_vec[-2] = aromatic[ith]

                        if self.include_charge is True:

                            node_vec[-1] = charge[ith]

                        molecule_features.append(node_vec)

                    # iterate over the edges to make the edge features. We build a vector for each edge and aggregate them into a list as a molecule matrix for the edge features.

                    # Note that each bond is included as two edges in the graph to make the graph undirected. This means that the edge features are also duplicated. This is a requirement for the pytorch geometric data object.

                    for ith, edg in enumerate(G.edges(data=True)):

                        edge_idxs.append([edg[0], edg[1]])

                        edge_idxs.append([edg[1], edg[0]])

                        log.debug(

                            f" edge is {edg[2]['order']} which is index {[jth for jth, ent in enumerate(self.bond_types) if edg[2]['order'] - ent < 1e-5][0]} in bond types {self.bond_types}"

                        )

                        edg_feat_vec = torch.tensor([float(edg[2]["order"])])

                        edge_features.append(edg_feat_vec)

                        edge_features.append(edg_feat_vec)

                    # make the data object for the molecule

                    mol_nodes = torch.tensor(molecule_features)

                    edges = torch.tensor(edge_idxs, dtype=torch.int).t().contiguous()

                    edge_attibutes = torch.stack(edge_features, dim=0)

                    y = torch.tensor([targ_y], dtype=torch.float32)

                    data.append(

                        Data(x=mol_nodes, edge_index=edges, edge_attr=edge_attibutes, y=y)

                    )

                # We now have the features for all molecules so we noe split the data into train, test and validation sets if the fractions are set alternatively we return the data as a single DataLoader

                # first the single dataloader case

                if any(

                    ent is None

                    for ent in [

                        self.train_fraction,

                        self.test_fraction,

                        self.validation_fraction,

                    ]

                ):

                    log.info(

                        "At least one of train, test and validation fractions is None so returning the data as a single DataLoader. If you intend to split the data please set all the fractions 0.0 "

                    )

                    g = torch.Generator()

                    g.manual_seed(0)

                    return DataLoader(

                        data,

                        batch_size=self.batch_size,

                        shuffle=True,

                        worker_init_fn=seed_worker,

                        generator=g,

                    )

                # now the case where we split the data to train, test and validation sets

                else:

                    log.info("Splitting data into train, test and validation sets")

                    if (

                        self.train_fraction + self.test_fraction + self.validation_fraction

                        != 1.0

                    ):

                        raise RuntimeError(

                            "ERROR - The train, test and validation fractions do not sum to 1.0"

                        )

                    n_train = int(len(data) * self.train_fraction)

                    n_test = int(len(data) * self.test_fraction)

                    indexes = [ith for ith in range(len(data))]

                    train_indices = np.random.choice(indexes, size=n_train, replace=False)

                    test_indices = np.array(list(set(indexes) - set(train_indices)))

                    test_indices = np.random.choice(test_indices, size=n_test, replace=False)

                    validation_indices = np.array(

                        list(set(indexes) - set(train_indices) - set(test_indices))

                    )

                    if any(ent in train_indices for ent in test_indices):

                        raise RuntimeError(

                            "ERROR - The train and test sets have overlapping indices"

                        )

                    if any(ent in train_indices for ent in validation_indices):

                        raise RuntimeError(

                            "ERROR - The train and validation sets have overlapping indices"

                        )

                    log.info(

                        f"Splitting data into train, test and validation sets with {len(train_indices)} data points for training, "

                        f"{len(test_indices)} data points for testing and the rest "

                        f" {len(validation_indices)} data points for validation."

                    )

                    log.info(

                        f"Train indices: {train_indices}{os.linesep}Test indicies: {test_indices}{os.linesep}Validation indicies: {validation_indices}"

                    )

                    g = torch.Generator()

                    g.manual_seed(0)

                    train = DataLoader(

                        # data[:train_stop_index],

                        [data[ith] for ith in train_indices],

                        batch_size=self.batch_size,

                        shuffle=True,

                        worker_init_fn=seed_worker,

                        generator=g,

                    )

                    test = DataLoader(

                        # data[train_stop_index:test_stop_index],

                        [data[ith] for ith in test_indices],

                        batch_size=1,

                        shuffle=False,

                        worker_init_fn=seed_worker,

                        generator=g,

                    )

                    validation = DataLoader(

                        # data[test_stop_index:],

                        [data[ith] for ith in validation_indices],

                        batch_size=1,

                        shuffle=False,

                        worker_init_fn=seed_worker,

                        generator=g,

                    )

                    return train, test, validation



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### bitstring_to_bit_vect

```python3
def bitstring_to_bit_vect(
    bstring: str
) -> rdkit.DataStructs.cDataStructs.ExplicitBitVect
```

Function to convert a bit string i.e. "100010101" to an RDKit explicit bit vector

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| bstring | str | bit string i.e. a string made up of 1 and 0 only | None |

**Returns:**

| Type | Description |
|---|---|
| cDataStructs.ExplicitBitVect | RDKit explicit bit vector |

??? example "View Source"
        def bitstring_to_bit_vect(bstring: str) -> cDataStructs.ExplicitBitVect:

            """

            Function to convert a bit string i.e. "100010101" to an RDKit explicit bit vector

            Args:

                bstring (str): bit string i.e. a string made up of 1 and 0 only

            Returns:

                cDataStructs.ExplicitBitVect: RDKit explicit bit vector

            Doctest:

            > bitstring_to_bit_vect('10101010001101') # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            <rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>

            """

            return cDataStructs.CreateFromBitString(bstring)


### df_rows_to_list_of_bit_vect

```python3
def df_rows_to_list_of_bit_vect(
    df: pandas.core.frame.DataFrame
) -> List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
```

Convert rows of binary values in a dataframe to a list of RDKit explicit bit vectors

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | _description_ | None |

**Returns:**

| Type | Description |
|---|---|
| List[cDataStructs.ExplicitBitVect] | _description_ |

??? example "View Source"
        def df_rows_to_list_of_bit_vect(df: pd.DataFrame) -> List[cDataStructs.ExplicitBitVect]:

            """

            Convert rows of binary values in a dataframe to a list of RDKit explicit bit vectors

            Args:

                df (pd.DataFrame): _description_

            Returns:

                List[cDataStructs.ExplicitBitVect]: _description_

            Doctest:

            > df_rows_to_list_of_bit_vect(pd.DataFrame([[1, 0, 1, 0, 1, 1, 1, 1]])) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]

            """

            bitvectors = []

            for _, row in df.iterrows():

                log.debug(row)

                bs = "".join([str(ent) for ent in row.to_list()])

                log.debug(bs)

                bitvectors.append(bitstring_to_bit_vect(bs))

            return bitvectors


### from_smiles_without_default_graph_feature_gen

```python3
def from_smiles_without_default_graph_feature_gen(
    smiles: str,
    validate_and_canonicalize: bool = False
) -> torch_geometric.data.data.Data
```

Converts a SMILES string to a :class:`torch_geometric.data.Data`

instance.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| smiles | str | The SMILES string. | None |
| with_hydrogen | bool | If set to :obj:`True`, will store<br>hydrogens in the molecule graph. (default: :obj:`False`) | None |
| kekulize | bool | If set to :obj:`True`, converts aromatic<br>bonds to single/double bonds. (default: :obj:`False`) | None |

??? example "View Source"
        def from_smiles_without_default_graph_feature_gen(

            smiles: str, validate_and_canonicalize: bool = False

        ) -> Data:

            r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`

            instance.

            Args:

                smiles (str): The SMILES string.

                with_hydrogen (bool, optional): If set to :obj:`True`, will store

                    hydrogens in the molecule graph. (default: :obj:`False`)

                kekulize (bool, optional): If set to :obj:`True`, converts aromatic

                    bonds to single/double bonds. (default: :obj:`False`)

            """

            if validate_and_canonicalize is True:

                smiles = utilities.validate_smile(smiles, canonicalize=True)

            data = Data(

                x=torch.tensor([1]),

                edge_index=torch.tensor([[0, 1], [1, 0]]),

                edge_attr=torch.tensor([[1], [1]]),

            )

            data.smiles = smiles

            return data


### get_count_ecfp

```python3
def get_count_ecfp(
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    radius: int = 2,
    hash_length: int = 2048,
    return_df: bool = False,
    return_np: bool = False,
    n_threads: int = 8,
    **kwargs
) -> Union[Tuple[rdkit.DataStructs.cDataStructs.UIntSparseIntVect], pandas.core.frame.DataFrame, numpy.ndarray]
```

Function to generate count ECFP representations from smiles

Args:
     data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
      If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
     smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
     smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
     radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.
     hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.
     return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
     return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

 Raises:
     RuntimeError: If incompatible inputs are given

 Returns:
     Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

??? example "View Source"
        def get_count_ecfp(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            radius: int = 2,

            hash_length: int = 2048,

            return_df: bool = False,

            return_np: bool = False,

            n_threads: int = 8,

            **kwargs,

        ) -> Union[Tuple[cDataStructs.UIntSparseIntVect], pd.DataFrame, np.ndarray]:

            """

             Function to generate count ECFP representations from smiles

             Args:

                 data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                  If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                 smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                 smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                 radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.

                 hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.

                 return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                 return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

             Raises:

                 RuntimeError: If incompatible inputs are given

             Returns:

                 Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Examples:

            ```python

             > get_count_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

             (<rdkit.DataStructs.cDataStructs.UIntSparseIntVect object at ...>,)

             > get_count_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

             array([[0, 0, 0, ..., 0, 0, 0]])

             ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = df[smiles_column].to_list()

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            log.info(f"Making CECFP fingerprints for {len(smiles)} molecules")

            fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

            if return_np is True and return_df is False:

                fps = fp_gen.GetCountFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                fps_np = np.array([ent.ToList() for ent in fps])

                return fps_np

            elif return_df is True and return_np is False:

                fps = fp_gen.GetCountFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                fps_ll = [ent.ToList() for ent in fps]

                fps_df = pd.DataFrame(

                    fps_ll,

                    columns=[f"ecfp_count_bit_{ith}" for ith in range(hash_length)],

                )

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                fps = fp_gen.GetCountFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                return fps


### get_deberta_smiles_rep

```python3
def get_deberta_smiles_rep(
    model_name: str = 'knowledgator/SMILES-DeBERTa-large',
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs
) -> Union[List[dict], pandas.core.frame.DataFrame, numpy.ndarray]
```

Function to generate the DeBERTa embedding representations from smiles using a transformer model. We use the Apache - 2.0 licened model (accessed 7/10/24)

from knowledgator/SMILES-DeBERTa-large https://huggingface.co/knowledgator/SMILES-DeBERTa-large. This function calls get_nlp_smiles_rep using this model.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model_name | str | the model name to use, | None |
| model_type | str | The model type to use i.e. bert, roberta or gpt2. | None |
| data_df | Optional[pd.DataFrame] | Dataframe containing at least the smiles strings to use.<br>If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None. | None |
| smiles_column | Optional[str] | Needed if data_df is given to define which column to find the smiles strings. Defaults to None. | None |
| smiles | Optional[list[str]] | A list of smiles strings to generate fingerprints for. Defaults to None. | None |
| return_df | bool | Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False. | False |
| return_np | bool | Whether to return a numpy array rather than a list of bit vectors. Defalts to False. | None |
| combine_strategy | Union[str, int] | How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean" | None |

**Returns:**

| Type | Description |
|---|---|
| Union[List[dict], pd.DataFrame, np.ndarray] | Depends on the return type asked for |

**Raises:**

| Type | Description |
|---|---|
| RuntimeError | If incompatible inputs are given |

??? example "View Source"
        def get_deberta_smiles_rep(

            model_name: str = "knowledgator/SMILES-DeBERTa-large",

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[List[dict], pd.DataFrame, np.ndarray]:

            """

            Function to generate the DeBERTa embedding representations from smiles using a transformer model. We use the Apache - 2.0 licened model (accessed 7/10/24)

            from knowledgator/SMILES-DeBERTa-large https://huggingface.co/knowledgator/SMILES-DeBERTa-large. This function calls get_nlp_smiles_rep using this model.

            Args:

                model_name (str): the model name to use,

                model_type (str): The model type to use i.e. bert, roberta or gpt2.

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

                combine_strategy (Union[str, int]): How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean"

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            """

            embedding_rep = get_nlp_smiles_rep(

                model_name=model_name,

                data_df=data_df,

                smiles_column=smiles_column,

                smiles=smiles,

                return_df=return_df,

                return_np=return_np,

                **kwargs,

            )

            return embedding_rep


### get_ecfp

```python3
def get_ecfp(
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    radius: int = 2,
    hash_length: int = 2048,
    return_df: bool = False,
    return_np: bool = False,
    n_threads: int = 8,
    **kwargs
) -> Union[Tuple[rdkit.DataStructs.cDataStructs.ExplicitBitVect], pandas.core.frame.DataFrame, numpy.ndarray]
```

Function to generate ECFP representations from smiles

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data_df | Optional[pd.DataFrame] | Dataframe containing at least the smiles strings to use.<br>If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None. | None |
| smiles_column | Optional[str] | Needed if data_df is given to define which column to find the smiles strings. Defaults to None. | None |
| smiles | Optional[list[str]] | A list of smiles strings to generate fingerprints for. Defaults to None. | None |
| radius | int | ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2. | 2 |
| hash_length | int | The length in number of vector elements of the fingerprint. Defaults to 2048. | 2048 |
| return_df | bool | Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False. | False |
| return_np | bool | Whether to return a numpy array rather than a list of bit vectors. Defalts to False. | None |

**Returns:**

| Type | Description |
|---|---|
| Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray] | Depends on the return type asked for |

**Raises:**

| Type | Description |
|---|---|
| RuntimeError | If incompatible inputs are given |

??? example "View Source"
        def get_ecfp(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            radius: int = 2,

            hash_length: int = 2048,

            return_df: bool = False,

            return_np: bool = False,

            n_threads: int = 8,

            **kwargs,

        ) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:

            """

            Function to generate ECFP representations from smiles

            Args:

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.

                hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Examples:

            ```python

            > get_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            (<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>,)

            > get_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)

            ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = df[smiles_column].to_list()

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            log.info(f"Making ECFP fingerprints for {len(smiles)} molecules")

            fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

            if return_np is True and return_df is False:

                fps = fp_gen.GetFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                fps_np = list_of_bitvects_to_numpy_arrays(fps)

                return fps_np

            elif return_df is True and return_np is False:

                fps = fp_gen.GetFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                fps_ll = list_of_bitvects_to_list_of_lists(fps)

                fps_df = pd.DataFrame(

                    fps_ll,

                    columns=[f"ecfp_bit_{ith}" for ith in range(hash_length)],

                )

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                fps = fp_gen.GetFingerprints(

                    [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads

                )

                return fps


### get_maccs

```python3
def get_maccs(
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs
) -> Union[Tuple[rdkit.DataStructs.cDataStructs.ExplicitBitVect], pandas.core.frame.DataFrame, numpy.ndarray]
```

Function to generate MACCS MDL keys representations from smiles

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data_df | Optional[pd.DataFrame] | Dataframe containing at least the smiles strings to use.<br>If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None. | None |
| smiles_column | Optional[str] | Needed if data_df is given to define which column to find the smiles strings. Defaults to None. | None |
| smiles | Optional[list[str]] | A list of smiles strings to generate fingerprints for. Defaults to None. | None |
| return_df | bool | Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False. | False |
| return_np | bool | Whether to return a numpy array rather than a list of bit vectors. Defalts to False. | None |

**Returns:**

| Type | Description |
|---|---|
| Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray] | Depends on the return type asked for |

**Raises:**

| Type | Description |
|---|---|
| RuntimeError | If incompatible inputs are given |

??? example "View Source"
        def get_maccs(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:

            """

            Function to generate MACCS MDL keys representations from smiles

            Args:

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Doctest:

            ```python

            > get_maccs(smiles=["c1ccccc1C"]) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]

            > get_maccs(smiles=["c1ccccc1C"])[0].GetOnBits()

            (160, 162, 163, 165)

            ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = df[smiles_column].to_list()

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            log.info(f"Making MACCS fingerprints for {len(smiles)} molecules")

            fps = [

                MACCSkeys.GenMACCSKeys(ent)

                for ent in [Chem.MolFromSmiles(smi) for smi in smiles]

            ]

            num_maccs_keys = len(fps[0])

            if return_np is True and return_df is False:

                fps_np = np.array([ent.ToList() for ent in fps])

                return fps_np

            elif return_df is True and return_np is False:

                fps_ll = [ent.ToList() for ent in fps]

                fps_df = pd.DataFrame(

                    fps_ll,

                    columns=[f"maccs_bit_{ith}" for ith in range(num_maccs_keys)],

                )

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                return fps


### get_nlp_smiles_rep

```python3
def get_nlp_smiles_rep(
    model_name: str = 'Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset',
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs
) -> Union[pandas.core.frame.DataFrame, numpy.ndarray]
```

Function to generate the NLP embedding representations from smiles using a transformer model

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model_name | str | the model name to use, default Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset apache 2.0 license accessed 30/10/24 https://huggingface.co/Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset | None |
| data_df | Optional[pd.DataFrame] | Dataframe containing at least the smiles strings to use.<br>If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None. | None |
| smiles_column | Optional[str] | Needed if data_df is given to define which column to find the smiles strings. Defaults to None. | None |
| smiles | Optional[list[str]] | A list of smiles strings to generate fingerprints for. Defaults to None. | None |
| return_df | bool | Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False. | False |
| return_np | bool | Whether to return a numpy array rather than a list of bit vectors. Defalts to False. | None |
| combine_strategy | Union[str, int] | How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean" | None |

**Returns:**

| Type | Description |
|---|---|
| Union[List[dict], pd.DataFrame, np.ndarray] | Depends on the return type asked for |

**Raises:**

| Type | Description |
|---|---|
| RuntimeError | If incompatible inputs are given |

??? example "View Source"
        def get_nlp_smiles_rep(

            model_name: str = "Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset",

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[pd.DataFrame, np.ndarray]:

            """

            Function to generate the NLP embedding representations from smiles using a transformer model

            Args:

                model_name (str): the model name to use, default Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset apache 2.0 license accessed 30/10/24 https://huggingface.co/Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

                combine_strategy (Union[str, int]): How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean"

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = list(df[smiles_column])

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            input_n = len(smiles)

            valid_smiles = validate_smiles(smiles, return_failed_as_None=False)

            if len(valid_smiles) != input_n:

                raise RuntimeError(

                    f"ERROR - only {(len(valid_smiles) / input_n) * 100.0}% of the input smiles were successfully read. Please correct the invalid smiles"

                )

            log.info(

                f"Making NLP embeddings using model {model_name} for {len(smiles)} molecules"

            )

            model = SentenceTransformer(model_name, **kwargs)

            embedding_rep = model.encode(smiles, **kwargs)

            if return_np is True and return_df is False:

                return embedding_rep

            elif return_df is True and return_np is False:

                fps_df = pd.DataFrame(embedding_rep)

                fps_df.columns = [

                    f"embedding_{model_name.replace('/', '-')}_{ith}"

                    for ith in range(fps_df.shape[1])

                ]

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                return embedding_rep


### get_rdkit_descriptors

```python3
def get_rdkit_descriptors(
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs
) -> Union[List[dict], pandas.core.frame.DataFrame, numpy.ndarray]
```

Function to generate Rdkit descriptor representations from smiles

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data_df | Optional[pd.DataFrame] | Dataframe containing at least the smiles strings to use.<br>If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None. | None |
| smiles_column | Optional[str] | Needed if data_df is given to define which column to find the smiles strings. Defaults to None. | None |
| smiles | Optional[list[str]] | A list of smiles strings to generate fingerprints for. Defaults to None. | None |
| return_df | bool | Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False. | False |
| return_np | bool | Whether to return a numpy array rather than a list of bit vectors. Defalts to False. | None |

**Returns:**

| Type | Description |
|---|---|
| Union[List[dict], pd.DataFrame, np.ndarray] | Depends on the return type asked for |

**Raises:**

| Type | Description |
|---|---|
| RuntimeError | If incompatible inputs are given |

??? example "View Source"
        def get_rdkit_descriptors(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[List[dict], pd.DataFrame, np.ndarray]:

            """

            Function to generate Rdkit descriptor representations from smiles

            Args:

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Example:

            ```python

            > type(get_rdkit_descriptors(smiles=["c1ccccc1C"])[0])

            <class 'dict'>

            > len(get_rdkit_descriptors(smiles=["c1ccccc1C"], return_df=True).columns)

            210

            ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    smiles = df[smiles_column].to_list()

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                df = data_df

            log.info(f"Making RDKit descriptors for {len(smiles)} molecules")

            fps = [

                CalcMolDescriptors(ent) for ent in [Chem.MolFromSmiles(smi) for smi in smiles]

            ]

            # make the order consistent sort on the keys in the dictionary important for the numpy return

            fps = [dict(sorted(ent.items())) for ent in fps]

            if return_np is True and return_df is False:

                fps_np = pd.DataFrame(fps).values

                return fps_np

            elif return_df is True and return_np is False:

                fps_df = pd.DataFrame(fps)

                fps_df.columns = [f"rdkit_descriptor_{name}" for name in fps_df.columns]

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                return fps


### get_t5_smiles_rep

```python3
def get_t5_smiles_rep(
    model_name: str = 'laituan245/molt5-large-smiles2caption',
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs
) -> Union[List[dict], pandas.core.frame.DataFrame, numpy.ndarray]
```

Function to generate the T5 embedding representations from smiles using a transformer model. We use the Apache - 2.0 licened model (accessed 7/10/24)

from laituan245/molt5-large-smiles2caption https://huggingface.co/laituan245/molt5-large-smiles2caption. This function calls get_nlp_smiles_rep using this model.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| model_name | str | the model name to use, | None |
| model_type | str | The model type to use i.e. bert, roberta or gpt2. | None |
| data_df | Optional[pd.DataFrame] | Dataframe containing at least the smiles strings to use.<br>If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None. | None |
| smiles_column | Optional[str] | Needed if data_df is given to define which column to find the smiles strings. Defaults to None. | None |
| smiles | Optional[list[str]] | A list of smiles strings to generate fingerprints for. Defaults to None. | None |
| return_df | bool | Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False. | False |
| return_np | bool | Whether to return a numpy array rather than a list of bit vectors. Defalts to False. | None |
| combine_strategy | Union[str, int] | How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean" | None |

**Returns:**

| Type | Description |
|---|---|
| Union[List[dict], pd.DataFrame, np.ndarray] | Depends on the return type asked for |

**Raises:**

| Type | Description |
|---|---|
| RuntimeError | If incompatible inputs are given |

??? example "View Source"
        def get_t5_smiles_rep(

            model_name: str = "laituan245/molt5-large-smiles2caption",

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[List[dict], pd.DataFrame, np.ndarray]:

            """

            Function to generate the T5 embedding representations from smiles using a transformer model. We use the Apache - 2.0 licened model (accessed 7/10/24)

            from laituan245/molt5-large-smiles2caption https://huggingface.co/laituan245/molt5-large-smiles2caption. This function calls get_nlp_smiles_rep using this model.

            Args:

                model_name (str): the model name to use,

                model_type (str): The model type to use i.e. bert, roberta or gpt2.

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

                combine_strategy (Union[str, int]): How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean"

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            """

            embedding_rep = get_nlp_smiles_rep(

                model_name=model_name,

                data_df=data_df,

                smiles_column=smiles_column,

                smiles=smiles,

                return_df=return_df,

                return_np=return_np,

                **kwargs,

            )

            return embedding_rep


### list_of_bitvects_to_list_of_lists

```python3
def list_of_bitvects_to_list_of_lists(
    bvs: List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
) -> List[List[int]]
```

Function to convert list of explicit bitvects from RDKit to list of lists. Note that at the time of writing RDKit has functions to do this one at a time but not in batches.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| bvs | List[cDataStructs.ExplicitBitVect] | List of bitvects from RDKit | None |

**Returns:**

| Type | Description |
|---|---|
| List[List[int]] | list of lists of integer binary values |

??? example "View Source"
        def list_of_bitvects_to_list_of_lists(

            bvs: List[cDataStructs.ExplicitBitVect],

        ) -> List[List[int]]:

            """

            Function to convert list of explicit bitvects from RDKit to list of lists. Note that at the time of writing RDKit has functions to do this one at a time but not in batches.

            Args:

                bvs (List[cDataStructs.ExplicitBitVect]): List of bitvects from RDKit

            Returns:

               List[List[int]]: list of lists of integer binary values

            Doctest:

            > list_of_bitvects_to_list_of_lists([cDataStructs.CreateFromBitString("1011")]) # doctest: +NORMALIZE_WHITESPACE

            [[1, 0, 1, 1]]

            """

            return [[int(ent) for ent in list(ent.ToBitString())] for ent in bvs]


### list_of_bitvects_to_numpy_arrays

```python3
def list_of_bitvects_to_numpy_arrays(
    bvs: List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
) -> numpy.ndarray
```

Function to convert list of explicit bitvectirs from RDKit to numpy arrays. Note that at the time of writing RDKit has functions to do this one at a time but not in batches.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| bvs | List[cDataStructs.ExplicitBitVect] | List of bitvects from RDKit | None |

**Returns:**

| Type | Description |
|---|---|
| np.ndarray | Numpy array of the bit vector arrays |

??? example "View Source"
        def list_of_bitvects_to_numpy_arrays(

            bvs: List[cDataStructs.ExplicitBitVect],

        ) -> np.ndarray:

            """

            Function to convert list of explicit bitvectirs from RDKit to numpy arrays. Note that at the time of writing RDKit has functions to do this one at a time but not in batches.

            Args:

                bvs (List[cDataStructs.ExplicitBitVect]): List of bitvects from RDKit

            Returns:

                np.ndarray: Numpy array of the bit vector arrays

            Doctest:

            > list_of_bitvects_to_numpy_arrays([cDataStructs.CreateFromBitString("1011")]) # doctest: +NORMALIZE_WHITESPACE

            array([[1, 0, 1, 1]], dtype=uint8)

            """

            return np.array(

                [[int(ent) for ent in list(ent.ToBitString())] for ent in bvs]

            ).astype("uint8")


### validate_smile

```python3
def validate_smile(
    smile: str,
    canonicalize: bool = True
) -> Optional[str]
```

Function to validate a single smiles string. This differs from get_valid_smiles as

it operates on a single smiles string and garuntees a return as if the smiles is invalid it returns None.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| smile | str | smiles string to check if it is valid | None |
| canaonicalize | bool | whether to return the input smiles or a canonicalized version for a valid smiles string | None |

**Returns:**

| Type | Description |
|---|---|
| Union[None, str] | None for a invalid smiles string and a smiles string for a valid one |

??? example "View Source"
        def validate_smile(smile: str, canonicalize: bool = True) -> Union[None, str]:

            """

            Function to validate a single smiles string. This differs from get_valid_smiles as

            it operates on a single smiles string and garuntees a return as if the smiles is invalid it returns None.

            Args:

                smile (str): smiles string to check if it is valid

                canaonicalize (bool): whether to return the input smiles or a canonicalized version for a valid smiles string

            Returns:

                Union[None, str]: None for a invalid smiles string and a smiles string for a valid one

            """

            try:

                m = Chem.MolFromSmiles(smile)

            except Exception:

                log.error("Exception when converting smiles to RDKit molecule object")

                return None

            if m is None:

                log.error(f"SMILES string: {smile} is invalid in RDKit and will be skipped")

                return None

            if canonicalize is True:

                return Chem.MolToSmiles(m)

            else:

                return smile


### validate_smiles

```python3
def validate_smiles(
    smiles: List[str],
    return_failed_as_None: bool = True,
    canonicalize: bool = True
) -> List[Optional[str]]
```

Function to validate a list of m smiles strings. This differs from get_valid_smiles as

it can guarntee a return, if return_failed_as_None is True (defualt), as if the smiles
is invalid it returns None.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| smile | str | smiles string to check if it is valid | None |
| return_failed_as_None | bool | whether to return None if the smiles string is invalid | None |
| canaonicalize | bool | whether to return the input smiles or a canonicalized version for a valid smiles string | None |

**Returns:**

| Type | Description |
|---|---|
| List[Union[None, str]] | None for a invalid smiles string and a smiles string for a valid one |

??? example "View Source"
        def validate_smiles(

            smiles: List[str], return_failed_as_None: bool = True, canonicalize: bool = True

        ) -> List[Union[None, str]]:

            """

            Function to validate a list of m smiles strings. This differs from get_valid_smiles as

            it can guarntee a return, if return_failed_as_None is True (defualt), as if the smiles

            is invalid it returns None.

            Args:

                smile (str): smiles string to check if it is valid

                return_failed_as_None (bool): whether to return None if the smiles string is invalid

                canaonicalize (bool): whether to return the input smiles or a canonicalized version for a valid smiles string

            Returns:

                List[Union[None, str]]: None for a invalid smiles string and a smiles string for a valid one

            """

            if return_failed_as_None is True:

                return [validate_smile(smile, canonicalize=canonicalize) for smile in smiles]

            else:

                tmp = [validate_smile(smile, canonicalize=canonicalize) for smile in smiles]

                return [ent for ent in tmp if ent is not None]


### validate_smiles_and_get_ecfp

```python3
def validate_smiles_and_get_ecfp(
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    radius: int = 2,
    hash_length: int = 2048,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs
) -> Union[Tuple[rdkit.DataStructs.cDataStructs.ExplicitBitVect], pandas.core.frame.DataFrame, numpy.ndarray]
```

Function to generate ECFP representations from smiles

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data_df | Optional[pd.DataFrame] | Dataframe containing at least the smiles strings to use.<br>If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None. | None |
| smiles_column | Optional[str] | Needed if data_df is given to define which column to find the smiles strings. Defaults to None. | None |
| smiles | Optional[list[str]] | A list of smiles strings to generate fingerprints for. Defaults to None. | None |
| radius | int | ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2. | 2 |
| hash_length | int | The length in number of vector elements of the fingerprint. Defaults to 2048. | 2048 |
| return_df | bool | Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False. | False |
| return_np | bool | Whether to return a numpy array rather than a list of bit vectors. Defalts to False. | None |

**Returns:**

| Type | Description |
|---|---|
| Union[List[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray] | Depends on the return type asked for |

**Raises:**

| Type | Description |
|---|---|
| RuntimeError | If incompatible inputs are given |

??? example "View Source"
        def validate_smiles_and_get_ecfp(

            data_df: Optional[pd.DataFrame] = None,

            smiles_column: Optional[str] = None,

            smiles: Optional[List[str]] = None,

            radius: int = 2,

            hash_length: int = 2048,

            return_df: bool = False,

            return_np: bool = False,

            **kwargs,

        ) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:

            """

            Function to generate ECFP representations from smiles

            Args:

                data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.

                 If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.

                smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.

                smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.

                radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diamater i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.

                hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.

                return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.

                return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defalts to False.

            Raises:

                RuntimeError: If incompatible inputs are given

            Returns:

                Union[List[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

            Example:

            ```python

            > validate_smiles_and_get_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]

            > validate_smiles_and_get_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE

            array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)

            ```

            """

            if sum([return_df, return_np]) > 1:

                raise RuntimeError(

                    f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."

                )

            if smiles is None:

                if all(ent is not None for ent in [data_df, smiles_column]):

                    df = data_df.copy()

                    input_n = len(df.index)

                    smiles = validate_smiles(df[smiles_column].to_list())

                    df["ECFP_smiles_standardized"] = smiles

                    df = df.dropna(axis=0, subset=["ECFP_smiles_standardized"])

                    smiles = df["ECFP_smiles_standardized"].to_list()

                    log.info(

                        f"{(len(smiles) / input_n) * 100.0}% ({len(smiles)} out of {input_n}) of the input smiles were successfully read"

                    )

                else:

                    raise RuntimeError(

                        "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"

                    )

            else:

                input_n = len(smiles)

                df = data_df

                smiles = validate_smiles(smiles, return_failed_as_None=False)

                log.info(

                    f"{(len(smiles) / input_n) * 100.0}% of the input smiles were successfully read"

                )

            fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

            if return_np is True and return_df is False:

                fps_np = np.array(

                    [

                        fp_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))

                        if smi is not None

                        else None

                        for smi in smiles

                    ]

                )

                return fps_np

            elif return_df is True and return_np is False:

                fps_df = pd.DataFrame(

                    [

                        fp_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))

                        if smi is not None

                        else None

                        for smi in smiles

                    ],

                    columns=[f"ecfp_bit_{ith}" for ith in range(hash_length)],

                )

                if df is not None:

                    return pd.concat([df, fps_df], axis=1)

                else:

                    return fps_df

            else:

                fps = [

                    fp_gen.GetFingerprint(Chem.MolFromSmiles(smi)) if smi is not None else None

                    for smi in smiles

                ]

                return fps

## Classes

### CECFPConverter

```python3
class CECFPConverter(
    hash_length: int = 2048,
    radius: int = 2,
    smiles_col: str = 'smiles'
)
```

This is a pipeline module class for converting SMILES strings to count ECFP fingerprints. It can be directly built into a sklearn pipeline.

For example:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from redxregressors import featurization
model = Pipeline([('smiles_converter', featurization.CECFPConverter()), ('scaler', StandardScaler()), ('RF', RandomForestClassifier())])

#### Ancestors (in MRO)

* sklearn.base.BaseEstimator
* sklearn.utils._estimator_html_repr._HTMLDocumentationLinkMixin
* sklearn.utils._metadata_requests._MetadataRequester
* sklearn.base.TransformerMixin
* sklearn.utils._set_output._SetOutputMixin

#### Methods


#### fit

```python3
def fit(
    self,
    X,
    y=None
)
```

??? example "View Source"
            def fit(self, X, y=None):

                return self


#### fit_transform

```python3
def fit_transform(
    self,
    X,
    y=None,
    **fit_params
)
```

Fit to data, then transform it.

Fits transformer to `X` and `y` with optional parameters `fit_params`
and returns a transformed version of `X`.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| X | array-like of shape (n_samples, n_features) | Input samples. | None |
| y | array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None | Target values (None for unsupervised transformations). | None |
| **fit_params | dict | Additional fit parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| ndarray array of shape (n_samples, n_features_new) | Transformed array. |

??? example "View Source"
            def fit_transform(self, X, y=None, **fit_params):

                """

                Fit to data, then transform it.

                Fits transformer to `X` and `y` with optional parameters `fit_params`

                and returns a transformed version of `X`.

                Parameters

                ----------

                X : array-like of shape (n_samples, n_features)

                    Input samples.

                y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \

                        default=None

                    Target values (None for unsupervised transformations).

                **fit_params : dict

                    Additional fit parameters.

                Returns

                -------

                X_new : ndarray array of shape (n_samples, n_features_new)

                    Transformed array.

                """

                # non-optimized default implementation; override when a better

                # method is possible for a given clustering algorithm

                # we do not route parameters here, since consumers don't route. But

                # since it's possible for a `transform` method to also consume

                # metadata, we check if that's the case, and we raise a warning telling

                # users that they should implement a custom `fit_transform` method

                # to forward metadata to `transform` as well.

                #

                # For that, we calculate routing and check if anything would be routed

                # to `transform` if we were to route them.

                if _routing_enabled():

                    transform_params = self.get_metadata_routing().consumes(

                        method="transform", params=fit_params.keys()

                    )

                    if transform_params:

                        warnings.warn(

                            (

                                f"This object ({self.__class__.__name__}) has a `transform`"

                                " method which consumes metadata, but `fit_transform` does not"

                                " forward metadata to `transform`. Please implement a custom"

                                " `fit_transform` method to forward metadata to `transform` as"

                                " well. Alternatively, you can explicitly do"

                                " `set_transform_request`and set all values to `False` to"

                                " disable metadata routed to `transform`, if that's an option."

                            ),

                            UserWarning,

                        )

                if y is None:

                    # fit method of arity 1 (unsupervised transformation)

                    return self.fit(X, **fit_params).transform(X)

                else:

                    # fit method of arity 2 (supervised transformation)

                    return self.fit(X, y, **fit_params).transform(X)


#### get_metadata_routing

```python3
def get_metadata_routing(
    self
)
```

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

**Returns:**

| Type | Description |
|---|---|
| MetadataRequest | A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating<br>routing information. |

??? example "View Source"
            def get_metadata_routing(self):

                """Get metadata routing of this object.

                Please check :ref:`User Guide <metadata_routing>` on how the routing

                mechanism works.

                Returns

                -------

                routing : MetadataRequest

                    A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating

                    routing information.

                """

                return self._get_metadata_request()


#### get_params

```python3
def get_params(
    self,
    deep=True
)
```

Get parameters for this estimator.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| deep | bool, default=True | If True, will return the parameters for this estimator and<br>contained subobjects that are estimators. | None |

**Returns:**

| Type | Description |
|---|---|
| dict | Parameter names mapped to their values. |

??? example "View Source"
            def get_params(self, deep=True):

                """

                Get parameters for this estimator.

                Parameters

                ----------

                deep : bool, default=True

                    If True, will return the parameters for this estimator and

                    contained subobjects that are estimators.

                Returns

                -------

                params : dict

                    Parameter names mapped to their values.

                """

                out = dict()

                for key in self._get_param_names():

                    value = getattr(self, key)

                    if deep and hasattr(value, "get_params") and not isinstance(value, type):

                        deep_items = value.get_params().items()

                        out.update((key + "__" + k, val) for k, val in deep_items)

                    out[key] = value

                return out


#### set_output

```python3
def set_output(
    self,
    *,
    transform=None
)
```

Set output container.

See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
for an example on how to use the API.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| transform | {"default", "pandas", "polars"}, default=None | Configure output of `transform` and `fit_transform`.<br><br>- `"default"`: Default output format of a transformer<br>- `"pandas"`: DataFrame output<br>- `"polars"`: Polars output<br>- `None`: Transform configuration is unchanged<br><br>.. versionadded:: 1.4<br>    `"polars"` option was added. | output |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            @available_if(_auto_wrap_is_configured)

            def set_output(self, *, transform=None):

                """Set output container.

                See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`

                for an example on how to use the API.

                Parameters

                ----------

                transform : {"default", "pandas", "polars"}, default=None

                    Configure output of `transform` and `fit_transform`.

                    - `"default"`: Default output format of a transformer

                    - `"pandas"`: DataFrame output

                    - `"polars"`: Polars output

                    - `None`: Transform configuration is unchanged

                    .. versionadded:: 1.4

                        `"polars"` option was added.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if transform is None:

                    return self

                if not hasattr(self, "_sklearn_output_config"):

                    self._sklearn_output_config = {}

                self._sklearn_output_config["transform"] = transform

                return self


#### set_params

```python3
def set_params(
    self,
    **params
)
```

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| **params | dict | Estimator parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            def set_params(self, **params):

                """Set the parameters of this estimator.

                The method works on simple estimators as well as on nested objects

                (such as :class:`~sklearn.pipeline.Pipeline`). The latter have

                parameters of the form ``<component>__<parameter>`` so that it's

                possible to update each component of a nested object.

                Parameters

                ----------

                **params : dict

                    Estimator parameters.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if not params:

                    # Simple optimization to gain speed (inspect is slow)

                    return self

                valid_params = self.get_params(deep=True)

                nested_params = defaultdict(dict)  # grouped by prefix

                for key, value in params.items():

                    key, delim, sub_key = key.partition("__")

                    if key not in valid_params:

                        local_valid_params = self._get_param_names()

                        raise ValueError(

                            f"Invalid parameter {key!r} for estimator {self}. "

                            f"Valid parameters are: {local_valid_params!r}."

                        )

                    if delim:

                        nested_params[key][sub_key] = value

                    else:

                        setattr(self, key, value)

                        valid_params[key] = value

                for key, sub_params in nested_params.items():

                    valid_params[key].set_params(**sub_params)

                return self


#### transform

```python3
def transform(
    self,
    X
) -> numpy.ndarray
```

??? example "View Source"
            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to CECFP representations .....")

                if len(X.shape) == 1:

                    fps = get_count_ecfp(

                        smiles=X,

                        hash_length=self.hash_length,

                        radius=self.radius,

                        return_np=True,

                    )

                    return fps

                else:

                    fps = get_count_ecfp(

                        smiles=X[self.smiles_col],

                        hash_length=self.hash_length,

                        radius=self.radius,

                        return_np=True,

                    )

                    return fps

### ECFPConverter

```python3
class ECFPConverter(
    hash_length: int = 2048,
    radius: int = 2,
    smiles_col: str = 'smiles'
)
```

This is a pipeline module class for converting SMILES strings to ECFP fingerprints. It can be directly built into a sklearn pipeline.

For example:
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from redxregressors import featurization
model = Pipeline([('smiles_converter', featurization.ECFPConverter()), ('RF', RandomForestClassifier())])

#### Ancestors (in MRO)

* sklearn.base.BaseEstimator
* sklearn.utils._estimator_html_repr._HTMLDocumentationLinkMixin
* sklearn.utils._metadata_requests._MetadataRequester
* sklearn.base.TransformerMixin
* sklearn.utils._set_output._SetOutputMixin

#### Methods


#### fit

```python3
def fit(
    self,
    X,
    y=None
)
```

??? example "View Source"
            def fit(self, X, y=None):

                return self


#### fit_transform

```python3
def fit_transform(
    self,
    X,
    y=None,
    **fit_params
)
```

Fit to data, then transform it.

Fits transformer to `X` and `y` with optional parameters `fit_params`
and returns a transformed version of `X`.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| X | array-like of shape (n_samples, n_features) | Input samples. | None |
| y | array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None | Target values (None for unsupervised transformations). | None |
| **fit_params | dict | Additional fit parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| ndarray array of shape (n_samples, n_features_new) | Transformed array. |

??? example "View Source"
            def fit_transform(self, X, y=None, **fit_params):

                """

                Fit to data, then transform it.

                Fits transformer to `X` and `y` with optional parameters `fit_params`

                and returns a transformed version of `X`.

                Parameters

                ----------

                X : array-like of shape (n_samples, n_features)

                    Input samples.

                y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \

                        default=None

                    Target values (None for unsupervised transformations).

                **fit_params : dict

                    Additional fit parameters.

                Returns

                -------

                X_new : ndarray array of shape (n_samples, n_features_new)

                    Transformed array.

                """

                # non-optimized default implementation; override when a better

                # method is possible for a given clustering algorithm

                # we do not route parameters here, since consumers don't route. But

                # since it's possible for a `transform` method to also consume

                # metadata, we check if that's the case, and we raise a warning telling

                # users that they should implement a custom `fit_transform` method

                # to forward metadata to `transform` as well.

                #

                # For that, we calculate routing and check if anything would be routed

                # to `transform` if we were to route them.

                if _routing_enabled():

                    transform_params = self.get_metadata_routing().consumes(

                        method="transform", params=fit_params.keys()

                    )

                    if transform_params:

                        warnings.warn(

                            (

                                f"This object ({self.__class__.__name__}) has a `transform`"

                                " method which consumes metadata, but `fit_transform` does not"

                                " forward metadata to `transform`. Please implement a custom"

                                " `fit_transform` method to forward metadata to `transform` as"

                                " well. Alternatively, you can explicitly do"

                                " `set_transform_request`and set all values to `False` to"

                                " disable metadata routed to `transform`, if that's an option."

                            ),

                            UserWarning,

                        )

                if y is None:

                    # fit method of arity 1 (unsupervised transformation)

                    return self.fit(X, **fit_params).transform(X)

                else:

                    # fit method of arity 2 (supervised transformation)

                    return self.fit(X, y, **fit_params).transform(X)


#### get_metadata_routing

```python3
def get_metadata_routing(
    self
)
```

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

**Returns:**

| Type | Description |
|---|---|
| MetadataRequest | A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating<br>routing information. |

??? example "View Source"
            def get_metadata_routing(self):

                """Get metadata routing of this object.

                Please check :ref:`User Guide <metadata_routing>` on how the routing

                mechanism works.

                Returns

                -------

                routing : MetadataRequest

                    A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating

                    routing information.

                """

                return self._get_metadata_request()


#### get_params

```python3
def get_params(
    self,
    deep=True
)
```

Get parameters for this estimator.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| deep | bool, default=True | If True, will return the parameters for this estimator and<br>contained subobjects that are estimators. | None |

**Returns:**

| Type | Description |
|---|---|
| dict | Parameter names mapped to their values. |

??? example "View Source"
            def get_params(self, deep=True):

                """

                Get parameters for this estimator.

                Parameters

                ----------

                deep : bool, default=True

                    If True, will return the parameters for this estimator and

                    contained subobjects that are estimators.

                Returns

                -------

                params : dict

                    Parameter names mapped to their values.

                """

                out = dict()

                for key in self._get_param_names():

                    value = getattr(self, key)

                    if deep and hasattr(value, "get_params") and not isinstance(value, type):

                        deep_items = value.get_params().items()

                        out.update((key + "__" + k, val) for k, val in deep_items)

                    out[key] = value

                return out


#### set_output

```python3
def set_output(
    self,
    *,
    transform=None
)
```

Set output container.

See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
for an example on how to use the API.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| transform | {"default", "pandas", "polars"}, default=None | Configure output of `transform` and `fit_transform`.<br><br>- `"default"`: Default output format of a transformer<br>- `"pandas"`: DataFrame output<br>- `"polars"`: Polars output<br>- `None`: Transform configuration is unchanged<br><br>.. versionadded:: 1.4<br>    `"polars"` option was added. | output |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            @available_if(_auto_wrap_is_configured)

            def set_output(self, *, transform=None):

                """Set output container.

                See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`

                for an example on how to use the API.

                Parameters

                ----------

                transform : {"default", "pandas", "polars"}, default=None

                    Configure output of `transform` and `fit_transform`.

                    - `"default"`: Default output format of a transformer

                    - `"pandas"`: DataFrame output

                    - `"polars"`: Polars output

                    - `None`: Transform configuration is unchanged

                    .. versionadded:: 1.4

                        `"polars"` option was added.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if transform is None:

                    return self

                if not hasattr(self, "_sklearn_output_config"):

                    self._sklearn_output_config = {}

                self._sklearn_output_config["transform"] = transform

                return self


#### set_params

```python3
def set_params(
    self,
    **params
)
```

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| **params | dict | Estimator parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            def set_params(self, **params):

                """Set the parameters of this estimator.

                The method works on simple estimators as well as on nested objects

                (such as :class:`~sklearn.pipeline.Pipeline`). The latter have

                parameters of the form ``<component>__<parameter>`` so that it's

                possible to update each component of a nested object.

                Parameters

                ----------

                **params : dict

                    Estimator parameters.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if not params:

                    # Simple optimization to gain speed (inspect is slow)

                    return self

                valid_params = self.get_params(deep=True)

                nested_params = defaultdict(dict)  # grouped by prefix

                for key, value in params.items():

                    key, delim, sub_key = key.partition("__")

                    if key not in valid_params:

                        local_valid_params = self._get_param_names()

                        raise ValueError(

                            f"Invalid parameter {key!r} for estimator {self}. "

                            f"Valid parameters are: {local_valid_params!r}."

                        )

                    if delim:

                        nested_params[key][sub_key] = value

                    else:

                        setattr(self, key, value)

                        valid_params[key] = value

                for key, sub_params in nested_params.items():

                    valid_params[key].set_params(**sub_params)

                return self


#### transform

```python3
def transform(
    self,
    X
) -> numpy.ndarray
```

??? example "View Source"
            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to ECFP representations .....")

                if len(X.shape) == 1:

                    fps = get_ecfp(

                        smiles=X,

                        hash_length=self.hash_length,

                        radius=self.radius,

                        return_np=True,

                    )

                    return fps

                else:

                    fps = get_ecfp(

                        smiles=X[self.smiles_col],

                        hash_length=self.hash_length,

                        radius=self.radius,

                        return_np=True,

                    )

                    return fps

### GCN_featurize

```python3
class GCN_featurize(
    smiles: list[str],
    targets: list[float],
    explicit_h: bool = True,
    include_aromaticity: bool = True,
    include_charge: bool = True,
    elements: Optional[List[str]] = None,
    train_fraction: Optional[float] = 0.8,
    test_fraction: Optional[float] = 0.1,
    validation_fraction: Optional[float] = 0.1,
    batch_size: Optional[int] = 1
)
```

Class to featurize molecules for use with a GCN model.

### GetAttentiveFPFeatures

```python3
class GetAttentiveFPFeatures(

)
```

Class for making the attentiveFP features. Its is based on the paper https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959 and the PyTorch Geometric documentation

https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/molecule_net.py accessed 21/10/24. The pytorch geometric code is licensed under
the MIT license which is given below.
Copyright (c) 2023 PyG Team <team@pyg.org>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

### MACCSConverter

```python3
class MACCSConverter(
    smiles_col: str = 'smiles'
)
```

This is a pipeline module class for converting SMILES strings to MACCS fingerprints. It can be directly built into a sklearn pipeline.

For example:
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from redxregressors import featurization
model = Pipeline([('smiles_converter', featurization.MACCSConverter()), ('RF', RandomForestClassifier())])

#### Ancestors (in MRO)

* sklearn.base.BaseEstimator
* sklearn.utils._estimator_html_repr._HTMLDocumentationLinkMixin
* sklearn.utils._metadata_requests._MetadataRequester
* sklearn.base.TransformerMixin
* sklearn.utils._set_output._SetOutputMixin

#### Methods


#### fit

```python3
def fit(
    self,
    X,
    y=None
)
```

??? example "View Source"
            def fit(self, X, y=None):

                return self


#### fit_transform

```python3
def fit_transform(
    self,
    X,
    y=None,
    **fit_params
)
```

Fit to data, then transform it.

Fits transformer to `X` and `y` with optional parameters `fit_params`
and returns a transformed version of `X`.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| X | array-like of shape (n_samples, n_features) | Input samples. | None |
| y | array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None | Target values (None for unsupervised transformations). | None |
| **fit_params | dict | Additional fit parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| ndarray array of shape (n_samples, n_features_new) | Transformed array. |

??? example "View Source"
            def fit_transform(self, X, y=None, **fit_params):

                """

                Fit to data, then transform it.

                Fits transformer to `X` and `y` with optional parameters `fit_params`

                and returns a transformed version of `X`.

                Parameters

                ----------

                X : array-like of shape (n_samples, n_features)

                    Input samples.

                y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \

                        default=None

                    Target values (None for unsupervised transformations).

                **fit_params : dict

                    Additional fit parameters.

                Returns

                -------

                X_new : ndarray array of shape (n_samples, n_features_new)

                    Transformed array.

                """

                # non-optimized default implementation; override when a better

                # method is possible for a given clustering algorithm

                # we do not route parameters here, since consumers don't route. But

                # since it's possible for a `transform` method to also consume

                # metadata, we check if that's the case, and we raise a warning telling

                # users that they should implement a custom `fit_transform` method

                # to forward metadata to `transform` as well.

                #

                # For that, we calculate routing and check if anything would be routed

                # to `transform` if we were to route them.

                if _routing_enabled():

                    transform_params = self.get_metadata_routing().consumes(

                        method="transform", params=fit_params.keys()

                    )

                    if transform_params:

                        warnings.warn(

                            (

                                f"This object ({self.__class__.__name__}) has a `transform`"

                                " method which consumes metadata, but `fit_transform` does not"

                                " forward metadata to `transform`. Please implement a custom"

                                " `fit_transform` method to forward metadata to `transform` as"

                                " well. Alternatively, you can explicitly do"

                                " `set_transform_request`and set all values to `False` to"

                                " disable metadata routed to `transform`, if that's an option."

                            ),

                            UserWarning,

                        )

                if y is None:

                    # fit method of arity 1 (unsupervised transformation)

                    return self.fit(X, **fit_params).transform(X)

                else:

                    # fit method of arity 2 (supervised transformation)

                    return self.fit(X, y, **fit_params).transform(X)


#### get_metadata_routing

```python3
def get_metadata_routing(
    self
)
```

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

**Returns:**

| Type | Description |
|---|---|
| MetadataRequest | A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating<br>routing information. |

??? example "View Source"
            def get_metadata_routing(self):

                """Get metadata routing of this object.

                Please check :ref:`User Guide <metadata_routing>` on how the routing

                mechanism works.

                Returns

                -------

                routing : MetadataRequest

                    A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating

                    routing information.

                """

                return self._get_metadata_request()


#### get_params

```python3
def get_params(
    self,
    deep=True
)
```

Get parameters for this estimator.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| deep | bool, default=True | If True, will return the parameters for this estimator and<br>contained subobjects that are estimators. | None |

**Returns:**

| Type | Description |
|---|---|
| dict | Parameter names mapped to their values. |

??? example "View Source"
            def get_params(self, deep=True):

                """

                Get parameters for this estimator.

                Parameters

                ----------

                deep : bool, default=True

                    If True, will return the parameters for this estimator and

                    contained subobjects that are estimators.

                Returns

                -------

                params : dict

                    Parameter names mapped to their values.

                """

                out = dict()

                for key in self._get_param_names():

                    value = getattr(self, key)

                    if deep and hasattr(value, "get_params") and not isinstance(value, type):

                        deep_items = value.get_params().items()

                        out.update((key + "__" + k, val) for k, val in deep_items)

                    out[key] = value

                return out


#### set_output

```python3
def set_output(
    self,
    *,
    transform=None
)
```

Set output container.

See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
for an example on how to use the API.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| transform | {"default", "pandas", "polars"}, default=None | Configure output of `transform` and `fit_transform`.<br><br>- `"default"`: Default output format of a transformer<br>- `"pandas"`: DataFrame output<br>- `"polars"`: Polars output<br>- `None`: Transform configuration is unchanged<br><br>.. versionadded:: 1.4<br>    `"polars"` option was added. | output |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            @available_if(_auto_wrap_is_configured)

            def set_output(self, *, transform=None):

                """Set output container.

                See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`

                for an example on how to use the API.

                Parameters

                ----------

                transform : {"default", "pandas", "polars"}, default=None

                    Configure output of `transform` and `fit_transform`.

                    - `"default"`: Default output format of a transformer

                    - `"pandas"`: DataFrame output

                    - `"polars"`: Polars output

                    - `None`: Transform configuration is unchanged

                    .. versionadded:: 1.4

                        `"polars"` option was added.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if transform is None:

                    return self

                if not hasattr(self, "_sklearn_output_config"):

                    self._sklearn_output_config = {}

                self._sklearn_output_config["transform"] = transform

                return self


#### set_params

```python3
def set_params(
    self,
    **params
)
```

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| **params | dict | Estimator parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            def set_params(self, **params):

                """Set the parameters of this estimator.

                The method works on simple estimators as well as on nested objects

                (such as :class:`~sklearn.pipeline.Pipeline`). The latter have

                parameters of the form ``<component>__<parameter>`` so that it's

                possible to update each component of a nested object.

                Parameters

                ----------

                **params : dict

                    Estimator parameters.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if not params:

                    # Simple optimization to gain speed (inspect is slow)

                    return self

                valid_params = self.get_params(deep=True)

                nested_params = defaultdict(dict)  # grouped by prefix

                for key, value in params.items():

                    key, delim, sub_key = key.partition("__")

                    if key not in valid_params:

                        local_valid_params = self._get_param_names()

                        raise ValueError(

                            f"Invalid parameter {key!r} for estimator {self}. "

                            f"Valid parameters are: {local_valid_params!r}."

                        )

                    if delim:

                        nested_params[key][sub_key] = value

                    else:

                        setattr(self, key, value)

                        valid_params[key] = value

                for key, sub_params in nested_params.items():

                    valid_params[key].set_params(**sub_params)

                return self


#### transform

```python3
def transform(
    self,
    X
) -> numpy.ndarray
```

??? example "View Source"
            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to MACCS representations .....")

                if len(X.shape) == 1:

                    fps = get_maccs(smiles=X, return_np=True)

                    return fps

                else:

                    fps = get_maccs(smiles=X[self.smiles_col], return_np=True)

                    return fps

### NLPConverter

```python3
class NLPConverter(
    model_name: str = 'laituan245/molt5-large-smiles2caption',
    smiles_col: str = 'smiles'
)
```

This is a pipeline module class for converting SMILES strings to sentence embeddings. It can be directly built into a sklearn pipeline.

For example:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from redxregressors import featurization
model = Pipeline([('smiles_converter', featurization.NLPConverter()), ('scaler', StandardScaler()), ('RF', RandomForestClassifier())])

#### Ancestors (in MRO)

* sklearn.base.BaseEstimator
* sklearn.utils._estimator_html_repr._HTMLDocumentationLinkMixin
* sklearn.utils._metadata_requests._MetadataRequester
* sklearn.base.TransformerMixin
* sklearn.utils._set_output._SetOutputMixin

#### Methods


#### fit

```python3
def fit(
    self,
    X,
    y=None
)
```

??? example "View Source"
            def fit(self, X, y=None):

                return self


#### fit_transform

```python3
def fit_transform(
    self,
    X,
    y=None,
    **fit_params
)
```

Fit to data, then transform it.

Fits transformer to `X` and `y` with optional parameters `fit_params`
and returns a transformed version of `X`.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| X | array-like of shape (n_samples, n_features) | Input samples. | None |
| y | array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None | Target values (None for unsupervised transformations). | None |
| **fit_params | dict | Additional fit parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| ndarray array of shape (n_samples, n_features_new) | Transformed array. |

??? example "View Source"
            def fit_transform(self, X, y=None, **fit_params):

                """

                Fit to data, then transform it.

                Fits transformer to `X` and `y` with optional parameters `fit_params`

                and returns a transformed version of `X`.

                Parameters

                ----------

                X : array-like of shape (n_samples, n_features)

                    Input samples.

                y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \

                        default=None

                    Target values (None for unsupervised transformations).

                **fit_params : dict

                    Additional fit parameters.

                Returns

                -------

                X_new : ndarray array of shape (n_samples, n_features_new)

                    Transformed array.

                """

                # non-optimized default implementation; override when a better

                # method is possible for a given clustering algorithm

                # we do not route parameters here, since consumers don't route. But

                # since it's possible for a `transform` method to also consume

                # metadata, we check if that's the case, and we raise a warning telling

                # users that they should implement a custom `fit_transform` method

                # to forward metadata to `transform` as well.

                #

                # For that, we calculate routing and check if anything would be routed

                # to `transform` if we were to route them.

                if _routing_enabled():

                    transform_params = self.get_metadata_routing().consumes(

                        method="transform", params=fit_params.keys()

                    )

                    if transform_params:

                        warnings.warn(

                            (

                                f"This object ({self.__class__.__name__}) has a `transform`"

                                " method which consumes metadata, but `fit_transform` does not"

                                " forward metadata to `transform`. Please implement a custom"

                                " `fit_transform` method to forward metadata to `transform` as"

                                " well. Alternatively, you can explicitly do"

                                " `set_transform_request`and set all values to `False` to"

                                " disable metadata routed to `transform`, if that's an option."

                            ),

                            UserWarning,

                        )

                if y is None:

                    # fit method of arity 1 (unsupervised transformation)

                    return self.fit(X, **fit_params).transform(X)

                else:

                    # fit method of arity 2 (supervised transformation)

                    return self.fit(X, y, **fit_params).transform(X)


#### get_metadata_routing

```python3
def get_metadata_routing(
    self
)
```

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

**Returns:**

| Type | Description |
|---|---|
| MetadataRequest | A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating<br>routing information. |

??? example "View Source"
            def get_metadata_routing(self):

                """Get metadata routing of this object.

                Please check :ref:`User Guide <metadata_routing>` on how the routing

                mechanism works.

                Returns

                -------

                routing : MetadataRequest

                    A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating

                    routing information.

                """

                return self._get_metadata_request()


#### get_params

```python3
def get_params(
    self,
    deep=True
)
```

Get parameters for this estimator.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| deep | bool, default=True | If True, will return the parameters for this estimator and<br>contained subobjects that are estimators. | None |

**Returns:**

| Type | Description |
|---|---|
| dict | Parameter names mapped to their values. |

??? example "View Source"
            def get_params(self, deep=True):

                """

                Get parameters for this estimator.

                Parameters

                ----------

                deep : bool, default=True

                    If True, will return the parameters for this estimator and

                    contained subobjects that are estimators.

                Returns

                -------

                params : dict

                    Parameter names mapped to their values.

                """

                out = dict()

                for key in self._get_param_names():

                    value = getattr(self, key)

                    if deep and hasattr(value, "get_params") and not isinstance(value, type):

                        deep_items = value.get_params().items()

                        out.update((key + "__" + k, val) for k, val in deep_items)

                    out[key] = value

                return out


#### set_output

```python3
def set_output(
    self,
    *,
    transform=None
)
```

Set output container.

See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
for an example on how to use the API.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| transform | {"default", "pandas", "polars"}, default=None | Configure output of `transform` and `fit_transform`.<br><br>- `"default"`: Default output format of a transformer<br>- `"pandas"`: DataFrame output<br>- `"polars"`: Polars output<br>- `None`: Transform configuration is unchanged<br><br>.. versionadded:: 1.4<br>    `"polars"` option was added. | output |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            @available_if(_auto_wrap_is_configured)

            def set_output(self, *, transform=None):

                """Set output container.

                See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`

                for an example on how to use the API.

                Parameters

                ----------

                transform : {"default", "pandas", "polars"}, default=None

                    Configure output of `transform` and `fit_transform`.

                    - `"default"`: Default output format of a transformer

                    - `"pandas"`: DataFrame output

                    - `"polars"`: Polars output

                    - `None`: Transform configuration is unchanged

                    .. versionadded:: 1.4

                        `"polars"` option was added.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if transform is None:

                    return self

                if not hasattr(self, "_sklearn_output_config"):

                    self._sklearn_output_config = {}

                self._sklearn_output_config["transform"] = transform

                return self


#### set_params

```python3
def set_params(
    self,
    **params
)
```

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| **params | dict | Estimator parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            def set_params(self, **params):

                """Set the parameters of this estimator.

                The method works on simple estimators as well as on nested objects

                (such as :class:`~sklearn.pipeline.Pipeline`). The latter have

                parameters of the form ``<component>__<parameter>`` so that it's

                possible to update each component of a nested object.

                Parameters

                ----------

                **params : dict

                    Estimator parameters.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if not params:

                    # Simple optimization to gain speed (inspect is slow)

                    return self

                valid_params = self.get_params(deep=True)

                nested_params = defaultdict(dict)  # grouped by prefix

                for key, value in params.items():

                    key, delim, sub_key = key.partition("__")

                    if key not in valid_params:

                        local_valid_params = self._get_param_names()

                        raise ValueError(

                            f"Invalid parameter {key!r} for estimator {self}. "

                            f"Valid parameters are: {local_valid_params!r}."

                        )

                    if delim:

                        nested_params[key][sub_key] = value

                    else:

                        setattr(self, key, value)

                        valid_params[key] = value

                for key, sub_params in nested_params.items():

                    valid_params[key].set_params(**sub_params)

                return self


#### transform

```python3
def transform(
    self,
    X
) -> numpy.ndarray
```

??? example "View Source"
            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to sentence embeddings .....")

                if len(X.shape) == 1:

                    embeddings = get_nlp_smiles_rep(model_name=self.model_name, smiles=X)

                    return embeddings

                else:

                    embeddings = get_nlp_smiles_rep(

                        model_name=self.model_name, smiles=X[self.smiles_col]

                    )

                    return embeddings

### RDKitDescriptorsConverter

```python3
class RDKitDescriptorsConverter(
    smiles_col: str = 'smiles'
)
```

This is a pipeline module class for converting SMILES strings to RDKit descriptors. It can be directly built into a sklearn pipeline.

For example:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from redxregressors import featurization
model = Pipeline([('smiles_converter', featurization.RDKitDescriptorsConverter()), ('scaler', StandardScaler()), ('RF', RandomForestClassifier())])

#### Ancestors (in MRO)

* sklearn.base.BaseEstimator
* sklearn.utils._estimator_html_repr._HTMLDocumentationLinkMixin
* sklearn.utils._metadata_requests._MetadataRequester
* sklearn.base.TransformerMixin
* sklearn.utils._set_output._SetOutputMixin

#### Methods


#### fit

```python3
def fit(
    self,
    X,
    y=None
)
```

??? example "View Source"
            def fit(self, X, y=None):

                return self


#### fit_transform

```python3
def fit_transform(
    self,
    X,
    y=None,
    **fit_params
)
```

Fit to data, then transform it.

Fits transformer to `X` and `y` with optional parameters `fit_params`
and returns a transformed version of `X`.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| X | array-like of shape (n_samples, n_features) | Input samples. | None |
| y | array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None | Target values (None for unsupervised transformations). | None |
| **fit_params | dict | Additional fit parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| ndarray array of shape (n_samples, n_features_new) | Transformed array. |

??? example "View Source"
            def fit_transform(self, X, y=None, **fit_params):

                """

                Fit to data, then transform it.

                Fits transformer to `X` and `y` with optional parameters `fit_params`

                and returns a transformed version of `X`.

                Parameters

                ----------

                X : array-like of shape (n_samples, n_features)

                    Input samples.

                y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \

                        default=None

                    Target values (None for unsupervised transformations).

                **fit_params : dict

                    Additional fit parameters.

                Returns

                -------

                X_new : ndarray array of shape (n_samples, n_features_new)

                    Transformed array.

                """

                # non-optimized default implementation; override when a better

                # method is possible for a given clustering algorithm

                # we do not route parameters here, since consumers don't route. But

                # since it's possible for a `transform` method to also consume

                # metadata, we check if that's the case, and we raise a warning telling

                # users that they should implement a custom `fit_transform` method

                # to forward metadata to `transform` as well.

                #

                # For that, we calculate routing and check if anything would be routed

                # to `transform` if we were to route them.

                if _routing_enabled():

                    transform_params = self.get_metadata_routing().consumes(

                        method="transform", params=fit_params.keys()

                    )

                    if transform_params:

                        warnings.warn(

                            (

                                f"This object ({self.__class__.__name__}) has a `transform`"

                                " method which consumes metadata, but `fit_transform` does not"

                                " forward metadata to `transform`. Please implement a custom"

                                " `fit_transform` method to forward metadata to `transform` as"

                                " well. Alternatively, you can explicitly do"

                                " `set_transform_request`and set all values to `False` to"

                                " disable metadata routed to `transform`, if that's an option."

                            ),

                            UserWarning,

                        )

                if y is None:

                    # fit method of arity 1 (unsupervised transformation)

                    return self.fit(X, **fit_params).transform(X)

                else:

                    # fit method of arity 2 (supervised transformation)

                    return self.fit(X, y, **fit_params).transform(X)


#### get_metadata_routing

```python3
def get_metadata_routing(
    self
)
```

Get metadata routing of this object.

Please check :ref:`User Guide <metadata_routing>` on how the routing
mechanism works.

**Returns:**

| Type | Description |
|---|---|
| MetadataRequest | A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating<br>routing information. |

??? example "View Source"
            def get_metadata_routing(self):

                """Get metadata routing of this object.

                Please check :ref:`User Guide <metadata_routing>` on how the routing

                mechanism works.

                Returns

                -------

                routing : MetadataRequest

                    A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating

                    routing information.

                """

                return self._get_metadata_request()


#### get_params

```python3
def get_params(
    self,
    deep=True
)
```

Get parameters for this estimator.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| deep | bool, default=True | If True, will return the parameters for this estimator and<br>contained subobjects that are estimators. | None |

**Returns:**

| Type | Description |
|---|---|
| dict | Parameter names mapped to their values. |

??? example "View Source"
            def get_params(self, deep=True):

                """

                Get parameters for this estimator.

                Parameters

                ----------

                deep : bool, default=True

                    If True, will return the parameters for this estimator and

                    contained subobjects that are estimators.

                Returns

                -------

                params : dict

                    Parameter names mapped to their values.

                """

                out = dict()

                for key in self._get_param_names():

                    value = getattr(self, key)

                    if deep and hasattr(value, "get_params") and not isinstance(value, type):

                        deep_items = value.get_params().items()

                        out.update((key + "__" + k, val) for k, val in deep_items)

                    out[key] = value

                return out


#### set_output

```python3
def set_output(
    self,
    *,
    transform=None
)
```

Set output container.

See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`
for an example on how to use the API.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| transform | {"default", "pandas", "polars"}, default=None | Configure output of `transform` and `fit_transform`.<br><br>- `"default"`: Default output format of a transformer<br>- `"pandas"`: DataFrame output<br>- `"polars"`: Polars output<br>- `None`: Transform configuration is unchanged<br><br>.. versionadded:: 1.4<br>    `"polars"` option was added. | output |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            @available_if(_auto_wrap_is_configured)

            def set_output(self, *, transform=None):

                """Set output container.

                See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`

                for an example on how to use the API.

                Parameters

                ----------

                transform : {"default", "pandas", "polars"}, default=None

                    Configure output of `transform` and `fit_transform`.

                    - `"default"`: Default output format of a transformer

                    - `"pandas"`: DataFrame output

                    - `"polars"`: Polars output

                    - `None`: Transform configuration is unchanged

                    .. versionadded:: 1.4

                        `"polars"` option was added.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if transform is None:

                    return self

                if not hasattr(self, "_sklearn_output_config"):

                    self._sklearn_output_config = {}

                self._sklearn_output_config["transform"] = transform

                return self


#### set_params

```python3
def set_params(
    self,
    **params
)
```

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as :class:`~sklearn.pipeline.Pipeline`). The latter have
parameters of the form ``<component>__<parameter>`` so that it's
possible to update each component of a nested object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| **params | dict | Estimator parameters. | None |

**Returns:**

| Type | Description |
|---|---|
| estimator instance | Estimator instance. |

??? example "View Source"
            def set_params(self, **params):

                """Set the parameters of this estimator.

                The method works on simple estimators as well as on nested objects

                (such as :class:`~sklearn.pipeline.Pipeline`). The latter have

                parameters of the form ``<component>__<parameter>`` so that it's

                possible to update each component of a nested object.

                Parameters

                ----------

                **params : dict

                    Estimator parameters.

                Returns

                -------

                self : estimator instance

                    Estimator instance.

                """

                if not params:

                    # Simple optimization to gain speed (inspect is slow)

                    return self

                valid_params = self.get_params(deep=True)

                nested_params = defaultdict(dict)  # grouped by prefix

                for key, value in params.items():

                    key, delim, sub_key = key.partition("__")

                    if key not in valid_params:

                        local_valid_params = self._get_param_names()

                        raise ValueError(

                            f"Invalid parameter {key!r} for estimator {self}. "

                            f"Valid parameters are: {local_valid_params!r}."

                        )

                    if delim:

                        nested_params[key][sub_key] = value

                    else:

                        setattr(self, key, value)

                        valid_params[key] = value

                for key, sub_params in nested_params.items():

                    valid_params[key].set_params(**sub_params)

                return self


#### transform

```python3
def transform(
    self,
    X
) -> numpy.ndarray
```

??? example "View Source"
            def transform(self, X) -> np.ndarray:

                log.info("Transforming SMILES data to RDKit descriptors .....")

                if len(X.shape) == 1:

                    fps = get_rdkit_descriptors(smiles=X, return_np=True)

                    return fps

                else:

                    fps = get_rdkit_descriptors(smiles=X[self.smiles_col], return_np=True)

                    return fps
