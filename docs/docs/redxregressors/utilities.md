# Module redxregressors.utilities

Module of utilities

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module of utilities

        """

        from typing import List, Union

        from rdkit import Chem

        from copy import deepcopy

        import torch

        import numpy as np

        import random

        import tensorflow as tf

        import keras

        import logging

        log = logging.getLogger(__name__)

        random_seed = 15751

        mlflow_local_uri = "http://localhost:8080"

        mlflow_server_start_up_command = "mlflow server --host 127.0.0.1 --port 8080"



        def prepend_dictionary_keys(d: dict, prepend: str) -> dict:

            """

            Function to prepend the same string to the keys in a dictionary. This is mainly for working with Scikit learn Pipelines, where we need to append eg "model__" to parameters for the model.

            Args:

                d (dict): The dictionary to prepend to the key of. We work on a copy of this so the origianl is not changed (mutable type)

                prepend (str): The string to prepend to all the keys

            Returns:

                dict: _description_

            """

            d_tmp = deepcopy(d)

            return {f"{prepend}{k}": v for k, v in d_tmp.items()}



        def validate_smile(

            smile: str, canonicalize: bool = True, return_mol: bool = False

        ) -> Union[None, str]:

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

            if return_mol is True:

                return m

            elif canonicalize is True:

                return Chem.MolToSmiles(m)

            else:

                return smile



        def validate_smiles(

            smiles: List[str],

            return_failed_as_None: bool = True,

            canonicalize: bool = True,

            return_mol: bool = False,

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

                return [

                    validate_smile(smile, canonicalize=canonicalize, return_mol=return_mol)

                    for smile in smiles

                ]

            else:

                tmp = [

                    validate_smile(smile, canonicalize=canonicalize, return_mol=return_mol)

                    for smile in smiles

                ]

                return [ent for ent in tmp if ent is not None]



        def seed_worker(worker_id):

            """

            Function from pytorch documentation to make the data loader reproducible

            https://pytorch.org/docs/stable/notes/randomness.html accessed 24/10/24

            """

            worker_seed = torch.initial_seed() % 2**32

            np.random.seed(worker_seed)

            random.seed(worker_seed)



        def seed_all() -> None:

            """

            Function to set the random seed for all the libraries used in the project

            """

            np.random.seed(random_seed)

            tf.random.set_seed(random_seed)

            random.seed(random_seed)

            torch.manual_seed(random_seed)

            keras.utils.set_random_seed(random_seed)

## Variables

```python3
log
```

```python3
mlflow_local_uri
```

```python3
mlflow_server_start_up_command
```

```python3
random_seed
```

## Functions


### prepend_dictionary_keys

```python3
def prepend_dictionary_keys(
    d: dict,
    prepend: str
) -> dict
```

Function to prepend the same string to the keys in a dictionary. This is mainly for working with Scikit learn Pipelines, where we need to append eg "model__" to parameters for the model.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| d | dict | The dictionary to prepend to the key of. We work on a copy of this so the origianl is not changed (mutable type) | None |
| prepend | str | The string to prepend to all the keys | None |

**Returns:**

| Type | Description |
|---|---|
| dict | _description_ |

??? example "View Source"
        def prepend_dictionary_keys(d: dict, prepend: str) -> dict:

            """

            Function to prepend the same string to the keys in a dictionary. This is mainly for working with Scikit learn Pipelines, where we need to append eg "model__" to parameters for the model.

            Args:

                d (dict): The dictionary to prepend to the key of. We work on a copy of this so the origianl is not changed (mutable type)

                prepend (str): The string to prepend to all the keys

            Returns:

                dict: _description_

            """

            d_tmp = deepcopy(d)

            return {f"{prepend}{k}": v for k, v in d_tmp.items()}


### seed_all

```python3
def seed_all(

) -> None
```

Function to set the random seed for all the libraries used in the project

??? example "View Source"
        def seed_all() -> None:

            """

            Function to set the random seed for all the libraries used in the project

            """

            np.random.seed(random_seed)

            tf.random.set_seed(random_seed)

            random.seed(random_seed)

            torch.manual_seed(random_seed)

            keras.utils.set_random_seed(random_seed)


### seed_worker

```python3
def seed_worker(
    worker_id
)
```

Function from pytorch documentation to make the data loader reproducible

https://pytorch.org/docs/stable/notes/randomness.html accessed 24/10/24

??? example "View Source"
        def seed_worker(worker_id):

            """

            Function from pytorch documentation to make the data loader reproducible

            https://pytorch.org/docs/stable/notes/randomness.html accessed 24/10/24

            """

            worker_seed = torch.initial_seed() % 2**32

            np.random.seed(worker_seed)

            random.seed(worker_seed)


### validate_smile

```python3
def validate_smile(
    smile: str,
    canonicalize: bool = True,
    return_mol: bool = False
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
        def validate_smile(

            smile: str, canonicalize: bool = True, return_mol: bool = False

        ) -> Union[None, str]:

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

            if return_mol is True:

                return m

            elif canonicalize is True:

                return Chem.MolToSmiles(m)

            else:

                return smile


### validate_smiles

```python3
def validate_smiles(
    smiles: List[str],
    return_failed_as_None: bool = True,
    canonicalize: bool = True,
    return_mol: bool = False
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

            smiles: List[str],

            return_failed_as_None: bool = True,

            canonicalize: bool = True,

            return_mol: bool = False,

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

                return [

                    validate_smile(smile, canonicalize=canonicalize, return_mol=return_mol)

                    for smile in smiles

                ]

            else:

                tmp = [

                    validate_smile(smile, canonicalize=canonicalize, return_mol=return_mol)

                    for smile in smiles

                ]

                return [ent for ent in tmp if ent is not None]
