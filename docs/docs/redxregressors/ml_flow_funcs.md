# Module redxregressors.ml_flow_funcs

Module of utilities for the redxregressors package working with mlflow

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module of utilities for the redxregressors package working with mlflow

        """

        import mlflow

        import logging

        from typing import Union, Optional

        from redxregressors import utilities

        log = logging.getLogger(__name__)



        def mlflow_get_or_create_experiment(name: Union[str, int]) -> int:

            """

            Function to get or create an experiment in MLFlow by name

            Args:

                experiment_name (str): the name of the experiment

            Returns:

                int: the experiment id

            """

            exp = mlflow.get_experiment_by_name(name)

            log.debug(f"Experiment {name} exists?: {exp}")

            if exp is not None:

                return exp.experiment_id

            else:

                return mlflow.create_experiment(name)



        def setup_for_mlflow(experiment_name: str, tracking_uri: Optional[str] = None) -> int:

            """

            Function to setup the mlflow experiment

            Args:

                experiment_name (str): the name of the experiment

                tracking_uri (str): the URI of the tracking server

            Returns:

                int: the experiment id

            """

            if tracking_uri is not None:

                log.info(f"Setting up MLFlow with tracking URI: {tracking_uri}")

                mlflow.set_tracking_uri(uri=tracking_uri)

            else:

                log.info(f"Setting up MLFlow with tracking local URI: {tracking_uri}")

                mlflow.set_tracking_uri(uri=utilities.mlflow_local_uri)

            return mlflow_get_or_create_experiment(experiment_name)



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### mlflow_get_or_create_experiment

```python3
def mlflow_get_or_create_experiment(
    name: Union[str, int]
) -> int
```

Function to get or create an experiment in MLFlow by name

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| experiment_name | str | the name of the experiment | None |

**Returns:**

| Type | Description |
|---|---|
| int | the experiment id |

??? example "View Source"
        def mlflow_get_or_create_experiment(name: Union[str, int]) -> int:

            """

            Function to get or create an experiment in MLFlow by name

            Args:

                experiment_name (str): the name of the experiment

            Returns:

                int: the experiment id

            """

            exp = mlflow.get_experiment_by_name(name)

            log.debug(f"Experiment {name} exists?: {exp}")

            if exp is not None:

                return exp.experiment_id

            else:

                return mlflow.create_experiment(name)


### setup_for_mlflow

```python3
def setup_for_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None
) -> int
```

Function to setup the mlflow experiment

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| experiment_name | str | the name of the experiment | None |
| tracking_uri | str | the URI of the tracking server | None |

**Returns:**

| Type | Description |
|---|---|
| int | the experiment id |

??? example "View Source"
        def setup_for_mlflow(experiment_name: str, tracking_uri: Optional[str] = None) -> int:

            """

            Function to setup the mlflow experiment

            Args:

                experiment_name (str): the name of the experiment

                tracking_uri (str): the URI of the tracking server

            Returns:

                int: the experiment id

            """

            if tracking_uri is not None:

                log.info(f"Setting up MLFlow with tracking URI: {tracking_uri}")

                mlflow.set_tracking_uri(uri=tracking_uri)

            else:

                log.info(f"Setting up MLFlow with tracking local URI: {tracking_uri}")

                mlflow.set_tracking_uri(uri=utilities.mlflow_local_uri)

            return mlflow_get_or_create_experiment(experiment_name)
