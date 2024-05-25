# -*- coding: utf-8 -*-
import numpy as np

from numpy.typing import ArrayLike
from typing import Any, Optional, Union
import warnings

_ParamStorage = dict[str, Any]


def _dict_length(d: dict[str, ArrayLike]):
    return next(iter(d.values())).shape[0]


class StaticParams:
    """Represents an array of static parameters generated a priori.

    Allows indexing or iteration, producing a dictionary of parameter values.

    Args:
        dynamic (_ParamStorage): Parameters that change from trial to trial.
        static (_ParamStorage, optional): Parameters held fixed. Defaults to {}.
        size (Optional[int], optional): Total number of parameter values. Defaults to None (inferred).
    """

    def __init__(self, dynamic: _ParamStorage, static: _ParamStorage = {}, size: Optional[int] = None) -> None:
        if len(dynamic) > 0:
            self.size = _dict_length(dynamic)
            if size is not None:
                assert size == self.size, "Provided size doesn't match inferred size from dynamic params."
        else:
            assert size is not None, "No dynamic parameters and no size specified."
            warnings.warn("No dynamic parameters found. Static parameters will be repeated `size` times.")
            self.size = size

        self.dynamic = dynamic
        self.static = static

    def __getitem__(self, i: int):
        params = {k: v[i] for k, v in self.dynamic.items()}
        params.update(self.static)
        return params

    def __len__(self):
        return self.size

    def to_pandas(self, with_static: bool = False):
        """Converts set of parameters to a pandas DataFrame.

        Args:
            with_static (bool, optional): Include static parameters. Defaults to False.

        Returns:
            DataFrame: Pandas DataFrame with each parameter as a column.
        """
        import pandas as pd

        df = pd.DataFrame(self.dynamic)
        if not with_static:
            return df

        for col, val in self.static.items():
            df[col] = val
        return df


class GridSearch(StaticParams):
    """Defines parameters for a Grid Search.

    Forms a grid of parameters given a list of values for each parameter.

    Args:
        space (dict[str, Union[Any, ArrayLike]]): Values to try for each parameter.

    Returns:
        StaticParams: Iterable over parameters.
    """

    def __init__(self, space: dict[str, Union[Any, ArrayLike]]):
        static = {k: v for k, v in space.items() if np.isscalar(v)}
        dynamic = {k: v for k, v in space.items() if not np.isscalar(v)}
        dynamic_values = [v.flatten()
                          for v in np.meshgrid(*dynamic.values())]
        dynamic = dict(zip(dynamic.keys(), dynamic_values))
        size = dynamic_values[0].size
        super().__init__(dynamic, static, size)
