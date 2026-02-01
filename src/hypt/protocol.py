import warnings
from typing import Any, FrozenSet, Optional, Protocol

from numpy.typing import ArrayLike


class Search(Protocol):
    static: dict
    dynamic: dict

    @property
    def dynamic_keys(self) -> FrozenSet:
        return self.dynamic.keys()
    
    # This should only be required for dynamic searches but it's
    # better to have it everywhere and ignore it by default
    # for static searches.
    def feedback(self, value: Any): ...
        

class StaticSearch(Search, Protocol):
    # redefine Sequence protocol to support python<3.12
    def __getitem__(self, index: int) -> dict: ...
    def __len__(self) -> int: ...

    # default implementation does nothing
    def feedback(self, value: Any):
        return


class DynamicSearch(Search, Protocol):
    def __iter__(self): ...


_ParamStorage = dict[str, Any]


def _dict_length(d: dict[str, ArrayLike]):
    return next(iter(d.values())).shape[0]


class StaticParams(StaticSearch):
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
