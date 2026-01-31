from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike

from hypt.distributions import RandomSearch
from hypt.protocol import StaticParams


class GridSearch(StaticParams):
    """Defines parameters for a Grid Search.

    Forms a grid of parameters given a list of values for each parameter.

    Args:
        space (dict[str, Union[Any, ArrayLike]]): Values to try for each parameter.

    Returns:
        Iterable over parameters.
    """

    def __init__(self, space: dict[str, Union[Any, ArrayLike]]):
        static = {k: v for k, v in space.items() if np.isscalar(v)}
        dynamic = {k: v for k, v in space.items() if not np.isscalar(v)}
        dynamic_values = [v.flatten()
                          for v in np.meshgrid(*dynamic.values(), indexing='ij')]
        dynamic = dict(zip(dynamic.keys(), dynamic_values))
        size = dynamic_values[0].size
        super().__init__(dynamic, static, size)
