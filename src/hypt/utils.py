from collections import defaultdict
from typing import ClassVar

import numpy as np

from hypt.protocol import DynamicSearch


class Recorder(DynamicSearch):
    objective_values_key: ClassVar[str] = "objective_values"

    def __init__(self, params_iterator: DynamicSearch):
        self.params_iterator = params_iterator
        self.params = None
        self.values = None

    @property
    def static(self):
        return self.params_iterator.static
    
    @property
    def dynamic(self):
        return self.params_iterator.dynamic
    
    def __iter__(self):
        self.params = defaultdict(list)
        self.values = []

        for param in self.params_iterator:
            for k in self.dynamic_keys:
                self.params[k].append(param[k])
            self.values.append(None) # placeholder
            yield param
    
    def feedback(self, value):
        self.values[-1] = value
        return self.params_iterator.feedback(value)
    
    def to_pandas(self, with_values: bool = True, with_static: bool = False):
        import pandas as pd

        df = self.params
        if with_values:
            df[self.objective_values_key] = self.values 
        df = pd.DataFrame(df)

        if not with_static:
            return df

        for col, val in self.static.items():
            df[col] = val
        return df
    
    def best_value(self, direction='min'):
        if direction.lower() == 'min':
            return min(self.values)
        elif direction.lower() == 'max':
            return max(self.values)
        raise ValueError(
            f"Expected direction to be 'max' or 'min'. Got '{direction}'."
        )
    
    def best_iteration(self, direction='min'):
        if direction.lower() == 'min':
            return np.argmin(self.values)
        elif direction.lower() == 'max':
            return np.argmax(self.values)
        else:
            raise ValueError(
                f"Expected direction to be 'max' or 'min'. Got '{direction}'."
            )
    
    def best_params(self, direction='min'):
        best = self.best_iteration(direction=direction)
        params = {k: v[best] for k, v in self.params.items()}
        params.update(self.static)
        return params
