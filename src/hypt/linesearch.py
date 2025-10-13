import operator

import numpy as np


def line_search(points, direction='min'):
    if direction=='min':
        agg, best = min, np.inf
    elif direction=='max':
        agg, best = max, -np.inf
    else:
        raise ValueError(f'Only "max" and "min" are valid values for direction. Got {direction}.')
        
    for point in points:
        val = yield point
        if val is not None:
            best = agg(val, best)
    
    return best
    

def early_stopping_line_search(points, patience=1, direction='min'):
    if direction=='min':
        comparison = operator.gt
        best = np.inf
        fails = 0
    elif direction=='max':
        comparison = operator.lt
        best = -np.inf
        fails = 0
    else:
        raise ValueError(f'Only "max" and "min" are valid values for direction. Got {direction}.')
        
    for point in points:
        val = yield point
        if comparison(val, best):
            fails += 1
            if fails >= patience:
                break
        else:
            fails = 0
            best = val
    
    return best

        
def golden_search(a, b, num_evals=5, direction='min'):
    if direction=='min':
        comparison = operator.lt
    elif direction=='max':
        comparison = operator.gt
    else:
        raise ValueError(f'Only "max" and "min" are valid values for direction. Got {direction}.')
    
    r = (np.sqrt(5) - 1)/2

    a, b = min(a, b), max(a, b)
    h = b - a
    
    c = b - r*h
    d = a + r*h
    
    yc = yield c
    yd = yield d
    
    for k in range(num_evals - 2):
        h *= r
        if comparison(yc, yd):
            b, d = d, c
            c = b - r*h
            yd = yc
            yc = yield c
        else:
            a, c = c, d
            d = a + r*h
            yc = yd
            yd = yield d
            
    return min(yc, yd) if direction == 'min' else max(yc, yd)


def in_log_space(search):
    def generator(*args, **kwargs):
        gen = search(*[np.log(arg) for arg in args], **kwargs)
            
        try:
            value = yield np.exp(next(gen))
            while True:
                value = yield np.exp(gen.send(value))
        except StopIteration as e:
            return e.value
    return generator

    
class ForParam:
    def __init__(self, generator_function, *args, **kwargs):
        self.generator_function = generator_function
        self.args = args
        self.kwargs = kwargs
        
        
    def fval(self, val):
        self.value = val
    
    def __next__(self):
        try:
            if self.value is None:
                return next(self.generator)
            
            return self.generator.send(self.value)
        except StopIteration as e: # intercept best value
            self.best = e.value
            raise StopIteration
        
    
    def __iter__(self):
        self.value = None
        self.best = None
        self.generator = self.generator_function(*self.args, **self.kwargs)
        return self
    
    
class LineSearch(ForParam):
    def __init__(self, points, direction='min'):
        return super().__init__(line_search, points, direction=direction)


class EarlyStoppingLineSearch(ForParam):
    def __init__(self, points, patience=1, direction='min'):
        return super().__init__(early_stopping_line_search, points, patience=patience, direction=direction)
    

class GoldenSearch(ForParam):
    def __init__(self, a, b, num_evals=5, log=False, direction='min'):
        generator = golden_search
        if log:
            generator = in_log_space(generator)
        
        return super().__init__(generator, a, b, num_evals=num_evals, direction=direction)