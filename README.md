# hypt
Simple hyperparameter tuning in Python.

`hypt`'s design philosophy is:
* `hypt` doesn't take over your script. You own the training loop, `hypt` only provides parameter values to test and stays out of the way. Model training doesn't have to be relegated to some callback that you provide to an `optimize` function, making debugging more cumbersome.
* `hypt` will have a small footprint, designed to be composable with other libraries rather than integrate them. `hypt` will not implement things like experiment tracking, results vizualization, parallelization, etc.. 

## Installation

`hypt` can be installed through pip:
```
pip install hypt
```

## Getting Started

The following is an illustrative example of tuning the parameters of a GBDT model using random search:
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm
import hypt
import hypt.random as r

num_samples = 50

# define random search
hparams = hypt.RandomSearch({
    'loss': 'squared_error',
    'learning_rate': r.LogUniform(0.001, 0.5),
    'max_iter': 200,
    'max_leaf_nodes': r.IntLogUniform(16, 256),
    'min_samples_leaf': r.IntLogUniform(1, 100),
    'l2_regularization': r.OrZero(r.LogUniform(0.001, 10)),  # half of samples will be 0
    'validation_fraction': 0.2,
    'n_iter_no_change': 10,
    'random_state': 1984,
}, num_samples, seed=123)

# get data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hpt loop
val_results = []
test_results = []
for hparam in tqdm(hparams): # progress bar
    gbm = HistGradientBoostingRegressor(**hparam) # hparam is a simple dict
    gbm.fit(X_train, y_train)

    val_results.append(gbm.validation_score_.max())
    test_results.append(gbm.score(X_test, y_test))

# print best hparam and test score
best = np.argmax(val_results)
print('Best params:')
for k, v in hparams[best].items():
    print(f'\t{k} : {v}')
print('Test r2 score:', test_results[best])
```

Outputs:
```
Best params:
	learning_rate : 0.16311465153429475
	max_leaf_nodes : 33
	min_samples_leaf : 23
	l2_regularization : 0.06800582912648896
	loss : squared_error
	max_iter : 200
	validation_fraction : 0.2
	n_iter_no_change : 10
	random_state : 1984
Test r2 score: 0.8447968218784379
```

For the moment, only static strategies where all parameters can be generated a priori (`GridSearch` and `RandomSearch`) are implemented.
In the future we will also implement strategies that require incorporating feedback of function values to generate new test points like TPE and others.


## Parallelization

In many instances parallelization over hyperparameter tuning trials is not helpful since the model training itself will already make use of multithreading or parallelism. However in cases where it doesn't, it is again trivial to integrate `hypt` with, for example, `joblib` when using a static strategy like `GridSearch` or `RandomSearch`:

```python
import numpy as np
import hypt
from joblib import Parallel, delayed

# Himmelblau's function
def loss(x1, x2):
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# define grid search
hparams = hypt.GridSearch({
    'x1': np.linspace(-5, 5, 101),
    'x2': np.linspace(-5, 5, 101),
})

fvals = Parallel(n_jobs=2)(delayed(loss)(**hparam) for hparam in hparams)
best = np.argmin(fvals)

print(f'Best value found was {fvals[best]} at point {hparams[best]}')
```

This is of course not the best way to run grid search on such a simple function. It is only meant as an example.