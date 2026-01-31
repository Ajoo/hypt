# hypt
Simple hyperparameter tuning in Python.

I wrote `hypt` as a minimalistic hyperparameter tuning library I could use for quick experiments, when established libraries like [Optuna](https://optuna.org/) felt like overkill. My goal was to have it work as a simple and easy to debug for loop over hyperparameter values, instead of having to rewrite my whole training script around it.

As such, `hypt` will have a small footprint, and avoid implementing things like experiment tracking, results vizualization, parallelization, etc.. I will also probably focus more on "out of the beaten path" approaches to hyperparameter optimization.

## Installation

`hypt` can be installed through pip:
```
pip install hypt
```

## Getting Started with Random Search

The following is an illustrative example of tuning the parameters of a GBDT model using 50 trials of random search:
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm
import hypt
import hypt.random as r

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
}, num_samples=50, seed=123)

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
100%|██████████| 50/50 [01:27<00:00,  1.76s/it]
Best params:
	learning_rate : 0.16311465153429477
	max_leaf_nodes : 33
	min_samples_leaf : 23
	l2_regularization : 0.06800582912648902
	loss : squared_error
	max_iter : 200
	validation_fraction : 0.2
	n_iter_no_change : 10
	random_state : 1984
Test r2 score: 0.8447968218784379
```


## Informed Line Searches

Now lets try something a bit more unconventional, relying on my past experience with tuning GBDT models... We will restrict ourselves to tuning only 3 hyperparameters using something more similar to a grid search:
1. The total number of leaf nodes
2. The minimum number of samples per leaf 
3. The learning rate

The first two hyperparameters control the regularization of the model. We will always start our search with the most regularized version first (smaller number of leaf nodes and larger minimum samples per leaf). For each setting of the regularization parameters we will find the optimal learning rate using a [golden-section search](https://en.wikipedia.org/wiki/Golden-section_search) with 5 function evaluations. If we find that, as we reduce the amount of regularization provided by any of the parameters, the function value gets worse, we stop the search in that direction.

The code to implement this is the following:

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

import hypt
from hypt.linesearch import NestedLineSearch, GoldenSearch, LineSearch

# define the nested line search
# note that the order of the dynamic parameters matters!
# the first parameter corresponds to the outermost loop
hparams = NestedLineSearch({
    'loss': 'squared_error',
    'max_iter': 200,
    'validation_fraction': 0.2,
    'n_iter_no_change': 10,
    'random_state': 1984,
    'l2_regularization': 0,
    'max_leaf_nodes': LineSearch([16, 32, 64, 128, 256], patience=1),
    'min_samples_leaf': LineSearch([30, 10, 1], patience=1),
    'learning_rate': GoldenSearch(0.001, 0.5, num_evals=5, log=True),
})
# wrapper utility to automatically record objective values and parameters
hparams = hypt.Recorder(hparams) 

# get data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hpt loop is similar
val_results = []
test_results = []
for hparam in tqdm(hparams): # progress bar
    gbm = HistGradientBoostingRegressor(**hparam) # hparam is a simple dict
    gbm.fit(X_train, y_train)

    val_results.append(gbm.validation_score_.max())
    test_results.append(gbm.score(X_test, y_test))

    # this is the main change as we need to provide feedback
    # (i.e., the function value) for the search procedure
    hparams.feedback(-val_results[-1])

# print best hparam and test score
best = np.argmax(val_results)
print('Best params:')
for k, v in hparams.best_params().items():
    print(f'\t{k} : {v}')
print('Test r2 score:', test_results[hparams.best_iteration()])
```
which outputs:
```
30it [00:30,  1.01s/it]
Best params:
	max_leaf_nodes : 32
	min_samples_leaf : 30
	learning_rate : 0.1153000814478148
	loss : squared_error
	max_iter : 200
	validation_fraction : 0.2
	n_iter_no_change : 10
	random_state : 1984
	l2_regularization : 0
Test r2 score: 0.8449648371371464
```

We obtained a similar test $R^2$ score and found similar hyperparameters in 1/3 of the time! Note that this speedup was only possible due to the early stopping in the line searches, allowing us to evaluate only the three smaller values of `max_leaf_nodes`. We also only evaluated the first two values of `min_samples_leaf` for each value of `max_leaf_nodes`. A full grid search would have required evaluating 75 different hyperparameter configurations. 


## Future Developments

Eventually I hope to have the time to implement some more hyperparameter search methods. This could include the ever popular TPE but also other more unconventional local search approaches.