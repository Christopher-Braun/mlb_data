"""
=========================================================================
Comparing randomized search and grid search for hyperparameter estimation
=========================================================================

Compare randomized search and grid search for optimizing hyperparameters of a
random forest.
All parameters that influence the learning are searched simultaneously
(except for the number of estimators, which poses a time / quality tradeoff).

The randomized search and the grid search explore exactly the same space of
parameters. The result in parameter settings is quite similar, while the run
time for randomized search is drastically lower.

The performance is slightly worse for the randomized search, though this
is most likely a noise effect and would not carry over to a held-out test set.

Note that in practice, one would not search over this many different parameters
simultaneously using grid search, but pick only the ones deemed most important.
"""
print(__doc__)

import numpy as np
import pandas as pd
from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestRegressor

# get some data
X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])
X2 = X['Expected_Runs']
X = X.drop('Expected_Runs', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#X, y = digits.data, digits.target

# build a classifier
clf = RandomForestRegressor(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [4, None],
              "max_features": sp_randint(1, 90),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["mse", "mae"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report1 = report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [10, None],
              "max_features": [1, 3, 87],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["mse", "mae"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)



'''
Model with rank: 1
Mean validation score: 0.045 (std: 0.012)
Parameters: {'bootstrap': False, 'criterion': 'mse', 'max_depth': 4, 'max_features': 7, 'min_samples_leaf': 10, 'min_samples_split': 5}

Model with rank: 2
Mean validation score: 0.037 (std: 0.011)
Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 4, 'max_features': 18, 'min_samples_leaf': 9, 'min_samples_split': 9}

Model with rank: 3
Mean validation score: 0.025 (std: 0.021)
Parameters: {'bootstrap': True, 'criterion': 'mae', 'max_depth': 4, 'max_features': 82, 'min_samples_leaf': 9, 'min_samples_split': 3}

Model with rank: 1
Mean validation score: 0.046 (std: 0.012)
Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 10, 'max_features': 1, 'min_samples_leaf': 10, 'min_samples_split': 10}

Model with rank: 2
Mean validation score: 0.037 (std: 0.012)
Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': None, 'max_features': 1, 'min_samples_leaf': 10, 'min_samples_split': 10}

Model with rank: 3
Mean validation score: 0.035 (std: 0.009)
Parameters: {'bootstrap': True, 'criterion': 'mse', 'max_depth': 10, 'max_features': 1, 'min_samples_leaf': 10, 'min_samples_split': 2}
'''
