"""
=====================================================
Prediction Intervals for Gradient Boosting Regression
=====================================================

This example shows how quantile regression can be used
to create prediction intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(1)

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])
X2 = X['Expected_Runs']
X = X.drop('Expected_Runs', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#----------------------------------------------------------------------


# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
xx = xx.astype(np.float32)

alpha = 0.95

clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                n_estimators=250, max_depth=3,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)

clf.fit(X_train, y_train)

# Make the prediction on the meshed x-axis
y_upper = clf.predict(X_test)

clf.set_params(alpha=1.0 - alpha)
clf.fit(X_train, y_train)

# Make the prediction on the meshed x-axis
y_lower = clf.predict(X_test)

clf.set_params(loss='ls')
clf.fit(X_train, y_train)

# Make the prediction on the meshed x-axis
y_pred = clf.predict(X_test)

# Plot the function, the prediction and the 90% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(X_train, y_train, 'g:') #label=u'$y_test$')
plt.plot(X_test, y_test, 'b.', markersize=10) #label=u'Observations')
plt.plot(X_test, y_pred, 'r-') #label=u'Prediction')
plt.plot(X_test, y_upper, 'k-')
plt.plot(X_test, y_lower, 'k-')
plt.fill(np.concatenate([X_test, X_test[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.5, fc='b', ec='None') #label='90% prediction interval')
plt.xlabel('$x_test$')
plt.ylabel('$y_test$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()
