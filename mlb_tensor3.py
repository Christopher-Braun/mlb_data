import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from numpy import linalg
from numpy.linalg import norm
import math
import xgboost
import tensorflow as tf

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

#y = pd.read_csv("MLB/scores1.csv", encoding='latin-1', names=['Score'])
#X = pd.read_csv("MLB/slim_pitch_trial.csv", encoding='latin-1')

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])
X = X.drop('Expected_Runs', axis = 1)

from constant_variables import features_top_list
X2 = X[features_top_list]
X3 = X2.drop('Expected_Runs', axis = 1)


#from PythonScripts.MLB import mlb_dataframes
#X = mlb_dataframes.pitch_trial

#from PythonScripts.MLB import mlb_dataframes
#y = mlb_dataframes.y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size = 0.25, random_state = 0)

# Feature Scaling (Important for high intensity computations)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras import objectives
from keras import backend as K

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer with dropout (start 0.1 (10%))
regressor.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu', input_dim = len(X.columns)))
regressor.add(Dropout(p=0.1))

# Adding the second hidden layer
regressor.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
regressor.add(Dropout(p=0.1))

# Adding the second hidden layer
regressor.add(Dense(output_dim = 35, init = 'uniform', activation = 'relu'))
regressor.add(Dropout(p=0.1))

# Adding the third hidden layer
regressor.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
regressor.add(Dropout(p=0.1))

# Adding the output layer
regressor.add(Dense(output_dim = 1, init = 'uniform'))

# Compiling the ANN (adam[SGD] - optimizer function to find optimal weights)
# Binary Dept Var(Binary_CrossEntropy) Dependent Var > 2 Outcomes (Categorical_CrossEntropy)
# [Accuracy] in brackets because list expected
regressor.compile(optimizer = 'adam', loss = 'mse', metrics=['binary_crossentropy','acc'])

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 35, epochs = 500)

# Predicting the Test set results
y_pred_test = regressor.predict(X_test)
y_pred_test_rd = np.round(y_pred_test)
y_test_index = [x for x in y_test.index]

# Predicting a new result
y_pred_train = regressor.predict(X_train)
y_pred_train_rd = np.round(y_pred_train)
y_train_index = [x for x in y_train.index]

expected = X['Expected_Runs']
y_expected_test = expected[y_test_index]


# Predicting the Full Results
y_pred_full = regressor.predict(X3)
y_pred_full_rd = np.round(y_pred_full)

# Predicting a new result
y_pred_train = regressor.predict(X_train)
y_pred_train_rd = np.round(y_pred_train)
y_score_train = regressor.evaluate(X_train, y_train, batch_size = 10)
y_score_test = regressor.evaluate(X_test, y_test, batch_size = 10)

y_train_zero = [0.1 if x==0 else x for x in y_train]
y_train_zero = pd.DataFrame(y_train_zero)

y_test_zero = [0.1 if x==0 else x for x in y_test]
y_test_zero = pd.DataFrame(y_test_zero)

