import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from numpy import linalg
from numpy.linalg import norm
import math
import xgboost
import tensorflow as tf
import pydot


import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

y = pd.read_csv("MLB/scores.csv", encoding='latin-1', names=['Score'])
X = pd.read_csv("MLB/pitch_trial.csv", encoding='latin-1')

#from PythonScripts.MLB import mlb_dataframes
#X = mlb_dataframes.pitch_trial

#from PythonScripts.MLB import mlb_dataframes
#y = mlb_dataframes.y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
regressor.add(Dense(output_dim = 70, init = 'uniform', activation = 'relu', input_dim = len(X.columns)))
regressor.add(Dropout(p=0.1))

# Adding the second hidden layer
regressor.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
regressor.add(Dropout(p=0.1))

# Adding the third hidden layer
regressor.add(Dense(output_dim = 35, init = 'uniform', activation = 'relu'))
regressor.add(Dropout(p=0.1))

# Adding the 4th hidden layer
regressor.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
regressor.add(Dropout(p=0.1))

# Adding the output layer
regressor.add(Dense(output_dim = 1, init = 'uniform'))

# Compiling the ANN (adam[SGD] - optimizer function to find optimal weights)
# Binary Dept Var(Binary_CrossEntropy) Dependent Var > 2 Outcomes (Categorical_CrossEntropy)
# [Accuracy] in brackets because list expected
regressor.compile(optimizer = 'adam', loss = 'mse', metrics=['categorical_accuracy','acc'])

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 35, epochs = 2000)

# Predicting the Test set results
y_pred_test = regressor.predict(X_test)
y_pred_test_rd = np.round(y_pred_test)

# Predicting a new result
y_pred_train = regressor.predict(X_train)
y_pred_train_rd = np.round(y_pred_train)
y_score_train = regressor.evaluate(X_train, y_train, batch_size = 10)
y_score_test = regressor.evaluate(X_test, y_test, batch_size = 10)

y_train_zero = [0.1 if x==0 else x for x in y_train]
y_train_zero = pd.DataFrame(y_train_zero)

y_test_zero = [0.1 if x==0 else x for x in y_test]
y_test_zero = pd.DataFrame(y_test_zero)

#from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
plot_model(regressor, to_file='model.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(regressor).create(prog='dot', format='svg'))



# list all data in history
print(history.history.keys())



score = 1-abs((y_test_zero-y_pred_test_rd)/y_test_zero)
score_train = 1-abs((y_train_zero-y_pred_train)/y_train_zero)

accuracy = score.mean()
accuracy_train = score_train.mean()

i=0
for a,b in zip(y_test_zero, y_pred_test_rd):
    if a == abs(b):
        i+=1
print(i)