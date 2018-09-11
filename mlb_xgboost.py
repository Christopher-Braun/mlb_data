import os

mingw_path = 'C:\Program Files\mingw-w64\x86_64-7.1.0-posix-seh-rt_v5-rev0\mingw64\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

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

#y = pd.read_csv("C:/Users/mrcrb/source/repos/MLB/scores.csv", encoding='latin-1', names=['Score'])
#X = pd.read_csv("C:/Users/mrcrb/source/repos/MLB/pitch_trial.csv", encoding='latin-1')

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])



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

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

feature_importance = classifier.feature_importances_
features = pd.Series(feature_importance)

features.sort_values()
features_index_sort = features.index()

X_col = list(X.columns)

