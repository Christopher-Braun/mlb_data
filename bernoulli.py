import numpy as np
import pandas as pd
from sklearn.neural_network import BernoulliRBM


X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])
X = X.drop('Expected_Runs', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

def eval_metrics(y_test, y_pred):
    mae = np.round(abs(y_test.values - y_pred).mean(),decimals=2)
    rmse = (((y_test.values - y_pred)**2).mean())**(1/2)
    r = (((y_test.values-y_test.values.mean())*(y_pred - y_pred.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred.std()))
    print('mae: {0:.2f} \nrmse: {1:.2f} \nr: {2:.4f}'.format(mae, rmse, r))
    return mae, rmse, r


model = BernoulliRBM(n_components=94)
model.fit(X)
BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=20, n_iter=10,
       random_state=None, verbose=0)

trans = model.transform(X)
score = model.score_samples(X)




